import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from exceptiongroup import catch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.transforms import NormalizeFeatures
from torchsort import soft_rank

# 自定义Mamba层核心实现
class MambaBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 参数化投影矩阵
        self.proj = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)

        # 选择性参数
        self.A = nn.Parameter(torch.ones(1, 1,hidden_dim))  # 修正形状
        self.B = nn.Parameter(torch.ones(1, hidden_dim, 1))  # 修正形状
        self.C = nn.Parameter(torch.ones(1, hidden_dim, 1))  # 修正形状
        self.D = nn.Parameter(torch.ones(1))

        # 离散化参数
        self.delta = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 初始化参数
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.normal_(self.B, mean=0.0, std=0.02)
        nn.init.normal_(self.C, mean=0.0, std=0.02)

    def discretization(self, delta, A, B):
        # 离散化过程
        # delta blh
        dA = torch.einsum('blh,bhn->bhl', torch.exp(delta * A), B)  #bhl = 1lh*11h  B;1h1修正einsum
        # dB = (delta.unsqueeze(-1) * B).cumsum(dim=1)
        dB = (delta.unsqueeze(-1) * B.unsqueeze(0)).cumsum(dim=1)  #  blh1 cosume 累计求和 along h
        dB = dB.permute(0, 2, 3,1)# bh1l
        return dA, dB

    def selective_scan(self, x, dA, dB, C):#bhl blh1 blh
        # 选择性扫描实现
        u = x.permute(0, 2, 1)  # [B, H, L]
        C = C.permute(0, 2, 1)
        # 创建卷积核
        kernel = torch.einsum('bhl,bhn->bhnl', dA, u) + dB
        y = torch.einsum('bhnl,bhn->bhl', kernel, C)
        # 万恶
        return y.permute(0, 2, 1)  # [B, L, H]

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = x.shape

        # 投影获取参数
        proj = self.proj(x)  # [B, L, 4H]
        delta, B, C, residual = torch.split(proj, self.hidden_dim, dim=-1)

        # 离散化过程
        delta = F.softplus(self.delta(delta))  # [B, L, H]
        dA, dB = self.discretization(delta, self.A, self.B) #bhl blh1

        # 选择性扫描
        y = self.selective_scan(x, dA, dB, C)# blh bhl blh1 blh

        # 残差连接
        output = y + residual * self.D
        return output


class MambaGNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_global_sort=True):
        super().__init__()

        self.gnn = SAGEConv(in_dim, out_dim)
        self.mamba = MambaBlock(out_dim)
        self.norm = nn.LayerNorm(out_dim)
        # GELU function for nonlinear
        self.act = nn.GELU()

        # 动态排序网络
        self.rank_net = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.use_global_sort = use_global_sort  # 控制是否全局排序

    def dynamic_sort(self, x, edge_idx, batch_mask=None):
        """动态排序核心逻辑
        Args:
            x: 节点特征 [num_nodes, hid_dim]
            batch_mask: 子图掩码（用于Reddit等大图）
        """
        # 计算重要性分数
        scores = self.rank_net(x).squeeze(-1)  # [num_nodes]
        # 添加度数重要性
        num_nodes = x.shape[0]
        degree = torch.zeros(num_nodes, device=x.device)

        # 计算每个节点的度数

        # max_idx = torch.max(edge_idx[0])
        # print("Max index in edge_idx[0]:", max_idx,"num_nodes",num_nodes)
        degree.index_add_(0, edge_idx[0], torch.ones(edge_idx[0].shape, device=x.device))  # 累加源节点的度数
        degree.index_add_(0, edge_idx[1], torch.ones(edge_idx[1].shape, device=x.device))  # 累加目标节点的度数


        scores = scores + degree.float()  # 假设直接加上度数
        scores = scores.unsqueeze(0)
        # 大图分批次排序
        # TODO 未测试，batch—loader give batch=1
        if batch_mask is not None and not self.use_global_sort:
            # Reddit等大图模式：逐子图排序
            sorted_x = []
            sorted_indices = []
            for b in torch.unique(batch_mask):
                mask = (batch_mask == b)
                sub_x = x[mask]
                sub_scores = scores[mask]
                sorted_idx = soft_rank(sub_scores, regularization='l2',
                                       regularization_strength=0.1).argsort(descending=True)
                sorted_x.append(sub_x[sorted_idx])
                sorted_indices.append(torch.where(mask)[0][sorted_idx])

            sorted_x = torch.cat(sorted_x)
            sorted_indices = torch.cat(sorted_indices)
            return sorted_x, sorted_indices


        # Cora/Citeseer小图模式：全局排序
        sorted_idx = soft_rank(scores, regularization='l2',
                               regularization_strength=0.1).argsort(descending=True)
        sorted_x = x[sorted_idx].squeeze(0)
        return sorted_x,sorted_idx




    def forward(self, x, edge_index, batch_mask=None):
        # GNN聚合
        x_gnn = self.gnn(x, edge_index) + x # resSage
        x_gnn = self.act(x_gnn)  # [num_nodes, hid_dim]


        # 动态排序
        x_sorted,sorted_idx = self.dynamic_sort(x_gnn,edge_index,batch_mask= batch_mask)
        x_sorted = x_sorted.unsqueeze(0) #1,2708,128
        # Mamba处理（保持batch维度）
        mamba_out = self.mamba(x_sorted)
        mamba_out = mamba_out.squeeze(0)
        mamba_out = F.layer_norm(mamba_out, (mamba_out.shape[-1],))
        # mamba_out = self.norm(mamba_out)
        mamba_out_original_order = mamba_out[sorted_idx.argsort()]
        x = x+mamba_out_original_order
        x = x.squeeze(0)
        # print(x.shape)
        # 残差连接 lh
        return x



# 完整模型架构（gnn+mamba）* Nblock
class IterativeMambaGNN(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(feat_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            MambaGNNBlock(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index):


        # 初始嵌入
        x = self.embed(x)
        # x  = x.unsqueeze(0)

        # 迭代处理
        for block in self.blocks:
            x = block(x, edge_index)

        return x


#  1.[resSage->ssm]*N
class IterativeMambaGNN2(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        # TODO 测试保留embed层的效果，先保留
        self.embed = nn.Linear(feat_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            MambaGNNBlock(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index):
        # 初始嵌入
        x = self.embed(x)
        # x  = x.unsqueeze(0)

        # 迭代处理
        for block in self.blocks:
            x = block(x, edge_index)
        x = self.classifier(x)

        return x

# 2.GNN * N +SSM
class IterativeMambaGNN2(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_classes, num_layers=2):
        super().__init__()
        # TODO 测试保留embed层的效果，先保留
        self.embed = nn.Linear(feat_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            MambaGNNBlock(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index):
        # 初始嵌入
        x = self.embed(x)
        # x  = x.unsqueeze(0)

        # 迭代处理
        for block in self.blocks:
            x = block(x, edge_index)
        x = self.classifier(x)

        return x

# 训练循环
# def train(data):
#     model.train()
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
#     return loss.item()
#
#
# # 验证函数
# @torch.no_grad()
# def test(data):
#     model.eval()
#     out = model(data)
#     pred = out.argmax(dim=1)
#
#     accs = []
#     for mask in [data.train_mask, data.val_mask, data.test_mask]:
#         acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
#         accs.append(acc)
#     return accs



# # 数据准备
# dataset = Planetoid(root='./data/Cora', name='Cora', transform=NormalizeFeatures())
# data = dataset[0]
# data.edge_index = torch_geometric.utils.add_self_loops(data.edge_index)[0]  # 添加自环
#
# # 模型初始化
# model = IterativeMambaGNN(
#     feat_dim=dataset.num_features,
#     hidden_dim=128,
#     num_classes=dataset.num_classes,
#     num_layers=3
# )
#
# # 训练配置
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)


# # 训练过程
# best_val_acc = 0
# for epoch in range(1, 5):
#     print("train---")
#     loss = train(data)
#     print("test---")
#     train_acc, val_acc, test_acc = test(data)
#
#     scheduler.step(val_acc)
#
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), 'best_model.pth')
#
#     print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
#           f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
#
# # 最终测试
# model.load_state_dict(torch.load('best_model.pth'))
# train_acc, val_acc, test_acc = test(data)
# print(f'\nFinal Test Accuracy: {test_acc:.4f}')