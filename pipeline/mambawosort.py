'''
mamba->gnn
之前忘记mamba的输入其实只有一层
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from exceptiongroup import catch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, JumpingKnowledge
from torch_geometric.transforms import NormalizeFeatures
from torchsort import soft_rank

from resGCN import ResGCNLayer, ResConv


# 自定义Mamba层核心实现
class MambaBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 参数化投影矩阵
        self.proj = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)

        # 选择性参数
        self.A = nn.Parameter(torch.ones(hidden_dim, 1))  # 修正形状
        self.B = nn.Parameter(torch.ones(hidden_dim, 1))  # 修正形状
        self.C = nn.Parameter(torch.ones(1, hidden_dim, 1))  # 修正形状
        self.D = nn.Parameter(torch.ones(hidden_dim))

        # 离散化参数
        self.delta = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # 初始化参数
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        nn.init.normal_(self.B, mean=0.0, std=0.02)
        nn.init.normal_(self.C, mean=0.0, std=0.02)

    def discretization(self, delta, A, B, residual):  # blh,hn,
        # 离散化过程
        # delta blh
        dA = torch.einsum('blh,hn->blhn', delta, A)  # blhn = blh*1h1  B;1h1修正einsum
        # dB = (delta.unsqueeze(-1) * B).cumsum(dim=1)
        # deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        deltaB_u = torch.einsum('b l h, h n, b l h -> b l h n', delta, B, residual)
        # dB = (delta.unsqueeze(-1) * B.unsqueeze(0)).cumsum(dim=1)  #  blh1 = 1blh* 11h1 cosume 累计求和 along l
        # dB = dB.permute(0, 2, 3,1)# bh1l
        return dA, deltaB_u  # blhn blhn

    def selective_scan(self, dA, dB, C):  # blh blhn blhn blh
        # 选择性扫描实现
        # u = x.permute(0, 2, 1)  # [B, H, L]
        # C = C.permute(0, 2, 1)
        # # 创建卷积核
        # kernel = torch.einsum('bhl,bhn->bhnl', dA, u) + dB
        # y = torch.einsum('bhnl,bhn->bhl', kernel, C)
        # 万恶
        ys = []

        x = torch.zeros((1, 128, 1), device=dA.device)
        for i in range(dA.shape[1]):  # 这里使用for循环的方式只用来说明核心的逻辑，原代码中采用并行扫描算法
            x = dA[:, i] * x + dB[:, i]  # x(t + 1) = Ax(t) + Bu(t)
            y = torch.einsum('b h n, b h -> b h', x, C[:, i, :])  # y(t) = Cx(t)  (B,D,N)*(B,N)->(B,D)
            ys.append(y)
        y = torch.stack(ys, dim=1)  # 大小 (b, l, d_in)  (B,L,D)

        return y

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = x.shape

        # 投影获取参数
        proj = self.proj(x)  # [B, L, 4H]
        # 依赖于输入 residual = u
        delta, B, C, residual = torch.split(proj, self.hidden_dim, dim=-1)

        # 离散化过程
        delta = F.softplus(delta)  # [B, L, H]
        dA, dB = self.discretization(delta, self.A, self.B, residual)  # blhn blh1
        #  deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))

        # 选择性扫描
        y = self.selective_scan(dA, dB, C)  # blh bhl blh1 blh

        # 残差连接
        output = y + residual * self.D
        return output


class MambaGNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, num_layers=2, Mamba_layers=3, use_global_sort=True):
        super().__init__()

        self.JKNet = torch.nn.ModuleList([SAGEConv(hidden_dim, hidden_dim)] +
                                         [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.resNet = ResConv(in_dim, hidden_dim, out_dim, num_layers)
        self.jump_lstm = JumpingKnowledge(mode='lstm', channels=128, num_layers=1)
        self.jump_cat = JumpingKnowledge(mode='cat')
        self.lin_JKNet = torch.nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.gcn_mamba = GCNConv(in_dim, hidden_dim)
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        self.gnn = SAGEConv(hidden_dim, out_dim)
        self.mamba = torch.nn.ModuleList([MambaBlock(hidden_dim) for _ in range(Mamba_layers)])
        # self.mamba = MambaBlock(hidden_dim)
        self.mamba_2 = MambaBlock(hidden_dim)
        self.norm = nn.LayerNorm(out_dim)

        # GELU function for nonlinear
        self.act = nn.GELU()

        self.fusion_layer = nn.Linear(2 * out_dim, out_dim)

        # 动态排序网络
        self.rank_net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.use_global_sort = use_global_sort  # 控制是否全局排序
        # self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的融合参数

        self.alpha = 0

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

        # scores = scores + degree.float()  # 假设直接加上度数
        scores = degree.float()  # 假设直接加上度数
        scores = scores.unsqueeze(0)
        # 大图分批次排序
        # TODO 未测试
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
        return sorted_x, sorted_idx

    # def forward(self, x, edge_index, batch_mask=None):
    #     # 1433
    #     # GNN聚合
    #     out_list = []
    #
    #     # 动态排序 mamba
    #     x_sorted, sorted_idx = self.dynamic_sort(x, edge_index, batch_mask=batch_mask)
    #     # num_features->hidden
    #     x_sorted = self.gcn_mamba(x_sorted,edge_index)
    #     x_sorted = x_sorted.unsqueeze(0)  # 1,2708,128
    #     # Mamba处理（保持batch维度）  mamba block-> hidden (default)
    #     mamba_out = self.mamba(x_sorted)
    #     mamba_out = mamba_out.squeeze(0)
    #     mamba_out = F.layer_norm(mamba_out, (mamba_out.shape[-1],))
    #     # mamba_out = self.norm(mamba_out)
    #     mamba_out_original_order = mamba_out[sorted_idx.argsort()]
    #     x_mamba = x_sorted.squeeze(0) + mamba_out_original_order.squeeze(0)
    #
    #     # gnn
    #     x = self.gcn_mamba(x,edge_index)
    #     for i, conv in enumerate(self.JKNet):
    #         res = conv(x, edge_index)
    #
    #         res = F.leaky_relu(res)
    #         res = self.bn(res)
    #         res = F.dropout(res, p=0.2, training=self.training)  # Dropout
    #         x = x + res
    #
    #     # out = torch.cat(out_list, dim=1)  # [num_nodes, num_layers * hidden_channels]
    #     #
    #     x = x_mamba + x
    #
    #     return x

    def forward(self, x, edge_index, batch_mask=None):
        # 1433
        # GNN聚合
        out_list = []
        '''
        resGCN
        '''
        # 动态排序 mamba
        x_sorted, sorted_idx = self.dynamic_sort(x, edge_index, batch_mask=batch_mask)
        # num_features->hidden
        x_2_sorted = self.gcn_mamba(x, edge_index)
        x_sorted = x_2_sorted[sorted_idx]  # 1,2708,128
        # x_sorted = x_sorted.unsqueeze(0)  # 1,2708,128
        # Mamba处理（保持batch维度）  mamba block-> hidden (default)
        mamba_out = x_2_sorted.unsqueeze(0)
        for mamba in self.mamba:
            mamba_out = mamba(mamba_out)
            mamba_out = F.layer_norm(mamba_out, (mamba_out.shape[-1],))

        mamba_out = mamba_out.squeeze(0)
        # mamba_out = F.layer_norm(mamba_out, (mamba_out.shape[-1],))
        # mamba_out = self.norm(mamba_out)
        
        x_mamba = mamba_out + x_2_sorted
        # h->o
        x_mamba = self.gnn(x_mamba, edge_index)
        x_mamba = F.dropout(x_mamba, p=0.5, training=self.training)  # 添加 Dropout

        # gnn
        x_gnn = self.resNet(x, edge_index)

        # x_combined = self.alpha * x_mamba + (1 - self.alpha) * x_gnn
        x_combined = torch.cat([x_gnn, x_mamba], dim=-1)  # 拼接
        x_combined = self.fusion_layer(x_combined)  # 线性变换
        # out = torch.cat(out_list, dim=1)  # [num_nodes, num_layers * hidden_channels]
        #

        return x_combined


# 串联


# 2.GNN * N +SSM
class IterativeMambaGNN5(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_classes, num_layers):
        super().__init__()
        # TODO 测试保留embed层的效果，先保留

        self.blocks = MambaGNNBlock(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=num_classes,
                                    num_layers=num_layers, Mamba_layers=2)

    def forward(self, x, edge_index):
        # 初始嵌入
        # x = self.embed(x)

        # 迭代处理

        x = self.blocks(x, edge_index)

        return x

