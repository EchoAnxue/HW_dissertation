import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv


# 特点 有mlp
class GINNodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_layers,dropout = 0.2):
        super(GINNodeClassifier, self).__init__()
        self.num_layers = num_layers

        # 创建 GINConv 层
        self.convs = torch.nn.ModuleList()

        # 第一层：输入维度 → 隐藏维度
        mlp = Sequential(Linear(in_channels, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(mlp,train_eps=True))
        self.dropout = dropout
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        # 中间层：隐藏维度 → 隐藏维度
        for _ in range(num_layers - 2):
            mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))

        # 最后一层：隐藏维度 → 输出维度（类别数）
        mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, out_channels))
        self.convs.append(GINConv(mlp,train_eps=True))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # 最后一层不加 ReLU
                x = self.bn(x)
                x = F.leaky_relu(x)
                x = F.dropout(x,p = self.dropout, training=self.training)
        return x  # 归一化分类输出

