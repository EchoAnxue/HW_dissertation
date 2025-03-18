import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super(MLPModel, self).__init__()
        layers = []
        
        # 输入层
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 隐藏层（根据 num_layers 动态创建）
        for _ in range(num_layers - 1):  # 已经加了 1 层，因此 -1
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            

        # 输出层
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x,edge_index = None):
        return self.mlp(x)
