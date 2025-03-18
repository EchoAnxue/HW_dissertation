import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv, GATConv, GINConv
from torch.nn import Sequential, Linear, ReLU
# class ResGCNLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResGCNLayer, self).__init__()
#         # self.conv = GINConv(in_channels, out_channels)
#         mlp = Sequential(Linear(in_channels, 256), ReLU(), Linear(256, out_channels))
#         self.conv = GINConv(mlp,train_eps=True)
#     def forward(self, x, edge_index):
#         #
#         residual = x
#         out = self.conv(x, edge_index)
      
        
#         return out + residual  #

# class ResGCNLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResGCNLayer, self).__init__()
#         self.conv = GATConv(in_channels, out_channels,heads=4,dropout=0.5)
#         self.fc = nn.Linear(out_channels * 4, in_channels)  
#     def forward(self, x, edge_index):
#         #
#         residual = x
#         out = self.conv(x, edge_index)
#         out = self.fc(out)
        
#         return out + residual  #


class ResGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResGCNLayer, self).__init__()
        self.conv = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        #
        residual = x
        out = self.conv(x, edge_index)

        
        return out + residual  #


class ResConv(nn.Module):
    def __init__(self, in_dim, hidden_dim,out_dim, num_layers,dropout=0.5):
        super(ResConv, self).__init__()
        self.resNet = torch.nn.ModuleList([GCNConv(in_dim, hidden_dim)] + [ResGCNLayer(hidden_dim, hidden_dim)
                                                                                  for _ in range(num_layers - 2)]
                                            + [GCNConv(out_dim, out_dim)])
        self.lin_resGCN = torch.nn.Linear(hidden_dim, out_dim)
        self.num_layers = num_layers
        self.dropout = dropout
    def forward(self, x, edge_index):
      for i, conv in enumerate(self.resNet):
          x = conv(x, edge_index)
          if i == self.num_layers - 2:
              x = self.lin_resGCN(x)
          if i < len(self.resNet) - 1:
              x = F.relu(x)
              x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

      return x


