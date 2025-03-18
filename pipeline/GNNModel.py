import torch
import torch.nn.functional as F
from torch_geometric.graphgym.optim import none_scheduler
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, JumpingKnowledge
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.optim as optim
from GIN import GINNodeClassifier
from resGCN import ResGCNLayer
from mamba import IterativeMambaGNN,IterativeMambaGNN2
from mamba2 import IterativeMambaGNN3
from mambawores import IterativeMambaGNN4
from mambawosort import IterativeMambaGNN5
from mambawomamba import IterativeMambaGNN6
from MLP import MLPModel 
from torch_geometric.loader import NeighborSampler
class GNNModel(torch.nn.Module):
    def __init__(self, model_type, in_channels, out_channels, num_layers=2, hidden_dim=64, heads=3, dropout=0.5):
        super(GNNModel, self).__init__()

        # define
        if model_type == "GCN":
            # 2，
            hidden_dim=hidden_dim*2
            self.convs = torch.nn.ModuleList([GCNConv(in_channels, hidden_dim)] +
                                             [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)] +
                                             [GCNConv(hidden_dim, out_channels)])
        elif model_type == "GraphSAGE":
            # 2，1
            if num_layers==2:              
                hidden_dim = hidden_dim*2
            if num_layers==8 or num_layers==32 :              
                hidden_dim = hidden_dim//4
            self.convs = torch.nn.ModuleList([SAGEConv(in_channels, hidden_dim)] +
                                             [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)] +
                                             [SAGEConv(hidden_dim, out_channels)])
        elif model_type == "GAT":
            # 3，3
            self.convs = torch.nn.ModuleList([GATConv(in_channels, hidden_dim, heads=heads)] +
                                             [GATConv(hidden_dim * heads, hidden_dim, heads=heads) for _ in
                                              range(num_layers - 2)] +
                                             [GATConv(hidden_dim * heads, out_channels, heads=1)])
        # elif model_type == "InceptionGCN":
        #     self.convs = torch.nn.ModuleList([GCNConv(in_channels, hidden_dim)] +
        #                                      [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)] +
        #                                      [GCNConv(hidden_dim, out_channels)])
        #     self.inception_layers = torch.nn.ModuleList(
        #         [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers // 2)])
        elif model_type == "resGCN":
            # 2，3
            if num_layers==2:              
                hidden_dim = hidden_dim*2
            if num_layers==8:              
                hidden_dim = hidden_dim*4
           
            self.convs = torch.nn.ModuleList([GCNConv(in_channels, hidden_dim)] + [ResGCNLayer(hidden_dim, hidden_dim)
                                                                                   for _ in range(num_layers - 2)]
                                             + [GCNConv(out_channels, out_channels)]
                                             )
        elif model_type == "JKNet":
            hidden_dim = hidden_dim*2
            self.convs = torch.nn.ModuleList([SAGEConv(in_channels, hidden_dim)] +
                                             [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
            self.jump = JumpingKnowledge(mode='cat')

        elif model_type == "GINConv":
            # 1，2
            hidden_dim = hidden_dim*2
            self.convs = GINNodeClassifier(in_channels=in_channels, hidden_dim=hidden_dim, out_channels=out_channels,
                                           num_layers=num_layers)

        elif model_type == "mlp":
            hidden_dim =hidden_dim
            self.convs = MLPModel( in_channels, hidden_dim, out_channels,num_layers=num_layers)
        #     resSage+SSM *N
        elif model_type == "mamba2":
            self.convs = IterativeMambaGNN2(   feat_dim=in_channels,
                                               hidden_dim=hidden_dim,
                                               num_classes=out_channels,
                                               num_layers=num_layers)
        elif model_type == "mamba3":
            self.convs = IterativeMambaGNN3(   feat_dim=in_channels,
                                               hidden_dim=hidden_dim,
                                               num_classes=out_channels,
                                               num_layers=num_layers)
        elif model_type == "mambawores":
            self.convs = IterativeMambaGNN4(   feat_dim=in_channels,
                                               hidden_dim=hidden_dim,
                                               num_classes=out_channels,
                                               num_layers=num_layers)
        elif model_type == "mambawosort":
            self.convs = IterativeMambaGNN5(   feat_dim=in_channels,
                                               hidden_dim=hidden_dim,
                                               num_classes=out_channels,
                                               num_layers=num_layers)
        elif model_type == "mambawomamba":
            self.convs = IterativeMambaGNN6(   feat_dim=in_channels,
                                               hidden_dim=hidden_dim,
                                               num_classes=out_channels,
                                               num_layers=num_layers)
        self.model_type = model_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lin_JKNet = torch.nn.Linear(hidden_dim * num_layers, out_channels)

        self.lin_resGCN = torch.nn.Linear(hidden_dim, out_channels)
        self.dropout = dropout
        self.bn = torch.nn.BatchNorm1d(hidden_dim)
        self.bn_head = torch.nn.BatchNorm1d(hidden_dim * heads)

    def forward(self, x, edge_index):
        out_list = []
        if self.model_type == "GINConv":
            x = self.convs(x, edge_index)
        elif self.model_type == "resGCN":
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i == self.num_layers - 2:
                    x = self.lin_resGCN(x)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

        elif self.model_type == "GAT":
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
           
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

        elif self.model_type == "JKNet":
            # 0,
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = self.bn(x)
                x = F.relu(x)
                
                x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout

                out_list.append(x)
        elif self.model_type == "mamba2" or self.model_type == "mamba3" or self.model_type == "mambawores"or  self.model_type == "mambawosort" or  self.model_type == "mambawomamba":
            x = self.convs(x, edge_index)
            # x = self.lin_JKNet(x)  # [num_nodes, out_channels]
            # hidden_dim, out_c
            # x = self.lin_resGCN(x)
            # print(x.shape)

        elif self.model_type == "mlp":
            x = self.convs(x, edge_index)

        elif self.model_type =="GCN":
            # 0,
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = self.bn(x)
                    x = F.relu(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout


        elif self.model_type =="GraphSAGE":
            
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                
               
                if i < len(self.convs) - 1:
                    x = self.bn(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
                x = F.relu(x)

        if self.model_type == "JKNet":
            out = torch.cat(out_list, dim=1)  # [num_nodes, num_layers * hidden_channels]
            #
            x = self.lin_JKNet(out)  # [num_nodes, out_channels]

        return F.log_softmax(x, dim=1)
# sage,GCN,JKNet，GAT,GIN,resGCN