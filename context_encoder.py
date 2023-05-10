import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from numpy import pi as PI

from utils import ShiftedSoftplus

class EdgeIndex(nn.Module):
    
    def __init__(self, cutoff):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cutoff = cutoff
        
    def forward(self, pos):
        dist = torch.cdist(pos,pos)
        diag = (self.cutoff+1)*torch.eye(dist.shape[0]).to(self.device)
        dist = dist+diag
        filt = (dist<self.cutoff).nonzero()
        return torch.transpose(filt, 0, 1)

class GaussianSmearing(nn.Module):
    
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.offset = torch.linspace(start, stop, num_gaussians).to(self.device)
        self.coeff = -0.5 / (self.offset[1] - self.offset[0]).item()**2
        # self.register_buffer('offset', self.offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class ConvBlock(MessagePassing):
    
    def __init__(self, in_channels, out_channels, num_filters, edge_channels, cutoff=10.0):
        super().__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.distnn = nn.Sequential(
            nn.Linear(edge_channels, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters)
        )
        self.cutoff = cutoff
        
    def forward(self, x, edge_index, edge_length, edge_attr):
        # print(edge_attr.shape)
        W = self.distnn(edge_attr)

        if self.cutoff is not None:
            C = 0.5 * (torch.cos(edge_length * PI / self.cutoff) + 1.0)
            C = C * (edge_length <= self.cutoff) * (edge_length >= 0.0)     # Modification: cutoff
            W = W * C.view(-1, 1)
        
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x
    
    def message(self, x_j, W):
        # print(x_j.shape, W.shape)
        return x_j * W

class InteractionBlock(nn.Module):

    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.conv = ConvBlock(hidden_channels, hidden_channels, num_filters, num_gaussians, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_length, edge_attr):
        x = self.conv(x, edge_index, edge_length, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x

class ContextEncoder(nn.Module):
    
    def __init__(self, hidden_channels=32, num_filters=64, num_interactions=5, edge_channels=64, cutoff=10.0):
        super().__init__()
        self.cutoff = cutoff
        self.find_edge_index = EdgeIndex(self.cutoff)
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        # self.block = InteractionBlock(hidden_channels, edge_channels, num_filters, cutoff)
        
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, edge_channels, num_filters, cutoff)
            self.interactions.append(block)
    
    def forward(self, node_attr, pos):
        edge_index = self.find_edge_index(pos)
        edge_length = torch.norm(pos[edge_index[0]]-pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        h = node_attr
        
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_length, edge_attr)
        # h = self.block(h, edge_index, edge_length, edge_attr)
        return h