
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from numpy import pi as PI

from context_encoder import GaussianSmearing
from utils import ShiftedSoftplus, myknn

class SpatialClassifier(nn.Module):
    
    def __init__(self, num_outputs, in_channels, num_filters=64, k=32, cutoff=10.0):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, num_filters)
        self.distnn = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters)
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_outputs),
        )
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_filters)
        self.k = k
        self.cutoff = cutoff
    
    def forward(self, pos_query, pos_ctx, node_attr_ctx):
        
        assign_idx = myknn(
            x=pos_ctx,
            y=pos_query,
            k=self.k
        )
        
        dist_ij = torch.norm(pos_query[assign_idx[0]] - pos_ctx[assign_idx[1]], p=2, dim=-1).view(-1, 1).to(self.device)
        node_attr_ctx_j = node_attr_ctx[assign_idx[1]]
        
        W = self.distnn(self.distance_expansion(dist_ij))
        h = self.lin2(W * self.lin1(node_attr_ctx_j))

        C = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)  # (A, 1)
        C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
        h = h * C.view(-1, 1)   # (A, 1)
        
        y = scatter_add(h, index=assign_idx[0], dim=0, dim_size=pos_query.size(0))
        y_cls = self.classifier(y)
        
        return y_cls


