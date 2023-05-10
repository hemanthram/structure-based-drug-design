import torch
import torch.nn as nn
from torch_scatter import scatter_add
import numpy as np
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()
        self.softplus = nn.Softplus() 

    def forward(self, x):
        return self.softplus(x) - self.shift

class AggregateKNN(nn.Module):

    def __init__(self, protein_atom_feature_dim, ligand_atom_feature_dim, k = 5):
        super().__init__()
        self.pfdim = protein_atom_feature_dim
        self.lfdim = ligand_atom_feature_dim
        self.k = k
        
    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature):
        
        if(ligand_atom_feature is None): 
            ligand_ctx = torch.zeros(self.lfdim)
            protein_ctx = torch.zeros(self.pfdim)
        else: 
            ligand_ctx = torch.sum(ligand_atom_feature, dim=0)
            # print(protein_pos.shape, ligand_pos.shape)
            assign_idx = myknn(
                x=protein_pos,
                y=ligand_pos,
                k=self.k
            )
            h = protein_atom_feature[assign_idx[1]]
            # h = torch.sum(h, dim=0)
            h = scatter_add(h, index=assign_idx[0], dim=0, dim_size=ligand_pos.size(0))
            protein_ctx = torch.mean(h, dim=0)
        
        h_ctx = torch.cat([ligand_ctx, protein_ctx])
        return h_ctx

def compose_context(protein_pos, h_protein, ligand_pos, h_ligand):
    
    if(h_ligand is not None): h_ctx = torch.cat([h_protein, h_ligand])
    else: h_ctx = torch.cat([h_protein])

    if(ligand_pos is not None): pos_ctx = torch.cat([protein_pos, ligand_pos])
    else: pos_ctx = torch.cat([protein_pos])

    return h_ctx, pos_ctx

# def get_nearest_point(points, centers):
    
#     dist = torch.cdist(points, centers)
#     print(points.shape, centers.shape, dist.shape)
#     print(torch.argmin(dist))

# generating box to place around points
def generate_box(limits, resolution):
    n_dims = 3
    grid_max = limits[1]
    grid_steps = int(grid_max * 2 * int(1/resolution)) + 1
    pts = np.linspace(-grid_max, grid_max, grid_steps)
    grid = np.meshgrid(*[pts for _ in range(n_dims)])
    grid = np.stack(grid, axis=-1)  # stack to array (instead of list)
    # reshape into 2d array of positions
    shape_a0 = np.prod(grid.shape[:n_dims])
    grid = np.reshape(grid, (shape_a0, -1))

    grid_dists = np.sqrt(np.sum(grid**2, axis=-1))
    grid_mask = np.logical_and(grid_dists >= limits[0],
                                grid_dists <= limits[1])
    return grid[grid_mask]

def myknn(x, y, k):
    dist = torch.cdist(y,x)
    sortix = torch.argsort(dist, dim=1)
    ksortix = sortix[:, :k]
    row = torch.arange(ksortix.size(0), dtype=torch.long).view(-1, 1).repeat(1, k).to(device)
    return torch.stack([torch.flatten(row), torch.flatten(ksortix)], dim=0)

# finding k nearest points to the center of the ligand in the protein
def find_nearest_protein_atoms_to_ligand(pp, lp, k=100):
    ligand_center = torch.mean(lp, axis=0)

    neighbors = myknn(
        x = pp,
        y = ligand_center.view(1,3),
        k = k
    )
    return neighbors[1]

# getting points for action space
def get_action_points(select_centers, centers, limits, resolution):
    box = torch.Tensor(generate_box(limits, resolution)).to(device)
    grid_all = []
    n_dims = 3
    if(select_centers is None): select_centers = centers
    for i, pos in enumerate(select_centers):
        grid_this = box + pos.view(1, n_dims)  # (N, 3)
        dists = torch.norm(
            grid_this.view(-1, 1, n_dims) - torch.cat([centers[:i],centers[i+1:]], dim=0).view(1, -1, n_dims),
            dim=-1, p=2,
        )   
        mask = (dists > limits[0]).all(dim=1)
        grid_this = grid_this[mask]
        grid_all.append(grid_this)
    res = torch.cat(grid_all, dim=0)
    return res

def equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3): 
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return a,b,c,d

def get_action_points_plane(select_centers, centers, limits, resolution, plane_eq, plane_dist_limit):

    # print(plane_eq)
    box = torch.Tensor(generate_box(limits, resolution)).to(device)
    grid_all = []
    n_dims = 3
    if(select_centers is None): select_centers = centers
    for i, pos in enumerate(select_centers):
        grid_this = box + pos.view(1, n_dims)  # (N, 3)
        dists = torch.norm(
            grid_this.view(-1, 1, n_dims) - torch.cat([centers[:i],centers[i+1:]], dim=0).view(1, -1, n_dims),
            dim=-1, p=2,
        )   
        mask = (dists > limits[0]).all(dim=1)
        grid_this = grid_this[mask]

        # print(grid_this.shape)
        plane_dist = plane_eq[0]*grid_this[:,0] + plane_eq[1]*grid_this[:,1] + plane_eq[2]*grid_this[:,2] + plane_eq[3]
        plane_dist = torch.abs(plane_dist) / (plane_eq[0]**2 + plane_eq[1]**2 + plane_eq[2]**2)**0.5

        plane_dist_mask = ( plane_dist < plane_dist_limit )#.all(dim=1)
        grid_this = grid_this[plane_dist_mask]

        # print(torch.min(plane_dist), torch.max(plane_dist))
        # print(grid_this.shape)

        grid_all.append(grid_this)
    res = torch.cat(grid_all, dim=0)
    return res