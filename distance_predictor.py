import torch.nn as nn

from context_encoder import ContextEncoder
from utils import compose_context, ShiftedSoftplus

class DistancePredictor(nn.Module):
    
    def __init__(self, hidden_channels, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, hidden_channels)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, hidden_channels)
        
        self.encoder = ContextEncoder(hidden_channels=hidden_channels)
        self.dist = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, 1)
        )

        
    def forward(self, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature):

        h_protein = self.protein_atom_emb(protein_atom_feature)
        
        if(ligand_atom_feature is not None): 
            h_ligand = self.ligand_atom_emb(ligand_atom_feature)
            n_ligand = ligand_atom_feature.shape[0]
        else:
            h_ligand = None
            n_ligand = 0
        
        h_ctx, pos_ctx = compose_context(protein_pos, h_protein, ligand_pos, h_ligand)
        h_ctx = self.encoder(h_ctx, pos_ctx)

        h_ligand =  h_ctx[-n_ligand:]

        h_dist = self.dist(h_ligand)
        return h_dist