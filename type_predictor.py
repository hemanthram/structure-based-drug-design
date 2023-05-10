import torch
import torch.nn as nn

from classifier import SpatialClassifier
from context_encoder import ContextEncoder
from utils import compose_context, ShiftedSoftplus, AggregateKNN

class TypePredictor(nn.Module):
    
    def __init__(self, num_classes, hidden_channels, protein_atom_feature_dim, ligand_atom_feature_dim, aggregate_k = 5):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, hidden_channels)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, hidden_channels)
        
        self.aggregate_feature = AggregateKNN(protein_atom_feature_dim, ligand_atom_feature_dim, aggregate_k)

        self.spatial_classifier = SpatialClassifier(num_outputs=hidden_channels, in_channels=protein_atom_feature_dim)

        self.aggregatenn = nn.Sequential(
            nn.Linear(protein_atom_feature_dim+ligand_atom_feature_dim, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
            # ShiftedSoftplus(),
            # nn.Linear(hidden_channels, num_classes)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, num_classes),
            nn.Softmax(dim=0)
        )

        self.terminatenn = nn.Sequential(
            nn.Linear(protein_atom_feature_dim+ligand_atom_feature_dim, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, pos_query, protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature):

        aggregate_ctx = self.aggregate_feature(protein_pos, protein_atom_feature, ligand_pos, ligand_atom_feature).to(self.device)
        aggregate_h = self.aggregatenn(aggregate_ctx)

        spatial_h = self.spatial_classifier(pos_query, protein_pos, protein_atom_feature)
        
        h = torch.cat([aggregate_h, spatial_h[0]])
        atom_type = self.classifier(h)

        terminate = self.terminatenn(aggregate_ctx)
        
        return atom_type, terminate