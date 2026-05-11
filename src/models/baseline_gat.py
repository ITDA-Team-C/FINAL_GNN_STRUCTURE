import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src.models.baseline_gcn import union_edge_index


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_heads=8, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, concat=True))
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
            )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout
        self._cached_edge_index = None

    def forward(self, x, edge_index_dict):
        if self._cached_edge_index is None:
            self._cached_edge_index = union_edge_index(edge_index_dict)
        edge_index = self._cached_edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)
