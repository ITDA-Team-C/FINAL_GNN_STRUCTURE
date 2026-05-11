import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


def union_edge_index(edge_index_dict):
    """Concatenate edge_index across all relations and dedup."""
    eis = [ei for ei in edge_index_dict.values() if ei.numel() > 0]
    if not eis:
        return torch.zeros((2, 0), dtype=torch.long, device=next(iter(edge_index_dict.values())).device)
    cat = torch.cat(eis, dim=1)
    return torch.unique(cat, dim=1)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
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
