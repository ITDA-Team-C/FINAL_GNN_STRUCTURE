import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv


class SkipChebBranch(nn.Module):
    """ChebConv-based relation branch with residual skip connection (v8 style).

    For each layer l ≥ 1, output = ReLU(Dropout(Conv(h_{l-1}) + h_{l-1})).
    Layer 0 first projects input to hidden_dim if needed so the residual add
    has matching dims.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3, K=3):
        super().__init__()
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.input_projection = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None
        )

        self.convs = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(ChebConv(in_dim, hidden_dim, K=K))
            in_dim = hidden_dim

    def forward(self, x, edge_index):
        h_prev = self.input_projection(x) if self.input_projection is not None else x
        for i, conv in enumerate(self.convs):
            out = conv(x if i == 0 else h_prev, edge_index)
            if i > 0:
                out = out + h_prev
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            h_prev = out
        return h_prev
