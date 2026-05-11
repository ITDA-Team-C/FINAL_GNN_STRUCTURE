import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

from src.models.skip_cheb_branch import SkipChebBranch
from src.models.gated_relation_fusion import GatedRelationFusion
from src.filtering.care_neighbor_filter import filter_edges_by_feature_similarity


# CAGE-CareRF GNN
#   = SkipChebBranch (per relation) + GatedRelationFusion + Main + Aux heads
#   + (optional) per-forward CARE neighbor filtering on each relation's edges
#
# Notes on leakage safety:
#   - care_filter uses ONLY node features (cosine similarity). No labels.
#   - When `care_inline=False` (recommended), filtering is applied once
#     OUTSIDE the model (before training) and filtered edge_index_dict is
#     passed in. This is faster and equivalent under fixed X.
#   - When `care_inline=True`, filtering runs each forward pass (slow, only
#     used when X itself changes between epochs, which is not our case).


ALL_RELATIONS = ["rur", "rtr", "rsr", "burst", "semsim", "behavior"]


class CAGECareRF_GNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=3,
        dropout=0.3,
        K=3,
        active_relations=None,
        use_skip=True,
        use_gating=True,
        use_aux_loss=True,
        care_inline=False,
        care_top_k=10,
        care_min_sim=None,
    ):
        super().__init__()
        self.active_relations = list(active_relations) if active_relations else list(ALL_RELATIONS)
        self.use_skip = use_skip
        self.use_gating = use_gating
        self.use_aux_loss = use_aux_loss
        self.care_inline = care_inline
        self.care_top_k = int(care_top_k)
        self.care_min_sim = care_min_sim
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # Per-relation branches
        self.branches = nn.ModuleDict()
        for rel in self.active_relations:
            if use_skip:
                self.branches[rel] = SkipChebBranch(
                    input_dim=input_dim, hidden_dim=hidden_dim,
                    num_layers=num_layers, dropout=dropout, K=K,
                )
            else:
                self.branches[rel] = _PlainChebBranch(
                    input_dim=input_dim, hidden_dim=hidden_dim,
                    num_layers=num_layers, dropout=dropout, K=K,
                )

        # Fusion: gating or mean-fusion fallback
        if use_gating:
            self.fusion = GatedRelationFusion(hidden_dim=hidden_dim, num_relations=len(self.active_relations))
            projection_in = hidden_dim
        else:
            self.fusion = None
            projection_in = hidden_dim * len(self.active_relations)

        self.projection = nn.Sequential(
            nn.Linear(projection_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if use_aux_loss:
            self.aux_heads = nn.ModuleDict(
                {rel: nn.Linear(hidden_dim, 1) for rel in self.active_relations}
            )
        else:
            self.aux_heads = None

        self.last_alpha = None

    def forward(self, x, edge_index_dict):
        embeddings = []
        for rel in self.active_relations:
            ei = edge_index_dict[rel]
            if self.care_inline and self.care_top_k > 0:
                ei = filter_edges_by_feature_similarity(
                    x, ei, top_k=self.care_top_k, min_sim=self.care_min_sim
                )
            h = self.branches[rel](x, ei)
            embeddings.append(h)

        relation_stack = torch.stack(embeddings, dim=1)  # (N, R, H)

        if self.use_gating:
            fused, alpha = self.fusion(relation_stack)
            self.last_alpha = alpha
        else:
            fused = torch.cat(embeddings, dim=1)
            self.last_alpha = None

        h_proj = self.projection(fused)
        logit = self.classifier(h_proj).squeeze(-1)

        aux_logits = None
        if self.use_aux_loss and self.aux_heads is not None:
            aux_logits = {
                rel: self.aux_heads[rel](h_rel).squeeze(-1)
                for rel, h_rel in zip(self.active_relations, embeddings)
            }

        return logit, aux_logits

    def get_relation_contribution(self):
        if self.last_alpha is None:
            return None
        # average over nodes -> (R,)
        return self.last_alpha.detach().squeeze(-1).mean(dim=0).cpu().numpy()


class _PlainChebBranch(nn.Module):
    """No-skip ChebConv branch, used when use_skip=False (ablation)."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3, K=3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(input_dim, hidden_dim, K=K))
        for _ in range(num_layers - 1):
            self.convs.append(ChebConv(hidden_dim, hidden_dim, K=K))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
