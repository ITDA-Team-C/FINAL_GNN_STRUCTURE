"""Baselines + CAGE-CareRF model for fraud detection on .mat datasets."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv


def _union_edge_index(edge_index_dict):
    eis = [ei for ei in edge_index_dict.values() if ei.numel() > 0]
    if not eis:
        return torch.zeros((2, 0), dtype=torch.long,
                            device=next(iter(edge_index_dict.values())).device)
    return torch.unique(torch.cat(eis, dim=1), dim=1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index_dict):
        return self.net(x).squeeze(-1)


class _BaseGNN(nn.Module):
    def __init__(self, conv_cls, input_dim, hidden_dim=128, num_layers=3,
                 dropout=0.3, **conv_kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(conv_cls(input_dim, hidden_dim, **conv_kwargs))
        for _ in range(num_layers - 1):
            self.convs.append(conv_cls(hidden_dim, hidden_dim, **conv_kwargs))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = dropout
        self._cached_edge_index = None

    def forward(self, x, edge_index_dict):
        if self._cached_edge_index is None:
            self._cached_edge_index = _union_edge_index(edge_index_dict)
        ei = self._cached_edge_index
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, ei)), p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


class GCN(_BaseGNN):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__(GCNConv, input_dim, hidden_dim, num_layers, dropout)


class GAT(_BaseGNN):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_heads=8, dropout=0.3):
        head_dim = hidden_dim // num_heads
        super().__init__(
            GATConv, input_dim, head_dim, num_layers, dropout,
            heads=num_heads, concat=True,
        )


class GraphSAGE(_BaseGNN):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__(SAGEConv, input_dim, hidden_dim, num_layers, dropout)


# ---------- CAGE-CareRF for .mat-style 3-relation fraud datasets ----------


class SkipChebBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3, K=3):
        super().__init__()
        self.dropout = dropout
        self.input_projection = (nn.Linear(input_dim, hidden_dim)
                                 if input_dim != hidden_dim else None)
        self.convs = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(ChebConv(in_dim, hidden_dim, K=K))
            in_dim = hidden_dim

    def forward(self, x, edge_index):
        h_prev = self.input_projection(x) if self.input_projection else x
        for i, conv in enumerate(self.convs):
            out = conv(x if i == 0 else h_prev, edge_index)
            if i > 0:
                out = out + h_prev
            out = F.dropout(F.relu(out), p=self.dropout, training=self.training)
            h_prev = out
        return h_prev


class GatedRelationFusion(nn.Module):
    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, stack):  # (N, R, H)
        n, r, h = stack.shape
        a = torch.softmax(self.gate_mlp(stack).view(n, r, 1), dim=1)
        fused = (a * stack).sum(dim=1)
        return fused, a


@torch.no_grad()
def care_filter(x, edge_index, top_k=10):
    """Per-src top-k filtering by cosine similarity. Label-free."""
    if edge_index.shape[1] == 0:
        return edge_index
    src, dst = edge_index[0], edge_index[1]
    xn = F.normalize(x, p=2, dim=1)
    sim = (xn[src] * xn[dst]).sum(dim=1)
    order = torch.argsort(src, stable=True)
    src_s, dst_s, sim_s = src[order], dst[order], sim[order]
    _, counts = torch.unique_consecutive(src_s, return_counts=True)
    keep = torch.zeros_like(src_s, dtype=torch.bool)
    offset = 0
    for cnt in counts.tolist():
        end = offset + cnt
        if cnt <= top_k:
            keep[offset:end] = True
        else:
            top = torch.topk(sim_s[offset:end], k=top_k).indices
            keep[offset + top] = True
        offset = end
    return torch.stack([src_s[keep], dst_s[keep]], dim=0)


class CAGECareRF(nn.Module):
    """3-relation CAGE-CareRF for Amazon / YelpChi .mat datasets."""

    def __init__(self, input_dim, relations, hidden_dim=128, num_layers=3,
                 dropout=0.3, K=3,
                 use_skip=True, use_gating=True, use_aux_loss=True,
                 use_care=True, care_top_k=10):
        super().__init__()
        self.relations = list(relations)
        self.use_gating = use_gating
        self.use_aux_loss = use_aux_loss
        self.use_care = use_care
        self.care_top_k = int(care_top_k)
        n_rel = len(self.relations)

        BranchCls = SkipChebBranch if use_skip else _PlainChebBranch
        self.branches = nn.ModuleDict({
            r: BranchCls(input_dim, hidden_dim, num_layers, dropout, K=K)
            for r in self.relations
        })

        if use_gating:
            self.fusion = GatedRelationFusion(hidden_dim, n_rel)
            proj_in = hidden_dim
        else:
            self.fusion = None
            proj_in = hidden_dim * n_rel

        self.projection = nn.Sequential(
            nn.Linear(proj_in, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if use_aux_loss:
            self.aux_heads = nn.ModuleDict({r: nn.Linear(hidden_dim, 1) for r in self.relations})
        else:
            self.aux_heads = None
        self.last_alpha = None
        self._cached_filtered = None

    def _get_filtered_edges(self, x, edge_index_dict):
        if not self.use_care:
            return edge_index_dict
        if self._cached_filtered is None:
            self._cached_filtered = {
                r: care_filter(x, edge_index_dict[r], top_k=self.care_top_k)
                for r in self.relations
            }
        return self._cached_filtered

    def forward(self, x, edge_index_dict):
        edges = self._get_filtered_edges(x, edge_index_dict)
        embs = [self.branches[r](x, edges[r]) for r in self.relations]
        stack = torch.stack(embs, dim=1)  # (N, R, H)
        if self.use_gating:
            fused, alpha = self.fusion(stack)
            self.last_alpha = alpha
        else:
            fused = torch.cat(embs, dim=1)
        h = self.projection(fused)
        logit = self.classifier(h).squeeze(-1)
        aux = None
        if self.use_aux_loss:
            aux = {r: self.aux_heads[r](emb).squeeze(-1) for r, emb in zip(self.relations, embs)}
        return logit, aux


class _PlainChebBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3, K=3):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(input_dim, hidden_dim, K=K))
        for _ in range(num_layers - 1):
            self.convs.append(ChebConv(hidden_dim, hidden_dim, K=K))

    def forward(self, x, ei):
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, ei)), p=self.dropout, training=self.training)
        return x


# ---------- Loss ----------


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        t = targets.float()
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        pt = torch.where(t == 1, p, 1 - p)
        a = torch.where(t == 1, torch.full_like(p, self.alpha),
                        torch.full_like(p, 1 - self.alpha))
        return (a * (1 - pt) ** self.gamma * ce).mean()


class FocalAuxLoss(nn.Module):
    def __init__(self, main_loss_fn, aux_weight=0.3):
        super().__init__()
        self.main = main_loss_fn
        self.aux_weight = aux_weight

    def forward(self, main_logit, targets, aux_logits_dict=None):
        loss = self.main(main_logit, targets)
        if aux_logits_dict and self.aux_weight > 0:
            aux = torch.stack([
                F.binary_cross_entropy_with_logits(a, targets.float())
                for a in aux_logits_dict.values()
            ]).mean()
            loss = loss + self.aux_weight * aux
        return loss
