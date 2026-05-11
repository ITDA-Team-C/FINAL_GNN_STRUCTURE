import torch
import torch.nn as nn


class GatedRelationFusion(nn.Module):
    """Per-node softmax gating over relation branch embeddings.

    Input:  relation_stack of shape (N, R, H)
    Output: fused embedding (N, H), alpha (N, R, 1)

    For each node, alpha is computed by a small MLP that maps each H-dim
    embedding to a scalar logit, then softmaxed over the R dim. Final
    embedding is the alpha-weighted sum of relation embeddings.
    """

    def __init__(self, hidden_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, relation_stack):
        # relation_stack: (N, R, H)
        n, r, h = relation_stack.shape
        gate_logits = self.gate_mlp(relation_stack).view(n, r, 1)
        alpha = torch.softmax(gate_logits, dim=1)
        fused = (alpha * relation_stack).sum(dim=1)
        return fused, alpha


def gate_entropy_regularizer(alpha):
    """Per-node entropy of the alpha distribution. Larger => more uniform.

    `alpha` shape: (N, R, 1) or (N, R). Returns scalar mean entropy.
    """
    a = alpha.squeeze(-1) if alpha.dim() == 3 else alpha
    a = a.clamp_min(1e-12)
    entropy = -(a * a.log()).sum(dim=1)
    return entropy.mean()
