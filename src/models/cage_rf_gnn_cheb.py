import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv


class CAGERFGNNBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3, K=3, use_skip_connection=False):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(input_dim, hidden_dim, K=K))
        for _ in range(num_layers - 1):
            self.convs.append(ChebConv(hidden_dim, hidden_dim, K=K))
        self.dropout = dropout
        self.use_skip_connection = use_skip_connection
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if use_skip_connection and input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = None

    def forward(self, x, edge_index):
        h = x
        if self.use_skip_connection and self.input_projection is not None:
            h = self.input_projection(h)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_skip_connection and i > 0:
                x = x + h
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            if self.use_skip_connection:
                h = x
        return x


class RelationGate(nn.Module):
    def __init__(self, hidden_dim, num_relations=6):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.num_relations = num_relations

    def forward(self, relation_embeddings):
        batch_size = relation_embeddings.shape[0]
        gate_logits = self.gate_mlp(relation_embeddings)
        gate_logits = gate_logits.view(batch_size, self.num_relations, 1)
        alpha = torch.softmax(gate_logits, dim=1)
        return alpha


class CAGERF_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3, use_gating=True,
                 use_ensemble=False, selected_relations=None, K=3, use_skip_connection=False, use_two_stage=False):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_gating = use_gating
        self.use_ensemble = use_ensemble
        self.selected_relations = selected_relations
        self.use_skip_connection = use_skip_connection
        self.use_two_stage = use_two_stage

        all_relations = ["rur", "rtr", "rsr", "burst", "semsim", "behavior"]
        if selected_relations is None:
            self.active_relations = all_relations
        else:
            self.active_relations = selected_relations if isinstance(selected_relations, list) else [selected_relations]

        self.branches = nn.ModuleDict()
        for rel in self.active_relations:
            self.branches[rel] = CAGERFGNNBranch(input_dim, hidden_dim, num_layers, dropout, K=K, use_skip_connection=use_skip_connection)

        num_active_relations = len(self.active_relations)

        if use_ensemble:
            self.ensemble_classifiers = nn.ModuleDict({
                rel: nn.Linear(hidden_dim, 1)
                for rel in self.active_relations
            })
            self.ensemble_weights = nn.Parameter(torch.ones(num_active_relations))
        elif use_gating:
            self.gate = RelationGate(hidden_dim, num_relations=num_active_relations)
        else:
            self.gate = None

        concat_dim = hidden_dim * num_active_relations
        projection_input_dim = hidden_dim if (use_gating or use_ensemble) else concat_dim

        self.projection = nn.Sequential(
            nn.Linear(projection_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self.aux_heads = nn.ModuleDict({
            rel: nn.Linear(hidden_dim, 1)
            for rel in self.active_relations
        })

        self.dropout = dropout

    def forward(self, x, edge_index_dict):
        h_dict = {}
        relation_embeddings_list = []

        for rel in self.active_relations:
            h = self.branches[rel](x, edge_index_dict[rel])
            h_dict[rel] = h
            relation_embeddings_list.append(h)

        relation_stack = torch.stack(relation_embeddings_list, dim=1)

        if self.use_ensemble:
            relation_scores = []
            for rel in self.active_relations:
                score = self.ensemble_classifiers[rel](h_dict[rel]).squeeze(-1)
                relation_scores.append(score)

            relation_scores = torch.stack(relation_scores, dim=1)
            ensemble_weights = torch.softmax(self.ensemble_weights, dim=0)
            logit = (relation_scores * ensemble_weights.unsqueeze(0)).sum(dim=1)
            self.last_ensemble_weights = ensemble_weights
        elif self.use_gating:
            alpha = self.gate(relation_stack)
            h_fused = (alpha.squeeze(-1).unsqueeze(-1) * relation_stack).sum(dim=1)
            self.last_alpha = alpha

            if self.use_two_stage:
                alpha_weights = alpha.squeeze(-1)
                weighted_relation_stack = relation_stack * alpha_weights.unsqueeze(-1)
                h_fused = weighted_relation_stack.sum(dim=1)
        else:
            h_cat = torch.cat(relation_embeddings_list, dim=1)
            h_fused = h_cat

        if not self.use_ensemble:
            h_proj = self.projection(h_fused)
            logit = self.classifier(h_proj).squeeze(-1)

        aux_logits = {
            rel: self.aux_heads[rel](h_dict[rel]).squeeze(-1)
            for rel in self.active_relations
        }

        return logit, aux_logits

    def get_relation_contribution(self):
        if self.use_ensemble and hasattr(self, 'last_ensemble_weights'):
            return self.last_ensemble_weights.detach().cpu().numpy()
        elif self.use_gating and hasattr(self, 'last_alpha'):
            return self.last_alpha.detach().cpu().numpy()
        return None
