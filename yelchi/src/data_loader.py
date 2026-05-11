"""YelpChi dataset loader (.mat format, CARE-GNN / PC-GNN standard)."""
import os
import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

# YelpChi.mat (CARE-GNN/PC-GNN format) keys:
#   features : sparse (N, 32) — node feature
#   label    : (N,) or (1, N) — {0, 1}, 1 = fraud
#   net_rur  : sparse (N, N) — review-user-review adjacency
#   net_rtr  : sparse (N, N) — review-time-review adjacency
#   net_rsr  : sparse (N, N) — review-star-review adjacency

RELATION_KEYS = ["net_rur", "net_rtr", "net_rsr"]
RELATION_NAMES = ["rur", "rtr", "rsr"]


def _adj_to_edge_index(adj):
    """scipy sparse -> torch LongTensor (2, E)."""
    coo = adj.tocoo()
    edge_index = np.vstack([coo.row, coo.col])
    return torch.from_numpy(edge_index).long()


def load_yelchi(mat_path: str, seed: int = 42,
                train_ratio: float = 0.64, valid_ratio: float = 0.16):
    """Load YelpChi.mat and return (x, y, edge_index_dict, masks).

    Returns:
        x: torch.FloatTensor (N, F)
        y: torch.LongTensor (N,)
        edge_index_dict: dict[str -> LongTensor (2, E)]
        train_mask, valid_mask, test_mask: torch.BoolTensor (N,)
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"YelpChi mat file not found at {mat_path}. "
            f"Put YelpChi.mat in yelchi/data/. "
            f"e.g. download from CARE-GNN repo: "
            f"https://github.com/YingtongDou/CARE-GNN/tree/master/data"
        )

    mat = sio.loadmat(mat_path)

    feats = mat["features"]
    if sp.issparse(feats):
        feats = feats.toarray()
    x = torch.from_numpy(feats.astype(np.float32))
    n_nodes = x.shape[0]

    label = mat["label"].squeeze().astype(np.int64)
    y = torch.from_numpy(label)

    edge_index_dict = {}
    for rel_key, rel_name in zip(RELATION_KEYS, RELATION_NAMES):
        if rel_key not in mat:
            raise KeyError(f"Missing relation '{rel_key}' in {mat_path}. "
                           f"Expected keys: {RELATION_KEYS}")
        adj = mat[rel_key]
        edge_index_dict[rel_name] = _adj_to_edge_index(adj)

    # Stratified split
    indices = np.arange(n_nodes)
    train_idx, temp_idx = train_test_split(
        indices, test_size=1 - train_ratio,
        stratify=label, random_state=seed,
    )
    valid_size = valid_ratio / (1 - train_ratio)
    valid_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - valid_size,
        stratify=label[temp_idx], random_state=seed,
    )
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    valid_mask[valid_idx] = True
    test_mask[test_idx] = True

    return x, y, edge_index_dict, train_mask, valid_mask, test_mask


def summary(x, y, edge_index_dict, train_mask, valid_mask, test_mask):
    print(f"[YelpChi] nodes={x.shape[0]} feats={x.shape[1]}")
    print(f"  label dist: {np.bincount(y.numpy())} "
          f"fraud_ratio={(y == 1).float().mean().item():.4f}")
    print(f"  split: train={train_mask.sum().item()} "
          f"valid={valid_mask.sum().item()} test={test_mask.sum().item()}")
    for k, ei in edge_index_dict.items():
        print(f"  relation {k}: edges={ei.shape[1]}")
