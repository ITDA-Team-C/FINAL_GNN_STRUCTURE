import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def extract_user_behavior(df):
    """Returns (user_ids ndarray, user_feature_matrix np.ndarray[n_users, 5])."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    user_stats = df.groupby("user_id").agg(
        review_count=("review_id", "count"),
        avg_rating=("rating", "mean"),
        rating_std=("rating", "std"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        product_diversity=("prod_id", "nunique"),
    )
    user_stats["active_days"] = (user_stats["last_date"] - user_stats["first_date"]).dt.days + 1
    user_stats = user_stats.fillna(0)

    feat_cols = ["review_count", "avg_rating", "rating_std", "active_days", "product_diversity"]
    user_ids = user_stats.index.values
    X = user_stats[feat_cols].values.astype(np.float64)
    X = StandardScaler().fit_transform(X)
    return user_ids, X


def _cosine_topk_chunked(X, top_k, sim_threshold, chunk_size=512):
    """Returns dict user_row_idx -> list of (neighbor_row_idx, sim)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = Xn.shape[0]
    out = {}
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = Xn[start:end] @ Xn.T
        for i_local, i_global in enumerate(range(start, end)):
            sims = block[i_local]
            sims[i_global] = -np.inf
            if top_k >= n:
                idxs = np.argsort(-sims)
            else:
                idxs = np.argpartition(-sims, top_k)[:top_k]
                idxs = idxs[np.argsort(-sims[idxs])]
            kept = [(int(j), float(sims[j])) for j in idxs if sims[j] > sim_threshold]
            if kept:
                out[i_global] = kept
    return out


def build_behavior(df, top_k=5, sim_threshold=0.3, max_reviews_per_user=3, seed=42):
    print("[R-UserBehavior-R] user-level cosine top-k -> review pair edges...")

    user_ids, X = extract_user_behavior(df)
    n_users = len(user_ids)
    print(f"  users={n_users}  feat_dim={X.shape[1]}")

    user_id_to_row = {uid: i for i, uid in enumerate(user_ids)}
    neighbor_map = _cosine_topk_chunked(X, top_k=top_k, sim_threshold=sim_threshold)
    print(f"  user-pairs (with cos>{sim_threshold}, top-{top_k}): "
          f"{sum(len(v) for v in neighbor_map.values())}")

    df = df.reset_index(drop=True)
    df["__node_idx__"] = np.arange(len(df))
    rng = np.random.default_rng(seed)
    user_to_node_idx = {}
    for uid, group in df.groupby("user_id"):
        idxs = group["__node_idx__"].values
        if len(idxs) > max_reviews_per_user:
            idxs = rng.choice(idxs, size=max_reviews_per_user, replace=False)
        user_to_node_idx[uid] = idxs

    edges_src = []
    edges_dst = []
    for src_uid in user_ids:
        src_row = user_id_to_row[src_uid]
        if src_row not in neighbor_map:
            continue
        src_nodes = user_to_node_idx.get(src_uid, np.array([], dtype=int))
        if len(src_nodes) == 0:
            continue
        for dst_row, _sim in neighbor_map[src_row]:
            dst_uid = user_ids[dst_row]
            dst_nodes = user_to_node_idx.get(dst_uid, np.array([], dtype=int))
            for s in src_nodes:
                for d in dst_nodes:
                    if s != d:
                        edges_src.append(int(s))
                        edges_dst.append(int(d))

    if len(edges_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_index = torch.unique(edge_index, dim=1)

    print(f"  엣지 수: {edge_index.shape[1]}")
    return edge_index
