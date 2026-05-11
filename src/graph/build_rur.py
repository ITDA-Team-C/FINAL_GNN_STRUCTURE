import numpy as np
import pandas as pd
import torch
from collections import defaultdict


def build_rur(df):
    print("[R-U-R] Review-User-Review 관계 구성 중...")

    user_to_reviews = defaultdict(list)
    for idx, user_id in enumerate(df["user_id"]):
        user_to_reviews[user_id].append(idx)

    edges_src = []
    edges_dst = []

    k = 10

    for user_id, review_indices in user_to_reviews.items():
        if len(review_indices) <= 1:
            continue

        for i, idx_i in enumerate(review_indices):
            candidates = [idx_j for j, idx_j in enumerate(review_indices) if i != j]

            if len(candidates) > k:
                candidates = candidates[-k:]

            for idx_j in candidates:
                edges_src.append(idx_i)
                edges_dst.append(idx_j)

    if len(edges_src) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

    edge_index = torch.unique(edge_index, dim=1)

    print(f"  엣지 수: {edge_index.shape[1]}")

    return edge_index
