import numpy as np
import pandas as pd
import torch
from collections import defaultdict


def build_rsr(df):
    print("[R-S-R] Review-Star-Review (같은상품 같은별점) 관계 구성 중...")

    prod_rating_to_reviews = defaultdict(list)
    for idx, (prod_id, rating) in enumerate(zip(df["prod_id"], df["rating"])):
        prod_rating_to_reviews[(prod_id, rating)].append(idx)

    edges_src = []
    edges_dst = []

    k = 10

    for (prod_id, rating), review_indices in prod_rating_to_reviews.items():
        if len(review_indices) <= 1:
            continue

        for i, idx_i in enumerate(review_indices):
            candidates = [idx_j for j, idx_j in enumerate(review_indices) if i != j]

            if len(candidates) > k:
                candidates = np.random.choice(candidates, size=k, replace=False).tolist()

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
