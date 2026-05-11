import numpy as np
import pandas as pd
import torch
from collections import defaultdict


def build_rtr(df):
    print("[R-T-R] Review-Time-Review (동기간 같은상품) 관계 구성 중...")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M")

    prod_month_to_reviews = defaultdict(list)
    for idx, (prod_id, month) in enumerate(zip(df["prod_id"], df["year_month"])):
        prod_month_to_reviews[(prod_id, month)].append(idx)

    edges_src = []
    edges_dst = []

    k = 10

    for (prod_id, month), review_indices in prod_month_to_reviews.items():
        if len(review_indices) <= 1:
            continue

        review_indices_sorted = sorted(
            review_indices,
            key=lambda idx: df.loc[idx, "date"]
        )

        for i, idx_i in enumerate(review_indices_sorted):
            distances = [abs(j - i) for j in range(len(review_indices_sorted))]
            sorted_indices = np.argsort(distances)[1:k+1]
            candidates = [review_indices_sorted[j] for j in sorted_indices]

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
