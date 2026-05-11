import numpy as np
import pandas as pd
import torch
from collections import defaultdict


def build_burst(df):
    print("[R-Burst-R] 버스트 리뷰 패턴 (단기집중) 관계 구성 중...")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    prod_to_reviews = defaultdict(list)
    for idx, prod_id in enumerate(df["prod_id"]):
        prod_to_reviews[prod_id].append(idx)

    edges_src = []
    edges_dst = []

    burst_days = 7
    rating_diff = 1
    k = 10

    for prod_id, review_indices in prod_to_reviews.items():
        if len(review_indices) <= 1:
            continue

        review_data = [(idx, df.loc[idx, "date"], df.loc[idx, "rating"])
                       for idx in review_indices]
        review_data.sort(key=lambda x: x[1])

        for i, (idx_i, date_i, rating_i) in enumerate(review_data):
            candidates = []

            for j, (idx_j, date_j, rating_j) in enumerate(review_data):
                if i == j:
                    continue

                date_diff = abs((date_j - date_i).days)
                rating_diff_val = abs(rating_j - rating_i)

                if date_diff <= burst_days and rating_diff_val <= rating_diff:
                    candidates.append(idx_j)

            if len(candidates) > k:
                candidates = candidates[:k]

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
