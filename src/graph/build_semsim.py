import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def build_semsim(df, text_embeddings):
    print("[R-SemSim-R] 의미론적 유사도 관계 구성 중...")

    prod_to_reviews = defaultdict(list)
    for idx, prod_id in enumerate(df["prod_id"]):
        prod_to_reviews[prod_id].append(idx)

    edges_src = []
    edges_dst = []

    k = 5

    for prod_id, review_indices in prod_to_reviews.items():
        if len(review_indices) <= 1:
            continue

        indices_array = np.array(review_indices)
        embeddings_subset = text_embeddings[indices_array]

        similarity_matrix = cosine_similarity(embeddings_subset)

        for i, idx_i in enumerate(review_indices):
            sims = similarity_matrix[i]

            sorted_indices = np.argsort(-sims)[1:k+1]
            candidates = [review_indices[j] for j in sorted_indices if sims[j] > 0]

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
