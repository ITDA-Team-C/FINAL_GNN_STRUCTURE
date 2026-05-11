import os
import torch
import torch.nn.functional as F

from src.utils import save_json


# CARE-inspired neighbor filter (feature-similarity based, label-free).
#
# For each src node in an edge_index, keeps only the top-k destination
# neighbors with highest cosine similarity to src in node-feature space.
# Optionally drops edges with cosine < min_sim.
#
# Uses ONLY node features (X). Does NOT consult train/valid/test labels,
# so it is leakage-safe under transductive setting.


@torch.no_grad()
def filter_edges_by_feature_similarity(x, edge_index, top_k=10, min_sim=None):
    """Returns filtered edge_index keeping per-src top-k neighbors.

    Args:
        x: (N, F) node feature tensor (float)
        edge_index: (2, E) long
        top_k: max neighbors retained per src node
        min_sim: optional float; drop edges below this cosine similarity
    """
    if edge_index.shape[1] == 0:
        return edge_index

    src, dst = edge_index[0], edge_index[1]
    x_norm = F.normalize(x, p=2, dim=1)
    sim = (x_norm[src] * x_norm[dst]).sum(dim=1)

    if min_sim is not None:
        mask = sim >= float(min_sim)
        src = src[mask]
        dst = dst[mask]
        sim = sim[mask]
        if src.numel() == 0:
            return torch.zeros((2, 0), dtype=edge_index.dtype, device=edge_index.device)

    # group by src using sorting (stable, no python loop)
    order = torch.argsort(src, stable=True)
    src_sorted = src[order]
    dst_sorted = dst[order]
    sim_sorted = sim[order]

    keep_mask = torch.zeros_like(src_sorted, dtype=torch.bool)
    # iterate over unique src using boundaries from sorted run-length
    unique_src, counts = torch.unique_consecutive(src_sorted, return_counts=True)
    offset = 0
    for cnt in counts.tolist():
        end = offset + cnt
        if cnt <= top_k:
            keep_mask[offset:end] = True
        else:
            local_sim = sim_sorted[offset:end]
            top_local = torch.topk(local_sim, k=top_k).indices
            keep_mask[offset + top_local] = True
        offset = end

    out_src = src_sorted[keep_mask]
    out_dst = dst_sorted[keep_mask]
    return torch.stack([out_src, out_dst], dim=0)


def filter_edge_index_dict(x, edge_index_dict, top_k_per_relation, min_sim_per_relation=None,
                           log_path=None):
    """Apply per-relation CARE filter. Returns (filtered_dict, log_dict).

    Args:
        x: node feature tensor
        edge_index_dict: {relation: LongTensor[2, E]}
        top_k_per_relation: dict relation -> int OR a single int applied to all
        min_sim_per_relation: dict relation -> float | None (optional)
        log_path: if set, writes a json log of before/after edge counts
    """
    if isinstance(top_k_per_relation, int):
        top_k_per_relation = {rel: top_k_per_relation for rel in edge_index_dict.keys()}
    min_sim_per_relation = min_sim_per_relation or {}

    filtered = {}
    log = {"relations": {}}
    for rel, edge_index in edge_index_dict.items():
        before = int(edge_index.shape[1])
        top_k = int(top_k_per_relation.get(rel, 10))
        min_sim = min_sim_per_relation.get(rel, None)
        filtered_ei = filter_edges_by_feature_similarity(x, edge_index, top_k=top_k, min_sim=min_sim)
        after = int(filtered_ei.shape[1])

        # isolated ratio (over all N nodes)
        n = x.shape[0]
        if after > 0:
            nonisolated = torch.unique(filtered_ei.flatten()).numel()
        else:
            nonisolated = 0
        iso_ratio = float((n - nonisolated) / n)

        filtered[rel] = filtered_ei
        log["relations"][rel] = {
            "before_edges": before,
            "after_edges": after,
            "kept_ratio": after / before if before > 0 else 0.0,
            "top_k": top_k,
            "min_sim": min_sim,
            "isolated_ratio_after": iso_ratio,
        }
        print(f"  [CARE] {rel:10s}  {before:>7d} -> {after:>7d}  "
              f"(kept {after / max(1, before):.2%})  iso={iso_ratio:.3f}")

    log["meta"] = {
        "num_nodes": int(x.shape[0]),
        "note": "Feature-cosine top-k filtering. No labels used.",
    }
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        save_json(log, log_path)
        print(f"[Save] {log_path}")

    return filtered, log
