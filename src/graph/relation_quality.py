import os
import argparse
import json
import numpy as np
import pandas as pd
import torch

from src.utils import save_json


# Computes per-relation quality metrics using TRAIN labels only (leakage-safe).
# Output: outputs/metrics/relation_quality.json + relation_quality.csv (for report).
#
# Metrics:
#   edge_count            : number of directed edges in stored edge_index
#   avg_degree            : mean degree over nodes that appear at least once
#   isolated_ratio        : fraction of nodes with degree == 0
#   fraud_fraud_ratio     : fraction of edges where both endpoints are TRAIN-fraud
#   normal_normal_ratio   : fraction of edges where both endpoints are TRAIN-normal
#   fraud_normal_ratio    : fraction of edges where one endpoint is TRAIN-fraud, other TRAIN-normal
#   fraud_edge_lift       : fraud_fraud_ratio / (train_fraud_ratio ** 2)
#                           (>1 means relation is more fraud-homophilous than chance)
#
# Edges where either endpoint is not in TRAIN are excluded from the ratio
# computations (denominator). The reasoning: only train labels are observable;
# valid/test labels must not influence relation diagnostics.


def _ratios(edge_index, fraud_set, normal_set):
    if edge_index.shape[1] == 0:
        return 0, 0.0, 0.0, 0.0

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    src_is_f = np.isin(src, list(fraud_set))
    src_is_n = np.isin(src, list(normal_set))
    dst_is_f = np.isin(dst, list(fraud_set))
    dst_is_n = np.isin(dst, list(normal_set))

    both_train = (src_is_f | src_is_n) & (dst_is_f | dst_is_n)
    if both_train.sum() == 0:
        return 0, 0.0, 0.0, 0.0

    ff = (src_is_f & dst_is_f & both_train).sum()
    nn = (src_is_n & dst_is_n & both_train).sum()
    fn = (((src_is_f & dst_is_n) | (src_is_n & dst_is_f)) & both_train).sum()

    denom = float(both_train.sum())
    return int(both_train.sum()), ff / denom, nn / denom, fn / denom


def compute_relation_quality(edge_index_dict, df):
    train_mask = df["split"].values == "train"
    labels = df["label"].values
    n = len(df)

    train_idx = np.where(train_mask)[0]
    fraud_set = set(train_idx[labels[train_idx] == 1].tolist())
    normal_set = set(train_idx[labels[train_idx] == 0].tolist())

    train_fraud_ratio = len(fraud_set) / max(1, len(train_idx))
    expected_ff_under_random = train_fraud_ratio ** 2

    results = {
        "meta": {
            "num_nodes": int(n),
            "num_train_nodes": int(len(train_idx)),
            "train_fraud_ratio": float(train_fraud_ratio),
            "expected_ff_under_random": float(expected_ff_under_random),
            "note": "fraud-fraud/normal-normal/fraud-normal ratios computed only on edges where BOTH endpoints are TRAIN nodes (leakage-safe).",
        },
        "relations": {},
    }

    for rel, edge_index in edge_index_dict.items():
        edge_count = int(edge_index.shape[1])

        if edge_count > 0:
            nodes_in_edges = torch.unique(edge_index.flatten()).cpu().numpy()
            non_isolated = len(nodes_in_edges)
            degrees = np.bincount(edge_index.flatten().cpu().numpy(), minlength=n)
            avg_degree = float(degrees[nodes_in_edges].mean()) if non_isolated > 0 else 0.0
            isolated_ratio = float((n - non_isolated) / n)
        else:
            avg_degree = 0.0
            isolated_ratio = 1.0

        eligible, ff, nn, fn = _ratios(edge_index, fraud_set, normal_set)
        fraud_edge_lift = (ff / expected_ff_under_random) if expected_ff_under_random > 0 else 0.0

        results["relations"][rel] = {
            "edge_count": edge_count,
            "avg_degree": avg_degree,
            "isolated_ratio": isolated_ratio,
            "train_eligible_edges": int(eligible),
            "fraud_fraud_ratio": float(ff),
            "normal_normal_ratio": float(nn),
            "fraud_normal_ratio": float(fn),
            "fraud_edge_lift": float(fraud_edge_lift),
        }

    return results


def save_csv(results, csv_path):
    rows = []
    for rel, m in results["relations"].items():
        row = {"relation": rel}
        row.update(m)
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def main(processed_dir="data/processed", out_dir="outputs/metrics", report_dir="outputs/reports"):
    df_path = os.path.join(processed_dir, "node_samples.csv")
    edge_path = os.path.join(processed_dir, "edge_index_dict.pt")
    assert os.path.exists(df_path), f"missing {df_path}"
    assert os.path.exists(edge_path), f"missing {edge_path}"

    df = pd.read_csv(df_path)
    assert "split" in df.columns and "label" in df.columns, "node_samples.csv needs 'split' and 'label'"

    edge_index_dict = torch.load(edge_path)

    results = compute_relation_quality(edge_index_dict, df)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "relation_quality.json")
    save_json(results, json_path)
    print(f"[Save] {json_path}")

    csv_path = os.path.join(report_dir, "relation_quality.csv")
    save_csv(results, csv_path)
    print(f"[Save] {csv_path}")

    print("\n=== Relation Quality (train-only labels) ===")
    print(f"Train fraud ratio: {results['meta']['train_fraud_ratio']:.4f}  "
          f"(expected ff under random: {results['meta']['expected_ff_under_random']:.4f})\n")
    for rel, m in results["relations"].items():
        print(f"  {rel:10s}  E={m['edge_count']:>7d}  deg={m['avg_degree']:>6.2f}  "
              f"iso={m['isolated_ratio']:.3f}  FF={m['fraud_fraud_ratio']:.4f}  "
              f"FN={m['fraud_normal_ratio']:.4f}  lift={m['fraud_edge_lift']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="outputs/metrics")
    parser.add_argument("--report-dir", default="outputs/reports")
    args = parser.parse_args()
    main(args.processed_dir, args.out_dir, args.report_dir)
