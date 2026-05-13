"""Check fraud_ratio of sampled data vs original (competition rule: ±2%p).

Usage:
    python check_fraud_ratio.py
    python check_fraud_ratio.py --tolerance 2.0
    python check_fraud_ratio.py --original data/raw/yelp_zip.csv
"""
import argparse
import os
import sys

import pandas as pd


def _fraud_ratio(labels):
    """Return fraud_ratio (%) regardless of label convention.

    Supports both {-1, 1} (raw: -1=fraud, 1=legit) and {0, 1} (converted: 1=fraud).
    """
    unique = set(pd.Series(labels).dropna().unique().tolist())
    if unique.issubset({-1, 1}):
        fraud_mask = pd.Series(labels) == -1
    elif unique.issubset({0, 1}):
        fraud_mask = pd.Series(labels) == 1
    else:
        raise ValueError(f"Unknown label set: {unique}")
    return float(fraud_mask.sum()) / len(labels) * 100.0


def _read_or_none(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original",
        default="data/interim/raw_data.csv",
        help="Path to original (full) dataset CSV. Falls back to data/raw/yelp_zip.csv.",
    )
    parser.add_argument(
        "--sampled",
        default="data/processed/sampled_reviews.csv",
        help="Path to sampled dataset CSV (must contain 'label' and optionally 'split').",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Allowed deviation in percentage points (default: 2.0).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Fraud Ratio Distribution Check")
    print("=" * 70)

    original_df = _read_or_none(args.original)
    if original_df is None:
        fallback = "data/raw/yelp_zip.csv"
        print(f"[Info] {args.original} 없음 → fallback {fallback}")
        original_df = _read_or_none(fallback)
    if original_df is None:
        print(f"[ERROR] 원본 CSV를 찾을 수 없습니다 ({args.original} / data/raw/yelp_zip.csv)")
        sys.exit(2)

    sampled_df = _read_or_none(args.sampled)
    if sampled_df is None:
        print(f"[ERROR] 샘플 CSV를 찾을 수 없습니다: {args.sampled}")
        sys.exit(2)

    original_ratio = _fraud_ratio(original_df["label"])
    sampled_ratio = _fraud_ratio(sampled_df["label"])
    diff = sampled_ratio - original_ratio

    print(f"\n[Original] {args.original}")
    print(f"  총 리뷰: {len(original_df):,}")
    print(f"  fraud_ratio: {original_ratio:.2f}%")

    print(f"\n[Sampled]  {args.sampled}")
    print(f"  총 리뷰: {len(sampled_df):,}")
    print(f"  fraud_ratio: {sampled_ratio:.2f}%")
    print(f"  원본 대비: {diff:+.2f}%p")

    if "split" in sampled_df.columns:
        print("\n[Per-Split fraud_ratio]")
        for split_name in ["train", "valid", "test"]:
            sub = sampled_df[sampled_df["split"] == split_name]
            if len(sub) == 0:
                continue
            ratio = _fraud_ratio(sub["label"])
            d = ratio - original_ratio
            print(f"  {split_name:5s}: n={len(sub):>6,}  fraud_ratio={ratio:.2f}%  (원본 {d:+.2f}%p)")

    print("\n" + "=" * 70)
    print(f"Tolerance: ±{args.tolerance:.2f}%p")
    if abs(diff) <= args.tolerance:
        print(f"[PASS] |diff| = {abs(diff):.2f}%p ≤ {args.tolerance:.2f}%p")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"[FAIL] |diff| = {abs(diff):.2f}%p > {args.tolerance:.2f}%p")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
