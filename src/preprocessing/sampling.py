import os
import pandas as pd
import numpy as np
from collections import defaultdict
from src.utils import set_seed

set_seed(42)

CONFIG = {
    "interim_dir": "data/interim",
    "processed_dir": "data/processed",
    "input_file": "labeled_data.csv",
    "target_nodes": 25000,
    "min_nodes": 10000,
    "max_nodes": 50000,
    "random_state": 42,
}


def add_temporal_features(df):
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M")
    df["days_since_epoch"] = (df["date"] - pd.Timestamp("1970-01-01")).dt.days
    return df


def product_user_time_hybrid_sampling(df):
    print("[Sampling] Product-User-Time Hybrid Dense Sampling 시작...")

    np.random.seed(CONFIG["random_state"])

    df = add_temporal_features(df)

    print(f"  총 데이터: {len(df)}")
    print(f"  상품 수: {df['prod_id'].nunique()}")
    print(f"  사용자 수: {df['user_id'].nunique()}")
    print(f"  기간: {df['date'].min()} ~ {df['date'].max()}")

    prod_counts = df["prod_id"].value_counts()
    print(f"\n  [Product] 상위 10개 상품: {prod_counts.head(10).to_dict()}")

    top_products = set(prod_counts.head(max(100, len(df) // 1000)).index)
    print(f"  상위 Product 선택: {len(top_products)}")

    user_counts = df["user_id"].value_counts()
    print(f"\n  [User] 상위 10명 사용자: {user_counts.head(10).to_dict()}")

    top_users = set(user_counts.head(max(100, len(df) // 1000)).index)
    print(f"  활동량 많은 User 선택: {len(top_users)}")

    month_counts = df["year_month"].value_counts()
    print(f"\n  [Time] 상위 10개 월: {month_counts.head(10).to_dict()}")

    top_months = set(month_counts.head(max(12, len(df) // 50000)).index)
    print(f"  리뷰 집중 Month 선택: {len(top_months)}")

    product_mask = df["prod_id"].isin(top_products)
    user_mask = df["user_id"].isin(top_users)
    month_mask = df["year_month"].isin(top_months)

    sampled_df = df[product_mask | user_mask | month_mask].copy()

    print(f"\n  Union (product OR user OR month): {len(sampled_df)}")

    if len(sampled_df) > CONFIG["max_nodes"]:
        print(f"  [Reduce] {len(sampled_df)} → {CONFIG['max_nodes']} (max_nodes 초과)")
        sampled_df = sampled_df.sample(
            n=CONFIG["max_nodes"],
            random_state=CONFIG["random_state"]
        )

    if len(sampled_df) < CONFIG["min_nodes"]:
        print(f"  [Expand] {len(sampled_df)} → {min(len(df), CONFIG['min_nodes'])}")
        remaining = df[~df.index.isin(sampled_df.index)]
        need = min(CONFIG["min_nodes"] - len(sampled_df), len(remaining))
        extra = remaining.sample(n=need, random_state=CONFIG["random_state"])
        sampled_df = pd.concat([sampled_df, extra]).drop_duplicates()

    print(f"\n  최종 샘플: {len(sampled_df)}")
    print(f"  목표: {CONFIG['target_nodes']}, 범위: [{CONFIG['min_nodes']}, {CONFIG['max_nodes']}]")

    assert CONFIG["min_nodes"] <= len(sampled_df) <= CONFIG["max_nodes"], \
        f"샘플 크기 범위 초과: {len(sampled_df)}"

    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df["node_id"] = np.arange(len(sampled_df))

    return sampled_df


def train_val_test_split(df):
    print("\n[Split] Train/Valid/Test 분할...")

    from sklearn.model_selection import train_test_split

    train_ratio = 0.64
    valid_ratio = 0.16
    test_ratio = 0.20

    indices = np.arange(len(df))
    labels = df["label"].values

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=1 - train_ratio,
        stratify=labels,
        random_state=CONFIG["random_state"]
    )

    valid_size = valid_ratio / (valid_ratio + test_ratio)
    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - valid_size,
        stratify=labels[temp_idx],
        random_state=CONFIG["random_state"]
    )

    df["split"] = "train"
    df.loc[valid_idx, "split"] = "valid"
    df.loc[test_idx, "split"] = "test"

    train_mask = df["split"] == "train"
    valid_mask = df["split"] == "valid"
    test_mask = df["split"] == "test"

    print(f"  Train: {train_mask.sum()} ({train_mask.sum() / len(df) * 100:.1f}%)")
    print(f"  Valid: {valid_mask.sum()} ({valid_mask.sum() / len(df) * 100:.1f}%)")
    print(f"  Test:  {test_mask.sum()} ({test_mask.sum() / len(df) * 100:.1f}%)")

    print(f"\n  Train 라벨 분포: {df[train_mask]['label'].value_counts().to_dict()}")
    print(f"  Valid 라벨 분포: {df[valid_mask]['label'].value_counts().to_dict()}")
    print(f"  Test  라벨 분포: {df[test_mask]['label'].value_counts().to_dict()}")

    return df


def save_sampled_data(df):
    os.makedirs(CONFIG["processed_dir"], exist_ok=True)

    save_path = os.path.join(CONFIG["processed_dir"], "sampled_reviews.csv")
    df.to_csv(save_path, index=False)
    print(f"\n[Save] {save_path}")

    stats_path = os.path.join(CONFIG["processed_dir"], "sampling_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("=== Sampling Statistics ===\n\n")
        f.write(f"Total sampled nodes: {len(df)}\n")
        f.write(f"Target: {CONFIG['target_nodes']}\n")
        f.write(f"Range: [{CONFIG['min_nodes']}, {CONFIG['max_nodes']}]\n\n")

        f.write("Label Distribution:\n")
        f.write(df["label"].value_counts().to_string() + "\n\n")

        f.write("Split Distribution:\n")
        f.write(df["split"].value_counts().to_string() + "\n\n")

        f.write("Unique values:\n")
        f.write(f"Products: {df['prod_id'].nunique()}\n")
        f.write(f"Users: {df['user_id'].nunique()}\n")

    print(f"[Save] {stats_path}")


if __name__ == "__main__":
    input_path = os.path.join(CONFIG["interim_dir"], "labeled_data.csv")
    df = pd.read_csv(input_path)

    df = product_user_time_hybrid_sampling(df)
    df = train_val_test_split(df)
    save_sampled_data(df)

    print("\n[Done] Phase 2-3: Sampling 완료")
