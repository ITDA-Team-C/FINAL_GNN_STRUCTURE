import os
import pandas as pd
from src.utils import set_seed

set_seed(42)

CONFIG = {
    "interim_dir": "data/interim",
    "input_file": "raw_data.csv",
}


def label_convert():
    input_path = os.path.join(CONFIG["interim_dir"], CONFIG["input_file"])

    print(f"[Load] {input_path} 로딩 중...")
    df = pd.read_csv(input_path)

    print(f"  원본 라벨 분포:")
    print(f"  {df['label'].value_counts().to_dict()}")

    df_converted = df.copy()
    df_converted["label"] = df_converted["label"].map({-1: 1, 1: 0})

    assert set(df_converted["label"].unique()).issubset(
        {0, 1}
    ), "라벨 변환 실패: {0, 1}만 존재해야 함"

    print(f"\n[Convert] 라벨 변환 완료:")
    print(f"  변환 규칙: -1 (사기) → 1, 1 (정상) → 0")
    print(f"  변환 후 라벨 분포:")
    print(f"  {df_converted['label'].value_counts().to_dict()}")

    pos_count = (df_converted["label"] == 1).sum()
    neg_count = (df_converted["label"] == 0).sum()
    print(f"\n  Positive (사기): {pos_count} ({pos_count / len(df_converted) * 100:.2f}%)")
    print(f"  Negative (정상): {neg_count} ({neg_count / len(df_converted) * 100:.2f}%)")

    save_path = os.path.join(CONFIG["interim_dir"], "labeled_data.csv")
    df_converted.to_csv(save_path, index=False)
    print(f"\n[Save] {save_path}")

    return df_converted


if __name__ == "__main__":
    df = label_convert()
    print("\n[Done] Phase 2-2: Label Conversion 완료")
