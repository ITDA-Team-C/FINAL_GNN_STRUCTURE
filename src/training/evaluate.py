import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_config, save_json
from src.utils.metrics import calculate_metrics, print_metrics


def evaluate_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_config("configs/default.yaml")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    nodes_df = pd.read_csv(os.path.join(config["data_processed"], "node_samples.csv"))
    features = np.load(os.path.join(config["data_processed"], "features.npy"))
    edge_index_dict = torch.load(os.path.join(config["data_processed"], "edge_index_dict.pt"))

    x = torch.FloatTensor(features).to(device)
    y = torch.LongTensor(nodes_df["label"].values).to(device)

    test_mask = torch.BoolTensor(nodes_df["split"] == "test").to(device)

    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

    from src.models.cage_rf_gnn import CAGERF_GNN

    # 새로 학습된 모델은 항상 config에서 읽기
    cage_rf_config = config.get("cage_rf", {})
    model = CAGERF_GNN(x.shape[1], hidden_dim=cage_rf_config.get("hidden_dim", 128),
                      num_layers=cage_rf_config.get("num_layers", 3),
                      dropout=cage_rf_config.get("dropout", 0.3),
                      use_gating=cage_rf_config.get("use_gating", True))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(x, edge_index_dict)
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits.detach().cpu().numpy()

    y_score = 1 / (1 + np.exp(-logits))
    y_true = y.cpu().numpy()

    # Load best threshold from threshold optimization
    best_threshold = 0.5
    threshold_results_path = os.path.join("outputs", "threshold_search_results.json")
    if os.path.exists(threshold_results_path):
        with open(threshold_results_path, 'r') as f:
            threshold_data = json.load(f)
            best_threshold = threshold_data["best_threshold"]
            print(f"[Threshold] Best threshold 로드: {best_threshold:.3f}")
    else:
        print(f"[Threshold] 파일 없음, 기본값 0.5 사용")

    y_pred = (y_score >= best_threshold).astype(int)

    test_mask_np = test_mask.cpu().numpy()
    metrics = calculate_metrics(y_true[test_mask_np], y_score[test_mask_np], y_pred[test_mask_np])

    print(f"\n[Evaluation] {checkpoint_path}")
    print_metrics(metrics)

    cm = confusion_matrix(y_true[test_mask_np], y_pred[test_mask_np])
    print(f"\nConfusion Matrix:")
    print(cm)

    report = classification_report(y_true[test_mask_np], y_pred[test_mask_np])
    print(f"\nClassification Report:")
    print(report)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model_cage_rf_gnn.pt")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint)
