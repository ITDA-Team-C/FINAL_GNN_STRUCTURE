import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import load_config, save_json


def find_optimal_threshold(checkpoint_path):
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

    valid_mask = torch.BoolTensor(nodes_df["split"] == "valid").to(device)

    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

    from src.models.cage_rf_gnn import CAGERF_GNN

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
        # Handle tuple return (logits, aux_logits_dict) or just logits
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits.detach().cpu().numpy()

    y_score = 1 / (1 + np.exp(-logits))
    valid_mask_np = valid_mask.cpu().numpy()
    y_true = y[valid_mask].cpu().numpy()
    y_score_valid = y_score[valid_mask_np]

    print("[Threshold Search] Valid set에서 최적 threshold 탐색 중...")

    best_threshold = 0.5
    best_f1 = 0.0
    results = []

    for threshold in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_score_valid >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        results.append({
            "threshold": float(threshold),
            "macro_f1": float(f1),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

        print(f"  Threshold {threshold:.2f}: F1={f1:.4f}")

    print(f"\n[Best] Threshold={best_threshold:.2f}, Macro F1={best_f1:.4f}")

    os.makedirs("outputs", exist_ok=True)
    results_path = os.path.join("outputs", "threshold_search_results.json")
    save_json({"best_threshold": best_threshold, "best_f1": best_f1, "results": results}, results_path)
    print(f"[Save] {results_path}")

    return best_threshold


if __name__ == "__main__":
    checkpoint_path = "outputs/best_model_cage_rf_gnn.pt"
    find_optimal_threshold(checkpoint_path)
