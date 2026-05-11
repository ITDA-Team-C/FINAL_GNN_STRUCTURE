"""Training loop for Amazon fraud detection."""
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.optim as optim

from amazon.src.data_loader import load_amazon, summary, RELATION_NAMES
from amazon.src.metrics import calculate_metrics, find_best_threshold, print_metrics
from amazon.src.models import (
    MLP, GCN, GAT, GraphSAGE, CAGECareRF, FocalLoss, FocalAuxLoss,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, in_dim: int, hidden_dim=128, num_layers=3, dropout=0.3):
    if name == "mlp":
        return MLP(in_dim, hidden_dim, dropout)
    if name == "gcn":
        return GCN(in_dim, hidden_dim, num_layers, dropout)
    if name == "gat":
        return GAT(in_dim, hidden_dim, num_layers, num_heads=8, dropout=dropout)
    if name == "graphsage":
        return GraphSAGE(in_dim, hidden_dim, num_layers, dropout)
    if name == "cage_carerf":
        return CAGECareRF(in_dim, RELATION_NAMES, hidden_dim, num_layers, dropout,
                          K=3, use_skip=True, use_gating=True, use_aux_loss=True,
                          use_care=True, care_top_k=10)
    if name == "cage_carerf_no_care":
        return CAGECareRF(in_dim, RELATION_NAMES, hidden_dim, num_layers, dropout,
                          K=3, use_skip=True, use_gating=True, use_aux_loss=True,
                          use_care=False)
    if name == "cage_carerf_no_aux":
        return CAGECareRF(in_dim, RELATION_NAMES, hidden_dim, num_layers, dropout,
                          K=3, use_skip=True, use_gating=True, use_aux_loss=False,
                          use_care=True, care_top_k=10)
    raise ValueError(f"Unknown model: {name}")


def train_one(model, x, y, edge_index_dict, train_mask, valid_mask, test_mask,
              device, epochs=200, lr=1e-3, patience=20, val_interval=5,
              focal_alpha=0.75, focal_gamma=2.0, aux_weight=0.3):
    model = model.to(device)
    x = x.to(device); y = y.to(device)
    train_mask = train_mask.to(device); valid_mask = valid_mask.to(device); test_mask = test_mask.to(device)
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

    loss_fn = FocalAuxLoss(FocalLoss(alpha=focal_alpha, gamma=focal_gamma), aux_weight=aux_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1, best_state, no_imp = 0.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        out = model(x, edge_index_dict)
        if isinstance(out, tuple):
            logits, aux = out
        else:
            logits, aux = out, None
        aux_sub = {k: v[train_mask] for k, v in aux.items()} if aux else None
        loss = loss_fn(logits[train_mask], y[train_mask], aux_sub)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if ep % val_interval == 0:
            model.eval()
            with torch.no_grad():
                out = model(x, edge_index_dict)
                logits = out[0] if isinstance(out, tuple) else out
                y_score = torch.sigmoid(logits).cpu().numpy()
            y_true = y.cpu().numpy()
            v_mask = valid_mask.cpu().numpy()
            m = calculate_metrics(y_true[v_mask], y_score[v_mask], (y_score[v_mask] >= 0.5).astype(int))
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
                marker = "✓"
            else:
                no_imp += 1
                marker = ""
            print(f"  ep {ep:3d}  loss {loss.item():.4f}  val_F1 {m['macro_f1']:.4f}  val_PR {m['pr_auc']:.4f}  {marker}")
            if no_imp >= patience:
                print(f"  early stopping at ep {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index_dict)
        logits = out[0] if isinstance(out, tuple) else out
        y_score = torch.sigmoid(logits).cpu().numpy()
    y_true = y.cpu().numpy()
    v_mask = valid_mask.cpu().numpy(); t_mask = test_mask.cpu().numpy()

    best_t, _ = find_best_threshold(y_true[v_mask], y_score[v_mask], metric="macro_f1")
    valid_m = calculate_metrics(y_true[v_mask], y_score[v_mask], (y_score[v_mask] >= best_t).astype(int))
    test_m = calculate_metrics(y_true[t_mask], y_score[t_mask], (y_score[t_mask] >= best_t).astype(int))
    return best_t, valid_m, test_m


def run_single(model_name, mat_path, device, out_dir="amazon/outputs",
               epochs=200, hidden_dim=128, num_layers=3, dropout=0.3, lr=1e-3,
               patience=20, val_interval=5, seed=42):
    set_seed(seed)
    print(f"\n{'='*70}\n[Amazon] training {model_name}\n{'='*70}")
    x, y, edge_index_dict, train_mask, valid_mask, test_mask = load_amazon(mat_path, seed=seed)
    summary(x, y, edge_index_dict, train_mask, valid_mask, test_mask)
    model = build_model(model_name, x.shape[1], hidden_dim, num_layers, dropout)
    best_t, vm, tm = train_one(model, x, y, edge_index_dict,
                                 train_mask, valid_mask, test_mask, device,
                                 epochs=epochs, lr=lr, patience=patience, val_interval=val_interval)
    print_metrics(tm, f"Test metrics — {model_name}")

    os.makedirs(out_dir, exist_ok=True)
    result = {"dataset": "amazon", "model": model_name, "best_threshold": best_t,
              "valid_metrics": vm, "test_metrics": tm}
    out_path = os.path.join(out_dir, f"metrics_{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[Save] {out_path}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["mlp", "gcn", "gat", "graphsage",
                            "cage_carerf", "cage_carerf_no_care", "cage_carerf_no_aux"])
    p.add_argument("--mat-path", default="amazon/data/Amazon.mat")
    p.add_argument("--epochs", type=int, default=200)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_single(args.model, args.mat_path, device, epochs=args.epochs)
