"""Metrics: PR-AUC / Macro-F1 / ROC-AUC / Precision / Recall / Accuracy + threshold search."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, roc_auc_score,
    precision_score, recall_score, accuracy_score, confusion_matrix,
    precision_recall_curve,
)


def calculate_metrics(y_true, y_score, y_pred):
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = 0.0
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except Exception:
        roc_auc = 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    recall_pos = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_neg = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    g_mean = float(np.sqrt(recall_pos * recall_neg))
    return {
        "pr_auc": float(pr_auc),
        "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(acc),
        "recall_pos": float(recall_pos),
        "recall_neg": float(recall_neg),
        "g_mean": g_mean,
    }


def find_best_threshold(y_true, y_score, metric="macro_f1"):
    """Search threshold on valid set."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t, best_v = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        v = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if v > best_v:
            best_v = v
            best_t = float(t)
    return best_t, float(best_v)


def print_metrics(metrics, title="Metrics"):
    print(f"\n{title}\n" + "-" * 60)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print("-" * 60)
