import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Tuple


def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)

    metrics = {}

    metrics["pr_auc"] = average_precision_score(y_true, y_score)
    metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["accuracy"] = np.mean(y_true == y_pred)

    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "macro_f1",
) -> Tuple[float, float]:
    best_threshold = 0.5
    best_score = 0.0

    for threshold in np.arange(0.0, 1.01, 0.01):
        y_pred = (y_score >= threshold).astype(int)

        if metric == "macro_f1":
            score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        elif metric == "f1_weighted":
            score = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        elif metric == "roc_auc":
            score = roc_auc_score(y_true, y_score)
            return 0.5, score  # threshold doesn't matter for AUC
        elif metric == "pr_auc":
            score = average_precision_score(y_true, y_score)
            return 0.5, score
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    if prefix:
        print(f"\n{prefix}")
    print("-" * 60)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("-" * 60)
