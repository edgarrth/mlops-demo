import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def calculate_metrics(y_true, scores, top_fraction: float = 0.10) -> dict:
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= 0.5).astype(int)
    k = max(1, int(np.ceil(len(y) * top_fraction)))
    top_idx = np.argsort(scores)[::-1][:k]
    positives = max(1, int(y.sum()))
    base_rate = float(y.mean())
    precision_at_k = float(y[top_idx].mean())
    recall_at_k = float(y[top_idx].sum() / positives)
    lift_at_k = float(precision_at_k / base_rate) if base_rate else 0.0
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision_at_10": precision_at_k,
        "recall_at_10": recall_at_k,
        "lift_at_10": lift_at_k,
    }


def quality_gate(metrics: dict, config: dict) -> None:
    limits = config["quality_gate"]
    failures = []
    for metric, config_key in [
        ("pr_auc", "min_pr_auc"),
        ("roc_auc", "min_roc_auc"),
        ("lift_at_10", "min_lift_at_10"),
    ]:
        if metrics[metric] < limits[config_key]:
            failures.append(f"{metric}={metrics[metric]:.4f} < {limits[config_key]}")
    if failures:
        raise RuntimeError("Quality gate falló: " + "; ".join(failures))
