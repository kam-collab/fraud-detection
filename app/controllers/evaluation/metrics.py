"""Evaluation: threshold-free ranking metrics, operating-point metrics, thresholds, costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from app.utils import config


def ranking_metrics(y, score) -> dict:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "brier": float("nan")}
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
    }


def threshold_metrics(y, score, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / len(y)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "flag_rate": float(pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate(y, score, threshold: float) -> dict:
    return {
        **ranking_metrics(y, score),
        **threshold_metrics(y, score, threshold),
        "n": int(len(y)),
        "fraud_rate": float(np.mean(y)),
    }


def threshold_for_precision(y, score, min_precision: float) -> float:
    """Lowest threshold (= highest recall) whose precision meets the floor.

    Chosen on the validation window only, then frozen. A minimum of 30 flagged
    transactions is required so the choice is not made on a handful of rows.
    """
    precision, recall, thresholds = precision_recall_curve(y, score)
    score = np.asarray(score)
    ok = [t for p, t in zip(precision[:-1], thresholds) if p >= min_precision and (score >= t).sum() >= 30]
    return float(min(ok)) if ok else float(np.quantile(score, 0.999))


def threshold_for_max_f1(y, score) -> float:
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def bootstrap_ci(y, score, threshold: float, n_boot: int = 300, seed: int = config.SEED) -> dict:
    """95% intervals for the headline metrics, so later drops can be judged against noise."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = evaluate(y[idx], score[idx], threshold)
        rows.append([m["pr_auc"], m["roc_auc"], m["precision"], m["recall"], m["f1"]])
    lo, hi = np.nanpercentile(np.array(rows), [2.5, 97.5], axis=0)
    names = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    return {n: [float(a), float(b)] for n, a, b in zip(names, lo, hi)}


def business_metrics(y, score, amount, review_threshold: float, block_threshold: float) -> dict:
    """Money and operations view of the three-band policy (approve / review / block).

    Uses the illustrative cost assumptions in config. Fraud that is approved is lost in
    full; reviewed fraud is caught at REVIEW_CATCH_RATE; blocked genuine payments are
    false declines.
    """
    y, score, amount = np.asarray(y), np.asarray(score), np.asarray(amount, dtype=float)
    block = score >= block_threshold
    review = (score >= review_threshold) & ~block
    approve = ~block & ~review
    fraud, genuine = y == 1, y == 0
    fraud_amount = amount[fraud].sum()
    loss_approved = amount[approve & fraud].sum()
    loss_review_missed = (1 - config.REVIEW_CATCH_RATE) * amount[review & fraud].sum()
    fraud_loss = loss_approved + loss_review_missed
    false_declines = int((block & genuine).sum())
    false_decline_cost = (
        config.FALSE_DECLINE_MARGIN * amount[block & genuine].sum() + config.FALSE_DECLINE_FIXED * false_declines
    )
    review_cost = config.REVIEW_COST * int(review.sum())
    return {
        "transactions": int(len(y)),
        "approval_rate": float(approve.mean()),
        "review_queue": int(review.sum()),
        "review_rate": float(review.mean()),
        "blocked": int(block.sum()),
        "false_declines": false_declines,
        "false_decline_amount": float(amount[block & genuine].sum()),
        "genuine_sent_to_review": int((review & genuine).sum()),
        "fraud_amount_total": float(fraud_amount),
        "fraud_loss_amount": float(fraud_loss),
        "fraud_amount_prevented": float(fraud_amount - fraud_loss),
        "fraud_value_recall": float(1 - fraud_loss / fraud_amount) if fraud_amount else float("nan"),
        "total_cost": float(fraud_loss + false_decline_cost + review_cost),
        "cost_without_model": float(fraud_amount),
    }


def pr_curve_points(y, score, n: int = 200) -> list[dict]:
    """Points for plotting, spaced evenly in recall so the high-precision end is not undersampled."""
    precision, recall, thresholds = precision_recall_curve(y, score)
    precision, recall = precision[:-1], recall[:-1]  # drop the (recall 0, precision 1) end point
    targets = np.linspace(recall.min(), recall.max(), n)
    order = np.argsort(recall)  # recall falls as the threshold rises
    idx = np.unique(order[np.clip(np.searchsorted(recall[order], targets), 0, len(order) - 1)])
    idx = idx[np.argsort(thresholds[idx])]
    return [
        {"threshold": float(thresholds[i]), "precision": float(precision[i]), "recall": float(recall[i])}
        for i in idx
        if recall[i] > 0
    ]


def recall_by_group(y, score, group, threshold: float) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "flag": np.asarray(score) >= threshold, "g": np.asarray(group)})
    df = df[df["y"] == 1]
    return df.groupby("g")["flag"].agg(frauds="size", recall="mean").reset_index().rename(columns={"g": "group"})
