"""
evaluate.py
-----------
Evaluation utilities that mirror the PAN official metrics.

Metrics computed
~~~~~~~~~~~~~~~~
* Accuracy
* Macro-F1  (primary PAN metric)
* Binary F1 (per-class)
* AUC-ROC
* Average precision (AP)
* Confusion matrix

All functions accept lists (or arrays) of integer ground-truth labels and
either integer predicted labels or float probability scores.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_score: Optional[List[float]] = None,
) -> dict:
    """Compute all PAN-relevant evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 = human, 1 = AI-generated).
    y_pred:
        Predicted binary labels.
    y_score:
        Optional predicted probability scores for the positive class.
        Required for AUC-ROC and AP.

    Returns
    -------
    dict with keys: ``accuracy``, ``f1_macro``, ``f1_human``, ``f1_ai``,
    ``roc_auc`` (NaN if y_score not provided), ``average_precision``
    (NaN if y_score not provided).
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro"))
    f1_human = float(f1_score(y_true_arr, y_pred_arr, pos_label=0, average="binary"))
    f1_ai = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, average="binary"))

    if y_score is not None:
        y_score_arr = np.array(y_score)
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_score_arr))
        except ValueError:
            roc_auc = float("nan")
        try:
            ap = float(average_precision_score(y_true_arr, y_score_arr))
        except ValueError:
            ap = float("nan")
    else:
        roc_auc = float("nan")
        ap = float("nan")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_human": f1_human,
        "f1_ai": f1_ai,
        "roc_auc": roc_auc,
        "average_precision": ap,
    }


def print_report(
    y_true: List[int],
    y_pred: List[int],
    y_score: Optional[List[float]] = None,
    title: str = "Evaluation Report",
) -> None:
    """Print a formatted evaluation report to stdout."""
    metrics = compute_metrics(y_true, y_pred, y_score)
    cm = confusion_matrix(y_true, y_pred)

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Accuracy         : {metrics['accuracy']:.4f}")
    print(f"  F1 (macro)       : {metrics['f1_macro']:.4f}  ← primary metric")
    print(f"  F1 (human)       : {metrics['f1_human']:.4f}")
    print(f"  F1 (AI-generated): {metrics['f1_ai']:.4f}")
    if not np.isnan(metrics["roc_auc"]):
        print(f"  AUC-ROC          : {metrics['roc_auc']:.4f}")
    if not np.isnan(metrics["average_precision"]):
        print(f"  Avg Precision    : {metrics['average_precision']:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Human", "AI-generated"]))
    print("Confusion matrix:")
    print("             Pred Human  Pred AI")
    print(f"  True Human     {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"  True AI        {cm[1,0]:5d}    {cm[1,1]:5d}")
    print(sep)


def optimal_threshold(
    y_true: List[int],
    y_score: List[float],
    metric: str = "f1_macro",
) -> Tuple[float, float]:
    """Find the decision threshold that maximises *metric* on the given data.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_score:
        Predicted probability scores.
    metric:
        One of ``"f1_macro"``, ``"accuracy"``.

    Returns
    -------
    Tuple ``(best_threshold, best_metric_value)``.
    """
    thresholds = np.linspace(0.0, 1.0, 201)
    best_thresh = 0.5
    best_val = -np.inf

    for thresh in thresholds:
        y_pred = (np.array(y_score) >= thresh).astype(int).tolist()
        if metric == "f1_macro":
            val = f1_score(y_true, y_pred, average="macro", zero_division=0)
        elif metric == "accuracy":
            val = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")
        if val > best_val:
            best_val = val
            best_thresh = float(thresh)

    return best_thresh, float(best_val)
