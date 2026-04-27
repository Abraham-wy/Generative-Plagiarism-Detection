"""
evaluate.py
-----------
与 PAN 官方指标对应的评估工具函数。

计算的指标
~~~~~~~~~~
* 准确率（Accuracy）
* 宏平均 F1（Macro-F1，PAN 主要指标）
* 各类别二元 F1
* AUC-ROC
* 平均精度（AP）
* 混淆矩阵

所有函数均接受整数真实标签列表（或数组），
以及整数预测标签或浮点概率分数。
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
    """计算所有与 PAN 相关的评估指标。

    参数
    ----
    y_true:
        真实二元标签（0 = 人类，1 = AI 生成）。
    y_pred:
        预测二元标签。
    y_score:
        可选，正类的预测概率分数（AUC-ROC 和 AP 计算所需）。

    返回
    ----
    包含以下键的字典：``accuracy``、``f1_macro``、``f1_human``、``f1_ai``、
    ``roc_auc``（未提供 y_score 时为 NaN）、``average_precision``
    （未提供 y_score 时为 NaN）。
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
    title: str = "评估报告",
) -> None:
    """将格式化的评估报告打印到标准输出。"""
    metrics = compute_metrics(y_true, y_pred, y_score)
    cm = confusion_matrix(y_true, y_pred)

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  准确率            : {metrics['accuracy']:.4f}")
    print(f"  F1（宏平均）      : {metrics['f1_macro']:.4f}  ← 主要指标")
    print(f"  F1（人类）        : {metrics['f1_human']:.4f}")
    print(f"  F1（AI 生成）     : {metrics['f1_ai']:.4f}")
    if not np.isnan(metrics["roc_auc"]):
        print(f"  AUC-ROC          : {metrics['roc_auc']:.4f}")
    if not np.isnan(metrics["average_precision"]):
        print(f"  平均精度         : {metrics['average_precision']:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["人类", "AI 生成"]))
    print("混淆矩阵：")
    print("               预测：人类  预测：AI")
    print(f"  真实：人类     {cm[0,0]:5d}    {cm[0,1]:5d}")
    print(f"  真实：AI       {cm[1,0]:5d}    {cm[1,1]:5d}")
    print(sep)


def optimal_threshold(
    y_true: List[int],
    y_score: List[float],
    metric: str = "f1_macro",
) -> Tuple[float, float]:
    """在给定数据上寻找使 *metric* 最大的决策阈值。

    参数
    ----
    y_true:
        真实二元标签。
    y_score:
        预测概率分数。
    metric:
        ``"f1_macro"`` 或 ``"accuracy"`` 之一。

    返回
    ----
    元组 ``(最优阈值, 对应指标值)``。
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
            raise ValueError(f"未知指标：{metric!r}")
        if val > best_val:
            best_val = val
            best_thresh = float(thresh)

    return best_thresh, float(best_val)
