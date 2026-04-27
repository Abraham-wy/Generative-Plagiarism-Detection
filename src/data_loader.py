"""
data_loader.py
--------------
用于加载和保存 PAN 2026 生成式剽窃检测数据集的工具函数。

支持的输入格式
~~~~~~~~~~~~~~
JSONL（每行一个 JSON 对象）::

    {"id": "doc1", "text": "The quick brown fox ...", "label": 1}
    {"id": "doc2", "text": "Lorem ipsum ...",          "label": 0}

对于配对任务，可额外包含 ``source_text`` 字段::

    {"id": "doc3", "text": "...", "source_text": "...", "label": 1}

``label`` 在测试集（预测模式）中为可选字段。

XML（PAN 语料库格式）
~~~~~~~~~~~~~~~~~~~~~
每篇文档单独存放为目录下的 ``<id>.txt`` 文件，
配套一个 ``truth.jsonl``（或 ``truth.xml``）存放标签。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import pandas as pd


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _parse_record(raw: dict) -> dict:
    """将原始 JSON 记录规范化为统一模式。"""
    record: dict = {
        "id": str(raw.get("id", "")),
        "text": str(raw.get("text", "")),
        "source_text": str(raw.get("source_text", "")) if raw.get("source_text") else None,
    }
    if "label" in raw:
        record["label"] = int(raw["label"])
    return record


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> List[dict]:
    """加载 JSONL 文件，返回规范化记录字典的列表。"""
    records: List[dict] = []
    with jsonlines.open(str(path)) as reader:
        for raw in reader:
            records.append(_parse_record(raw))
    return records


def load_directory(
    text_dir: str | Path,
    truth_path: Optional[str | Path] = None,
) -> List[dict]:
    """加载纯文本文件目录，并可选地读取真值标签文件。

    参数
    ----
    text_dir:
        包含 ``<id>.txt`` 文件的目录。
    truth_path:
        可选，指向 JSONL 真值文件（映射 ``id`` → ``label``）的路径。

    返回
    ----
    记录字典列表（与 :func:`load_jsonl` 返回的格式相同）。
    """
    text_dir = Path(text_dir)
    labels: Dict[str, int] = {}
    if truth_path is not None:
        with jsonlines.open(str(truth_path)) as reader:
            for row in reader:
                labels[str(row["id"])] = int(row["label"])

    records: List[dict] = []
    for txt_file in sorted(text_dir.glob("*.txt")):
        doc_id = txt_file.stem
        text = txt_file.read_text(encoding="utf-8")
        record: dict = {"id": doc_id, "text": text, "source_text": None}
        if doc_id in labels:
            record["label"] = labels[doc_id]
        records.append(record)
    return records


def to_dataframe(records: List[dict]) -> pd.DataFrame:
    """将记录字典列表转换为 :class:`pandas.DataFrame`。"""
    return pd.DataFrame(records)


def save_predictions(
    predictions: List[dict],
    output_path: str | Path,
) -> None:
    """将预测结果保存为 JSONL 文件。

    *predictions* 中的每个条目至少需包含 ``id`` 和 ``score``
    （[0, 1] 范围内的浮点数，表示 AI 生成的概率）。
    ``label`` 键（0 或 1）将根据硬阈值 0.5 自动添加。

    参数
    ----
    predictions:
        包含 ``id`` 和 ``score`` 键的字典列表。
    output_path:
        目标文件路径（文件将被创建或覆盖）。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(str(output_path), mode="w") as writer:
        for pred in predictions:
            record = {
                "id": pred["id"],
                "score": float(pred["score"]),
                "label": int(pred["score"] >= 0.5),
            }
            writer.write(record)


def load_predictions(path: str | Path) -> List[dict]:
    """加载先前保存的预测 JSONL 文件。"""
    with jsonlines.open(str(path)) as reader:
        return list(reader)
