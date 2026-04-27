"""
zero_shot.py
------------
基于无监督信号的零样本 AI 生成文本检测器。

检测器说明
~~~~~~~~~~
* **PerplexityThresholdDetector** —— 以 GPT-2 困惑度为阈值。
  AI 文本倾向于具有*更低*困惑度（更可预测），
  因此低困惑度将触发 AI 预测。
* **LLRDetector** —— 使用评分模型与参考模型之间的
  对数似然比（DetectGPT 风格）。
  较大的正 LLR 值表明文本为 AI 生成。
* **RobertaDetector** —— 封装 ``openai-community/roberta-base-openai-detector``
  （或兼容模型）进行零样本分类。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from transformers import pipeline

from ..features.perplexity import PerplexityFeatures


class PerplexityThresholdDetector:
    """零样本检测器：困惑度低于阈值时判定为 AI 生成。

    参数
    ----
    threshold:
        困惑度阈值。困惑度低于此值的文档将被判定为 AI 生成
        （得分 = 1 − 困惑度/阈值，截断至 [0, 1]）。
    scoring_model_name:
        用于计算困惑度的因果语言模型。
    reference_model_name:
        用于对数似然比计算的参考语言模型。
    device:
        Torch 设备。
    """

    def __init__(
        self,
        threshold: float = 50.0,
        scoring_model_name: str = "gpt2",
        reference_model_name: str = "distilgpt2",
        device: Optional[str] = None,
    ) -> None:
        self.threshold = threshold
        self._ppl = PerplexityFeatures(
            scoring_model_name=scoring_model_name,
            reference_model_name=reference_model_name,
            device=device,
        )

    def predict(self, texts: List[str]) -> List[dict]:
        """返回包含 ``id``（空）、``score``、``label`` 键的预测字典列表。

        参数
        ----
        texts:
            原始文本字符串列表。

        返回
        ----
        字典列表，格式为 ``{"score": float, "label": int}``。
        """
        results = []
        for text in texts:
            feats = self._ppl.extract(text)
            ppl = feats["perplexity"]
            if np.isnan(ppl):
                score = 0.5
            else:
                # 得分 ∈ [0, 1]：越高越可能为 AI 生成
                score = float(np.clip(1.0 - ppl / self.threshold, 0.0, 1.0))
            results.append({"score": score, "label": int(score >= 0.5)})
        return results


class LLRDetector:
    """使用对数似然比的零样本检测器（DetectGPT 风格）。

    较大的正 LLR（评分模型对文本赋予更高对数似然）表明该文本更符合
    评分（更大、经指令调优）语言模型的分布，提示文本为 AI 生成。

    参数
    ----
    llr_threshold:
        高于此 LLR 值时，文本被判定为 AI 生成。
    """

    def __init__(
        self,
        llr_threshold: float = 0.0,
        scoring_model_name: str = "gpt2",
        reference_model_name: str = "distilgpt2",
        device: Optional[str] = None,
    ) -> None:
        self.llr_threshold = llr_threshold
        self._ppl = PerplexityFeatures(
            scoring_model_name=scoring_model_name,
            reference_model_name=reference_model_name,
            device=device,
        )

    def predict(self, texts: List[str]) -> List[dict]:
        """返回包含 ``score`` 和 ``label`` 键的预测字典列表。"""
        results = []
        for text in texts:
            feats = self._ppl.extract(text)
            llr = feats["llr"]
            if np.isnan(llr):
                score = 0.5
            else:
                # Sigmoid 将 LLR 映射到 [0, 1]
                score = float(1.0 / (1.0 + np.exp(-llr)))
            results.append({"score": score, "label": int(score >= 0.5)})
        return results


class RobertaDetector:
    """使用基于 RoBERTa 的 OpenAI 文本检测器的零样本检测器。

    封装 ``openai-community/roberta-base-openai-detector``
    （或任何兼容的二元文本分类模型）。

    参数
    ----
    model_name:
        HuggingFace 模型 ID。
    device:
        Torch 设备（``-1`` 表示 CPU，``0`` 表示第一块 GPU）。
    batch_size:
        推理批次大小。
    """

    _FAKE_LABEL = "LABEL_1"  # "伪造" / AI 生成标签

    def __init__(
        self,
        model_name: str = "openai-community/roberta-base-openai-detector",
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _device_idx = 0 if _device == "cuda" else -1
        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            device=_device_idx,
            truncation=True,
            max_length=512,
        )
        self.batch_size = batch_size

    def predict(self, texts: List[str]) -> List[dict]:
        """返回包含 ``score`` 和 ``label`` 键的预测字典列表。"""
        outputs = self._pipe(texts, batch_size=self.batch_size)
        results = []
        for out in outputs:
            # 得分为"伪造"/AI 标签的概率
            if out["label"] == self._FAKE_LABEL:
                score = float(out["score"])
            else:
                score = 1.0 - float(out["score"])
            results.append({"score": score, "label": int(score >= 0.5)})
        return results
