"""
perplexity.py
-------------
基于困惑度的 AI 生成文本检测特征。

已实现的方法
~~~~~~~~~~~~
* **GPT-2 困惑度** —— 在因果语言模型下困惑度越低，说明文本越"可预测"，
  这是大语言模型输出的典型特征。
* **对数似然比（DetectGPT 风格）** —— 评分模型与参考模型之间的
  对数似然差值。较大的正值表明文本在评分模型下比通用参考模型更可能出现，
  是 AI 生成的信号。
* **突发性（Burstiness）** —— 每词元对数概率的方差。
  人类文本往往具有更高的突发性（高低惊奇度的词元交替出现），
  而大语言模型输出则更为均匀流畅。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_SCORING_MODEL = "gpt2"
_REFERENCE_MODEL = "distilgpt2"
_MAX_LENGTH = 512


class PerplexityFeatures:
    """为文本列表计算困惑度及相关特征。

    参数
    ----
    scoring_model_name:
        主要因果语言模型的 HuggingFace 模型 ID。
    reference_model_name:
        用于对数似然比计算的参考因果语言模型的 HuggingFace 模型 ID。
    max_length:
        每篇文档最多考虑的词元数。
    device:
        Torch 设备字符串（``"cuda"`` 或 ``"cpu"``），默认优先使用 CUDA。
    """

    def __init__(
        self,
        scoring_model_name: str = _SCORING_MODEL,
        reference_model_name: str = _REFERENCE_MODEL,
        max_length: int = _MAX_LENGTH,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        self._scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model_name)
        self._scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model_name).to(
            self.device
        )
        self._scoring_model.eval()

        self._ref_tokenizer = AutoTokenizer.from_pretrained(reference_model_name)
        self._ref_model = AutoModelForCausalLM.from_pretrained(reference_model_name).to(
            self.device
        )
        self._ref_model.eval()

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _token_log_probs(
        self,
        text: str,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
    ) -> List[float]:
        """返回 *text* 中每个词元的对数概率列表。"""
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 2:
            return []

        with torch.no_grad():
            outputs = model(**inputs, labels=input_ids)

        # 平移 logits，使每个位置预测下一个词元
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_lps = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        return token_lps.squeeze(0).cpu().tolist()

    def _perplexity(self, log_probs: Sequence[float]) -> float:
        """由每词元对数概率序列计算困惑度。"""
        if not log_probs:
            return float("nan")
        return math.exp(-sum(log_probs) / len(log_probs))

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict:
        """为单篇 *text* 提取所有困惑度特征。

        返回
        ----
        包含以下键的字典：

        * ``perplexity``         —— GPT-2 困惑度（越低越像 AI 生成）
        * ``log_likelihood``     —— 评分模型下的平均对数概率
        * ``llr``                —— 对数似然比（评分模型 − 参考模型）
        * ``burstiness``         —— 每词元对数概率的方差
        * ``entropy``            —— 词元对数概率分布的香农熵
        """
        score_lps = self._token_log_probs(
            text, self._scoring_tokenizer, self._scoring_model
        )
        ref_lps = self._token_log_probs(
            text, self._ref_tokenizer, self._ref_model
        )

        # 对齐序列长度（不同词表/分词器可能导致长度不同）
        min_len = min(len(score_lps), len(ref_lps))

        if min_len == 0:
            return {
                "perplexity": float("nan"),
                "log_likelihood": float("nan"),
                "llr": float("nan"),
                "burstiness": float("nan"),
                "entropy": float("nan"),
            }

        score_lps_arr = np.array(score_lps[:min_len])
        ref_lps_arr = np.array(ref_lps[:min_len])

        ppl = self._perplexity(score_lps)
        mean_ll = float(np.mean(score_lps_arr))
        llr = float(np.mean(score_lps_arr - ref_lps_arr))
        burstiness = float(np.var(score_lps_arr))
        probs = np.exp(score_lps_arr)
        probs = np.clip(probs, 1e-12, 1.0)
        entropy = float(-np.sum(probs * np.log(probs)))

        return {
            "perplexity": ppl,
            "log_likelihood": mean_ll,
            "llr": llr,
            "burstiness": burstiness,
            "entropy": entropy,
        }

    def extract_batch(self, texts: List[str]) -> List[dict]:
        """为文本列表批量提取特征。"""
        return [self.extract(t) for t in texts]
