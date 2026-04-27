"""
stylometric.py
--------------
用于 AI 生成文本检测的文体特征提取。

计算的特征
~~~~~~~~~~
* 词型词例比（TTR）与修正 TTR（CTTR）
* 句子长度（词数）的均值 / 标准差
* 词语长度（字符数）的均值 / 标准差
* 标点密度（每字符频率）
* 词汇丰富度（Hapax Legomena 比率）
* 功能词比率
* 平均句法树深度（通过句子长度启发式近似）
* 句子长度突发性
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import List

import nltk
import numpy as np

# 确保所需 NLTK 数据已下载
_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
}
for _pkg, _path in _NLTK_RESOURCES.items():
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

_STOP_WORDS = set(stopwords.words("english"))
_PUNCTUATION = set(string.punctuation)


class StylometricFeatures:
    """从原始文本中计算文体特征。

    所有方法均无状态；实例化时仅缓存停用词集合。
    """

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return sent_tokenize(text)

    @staticmethod
    def _words(text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalpha()]

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict:
        """为单篇 *text* 提取所有文体特征。

        返回
        ----
        包含以下浮点值键的字典：

        ``ttr``、``cttr``、``mean_sent_len``、``std_sent_len``、
        ``mean_word_len``、``std_word_len``、``punct_density``、
        ``hapax_ratio``、``func_word_ratio``、``sent_len_burstiness``
        """
        if not text or not text.strip():
            return {k: float("nan") for k in (
                "ttr", "cttr", "mean_sent_len", "std_sent_len",
                "mean_word_len", "std_word_len", "punct_density",
                "hapax_ratio", "func_word_ratio", "sent_len_burstiness",
            )}

        sentences = self._sentences(text)
        words = self._words(text)

        # ------ 词型词例比 ------
        n_tokens = len(words)
        n_types = len(set(words))
        ttr = n_types / n_tokens if n_tokens > 0 else float("nan")
        cttr = n_types / math.sqrt(2 * n_tokens) if n_tokens > 0 else float("nan")

        # ------ 句子长度 ------
        sent_lens = np.array([len(word_tokenize(s)) for s in sentences], dtype=float)
        mean_sent_len = float(np.mean(sent_lens)) if len(sent_lens) > 0 else float("nan")
        std_sent_len = float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0
        sent_len_burstiness = (
            float(np.var(sent_lens) / (np.mean(sent_lens) ** 2))
            if len(sent_lens) > 1 and np.mean(sent_lens) > 0
            else float("nan")
        )

        # ------ 词语长度 ------
        word_lens = np.array([len(w) for w in words], dtype=float)
        mean_word_len = float(np.mean(word_lens)) if len(word_lens) > 0 else float("nan")
        std_word_len = float(np.std(word_lens)) if len(word_lens) > 1 else 0.0

        # ------ 标点密度 ------
        n_chars = max(len(text), 1)
        n_punct = sum(1 for c in text if c in _PUNCTUATION)
        punct_density = n_punct / n_chars

        # ------ 词汇丰富度 ------
        word_freq = Counter(words)
        hapax = sum(1 for freq in word_freq.values() if freq == 1)
        hapax_ratio = hapax / n_tokens if n_tokens > 0 else float("nan")

        # ------ 功能词比率 ------
        func_words = [w for w in words if w in _STOP_WORDS]
        func_word_ratio = len(func_words) / n_tokens if n_tokens > 0 else float("nan")

        return {
            "ttr": ttr,
            "cttr": cttr,
            "mean_sent_len": mean_sent_len,
            "std_sent_len": std_sent_len,
            "mean_word_len": mean_word_len,
            "std_word_len": std_word_len,
            "punct_density": punct_density,
            "hapax_ratio": hapax_ratio,
            "func_word_ratio": func_word_ratio,
            "sent_len_burstiness": sent_len_burstiness,
        }

    def extract_batch(self, texts: List[str]) -> List[dict]:
        """为文本列表批量提取特征。"""
        return [self.extract(t) for t in texts]
