"""
embeddings.py
-------------
用于 AI 生成文本检测的句子嵌入特征。

当*可疑*文档旁边还有*原始*文档时，
两者之间的语义相似度可以揭示可疑文本是否为原文的
改写或复述——这是生成式剽窃的强信号。

计算的特征
~~~~~~~~~~
* 文档级嵌入向量（来自句子 Transformer 的 768 维向量）
* 可疑文档与原始文档嵌入之间的余弦相似度（配对模式）
* 文档内句子嵌入的平均两两余弦相似度
  （自相似度——AI 文本往往具有更均匀的自相似性）
"""

from __future__ import annotations

from typing import List, Optional

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

for _nltk_pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_nltk_pkg}")
    except LookupError:
        nltk.download(_nltk_pkg, quiet=True)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingFeatures:
    """为生成文本检测计算基于嵌入的特征。

    参数
    ----
    model_name:
        HuggingFace / sentence-transformers 模型标识符。
    device:
        ``"cuda"`` 或 ``"cpu"``，默认使用 CPU。
    batch_size:
        句子编码器的批次大小。
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """为文本列表返回 (N, D) 嵌入矩阵。"""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def extract(
        self,
        text: str,
        source_text: Optional[str] = None,
    ) -> dict:
        """为 *text* 提取嵌入特征。

        参数
        ----
        text:
            可疑文档。
        source_text:
            可选的原始文档，用于配对比较。

        返回
        ----
        包含以下键的字典：

        * ``embedding``            —— (D,) numpy 数组（文档嵌入向量）
        * ``self_sim``             —— 文档内句子平均自相似度
        * ``source_sim``           —— 与原文的余弦相似度（不存在时为 NaN）
        """
        sentences = sent_tokenize(text) if text.strip() else [text]
        doc_emb = self.encode([text])[0]

        # 文档内自相似度
        if len(sentences) > 1:
            sent_embs = self.encode(sentences)  # (S, D)
            sim_matrix = cosine_similarity(sent_embs)
            # 排除对角线（自身相似度 = 1.0）
            mask = ~np.eye(len(sentences), dtype=bool)
            self_sim = float(sim_matrix[mask].mean())
        else:
            self_sim = float("nan")

        # 配对原文相似度
        if source_text and source_text.strip():
            src_emb = self.encode([source_text])[0]
            source_sim = float(cosine_similarity([doc_emb], [src_emb])[0, 0])
        else:
            source_sim = float("nan")

        return {
            "embedding": doc_emb,
            "self_sim": self_sim,
            "source_sim": source_sim,
        }

    def extract_batch(
        self,
        texts: List[str],
        source_texts: Optional[List[Optional[str]]] = None,
    ) -> List[dict]:
        """为文本列表批量提取特征。"""
        if source_texts is None:
            source_texts = [None] * len(texts)
        return [
            self.extract(t, s) for t, s in zip(texts, source_texts)
        ]
