"""
embeddings.py
-------------
Sentence-embedding features for AI-generated text detection.

When a *source* document is available alongside the *suspicious* document,
semantic similarity between the two can reveal whether the suspicious text is
a paraphrase or rewrite of the source — a strong signal of generated plagiarism.

Features computed
~~~~~~~~~~~~~~~~~
* Document-level embedding (768-d vector from a sentence-transformer)
* Cosine similarity between suspicious and source embeddings (paired mode)
* Mean pairwise cosine similarity of sentence embeddings within a document
  (self-similarity — AI text tends to be more uniformly self-similar)
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
    """Compute embedding-based features for generated-text detection.

    Parameters
    ----------
    model_name:
        HuggingFace / sentence-transformers model identifier.
    device:
        ``"cuda"`` or ``"cpu"``.  Defaults to CPU.
    batch_size:
        Batch size for the sentence encoder.
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
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return a (N, D) embedding matrix for a list of texts."""
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
        """Extract embedding features for *text*.

        Parameters
        ----------
        text:
            The suspicious document.
        source_text:
            Optional source document for paired comparison.

        Returns
        -------
        dict with keys:

        * ``embedding``            – (D,) numpy array (document embedding)
        * ``self_sim``             – mean intra-document sentence similarity
        * ``source_sim``           – cosine similarity to source (NaN if absent)
        """
        sentences = sent_tokenize(text) if text.strip() else [text]
        doc_emb = self.encode([text])[0]

        # Intra-document self-similarity
        if len(sentences) > 1:
            sent_embs = self.encode(sentences)  # (S, D)
            sim_matrix = cosine_similarity(sent_embs)
            # Exclude diagonal (self-similarity = 1.0)
            mask = ~np.eye(len(sentences), dtype=bool)
            self_sim = float(sim_matrix[mask].mean())
        else:
            self_sim = float("nan")

        # Paired source similarity
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
        """Extract features for a list of texts."""
        if source_texts is None:
            source_texts = [None] * len(texts)
        return [
            self.extract(t, s) for t, s in zip(texts, source_texts)
        ]
