"""
stylometric.py
--------------
Stylometric feature extraction for AI-generated text detection.

Features computed
~~~~~~~~~~~~~~~~~
* Type-token ratio (TTR) and corrected TTR (CTTR)
* Mean / std of sentence length (word count)
* Mean / std of word length (character count)
* Punctuation density (per-character rate)
* Vocabulary richness (hapax legomena ratio)
* Function-word ratio
* Average parse-tree depth (approximated via sentence length heuristic)
* Sentence-length burstiness
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import List

import nltk
import numpy as np

# Ensure required NLTK data is available
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
    """Compute stylometric features from raw text.

    All methods are stateless; instantiation simply caches the stop-word set.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return sent_tokenize(text)

    @staticmethod
    def _words(text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalpha()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict:
        """Extract all stylometric features for a single *text*.

        Returns
        -------
        dict with the following float-valued keys:

        ``ttr``, ``cttr``, ``mean_sent_len``, ``std_sent_len``,
        ``mean_word_len``, ``std_word_len``, ``punct_density``,
        ``hapax_ratio``, ``func_word_ratio``, ``sent_len_burstiness``
        """
        if not text or not text.strip():
            return {k: float("nan") for k in (
                "ttr", "cttr", "mean_sent_len", "std_sent_len",
                "mean_word_len", "std_word_len", "punct_density",
                "hapax_ratio", "func_word_ratio", "sent_len_burstiness",
            )}

        sentences = self._sentences(text)
        words = self._words(text)

        # ------ type-token ratio ------
        n_tokens = len(words)
        n_types = len(set(words))
        ttr = n_types / n_tokens if n_tokens > 0 else float("nan")
        cttr = n_types / math.sqrt(2 * n_tokens) if n_tokens > 0 else float("nan")

        # ------ sentence length ------
        sent_lens = np.array([len(word_tokenize(s)) for s in sentences], dtype=float)
        mean_sent_len = float(np.mean(sent_lens)) if len(sent_lens) > 0 else float("nan")
        std_sent_len = float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0
        sent_len_burstiness = (
            float(np.var(sent_lens) / (np.mean(sent_lens) ** 2))
            if len(sent_lens) > 1 and np.mean(sent_lens) > 0
            else float("nan")
        )

        # ------ word length ------
        word_lens = np.array([len(w) for w in words], dtype=float)
        mean_word_len = float(np.mean(word_lens)) if len(word_lens) > 0 else float("nan")
        std_word_len = float(np.std(word_lens)) if len(word_lens) > 1 else 0.0

        # ------ punctuation density ------
        n_chars = max(len(text), 1)
        n_punct = sum(1 for c in text if c in _PUNCTUATION)
        punct_density = n_punct / n_chars

        # ------ vocabulary richness ------
        word_freq = Counter(words)
        hapax = sum(1 for freq in word_freq.values() if freq == 1)
        hapax_ratio = hapax / n_tokens if n_tokens > 0 else float("nan")

        # ------ function-word ratio ------
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
        """Extract features for a list of texts."""
        return [self.extract(t) for t in texts]
