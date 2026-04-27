"""
zero_shot.py
------------
Zero-shot AI-generated text detectors based on unsupervised signals.

Detectors
~~~~~~~~~
* **PerplexityThresholdDetector** – thresholds GPT-2 perplexity.  AI text
  tends to have *lower* perplexity (more predictable), so a low perplexity
  triggers an AI prediction.
* **LLRDetector** – uses the log-likelihood ratio between a scoring model and
  a reference model (DetectGPT-style).  Large positive LLR suggests AI text.
* **RobertaDetector** – wraps the ``openai-community/roberta-base-openai-detector``
  (or compatible) model for zero-shot classification.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from transformers import pipeline

from ..features.perplexity import PerplexityFeatures


class PerplexityThresholdDetector:
    """Zero-shot detector: classify as AI-generated if perplexity < threshold.

    Parameters
    ----------
    threshold:
        Perplexity threshold.  Documents with perplexity below this value are
        classified as AI-generated (score = 1 − perplexity/threshold, clipped
        to [0, 1]).
    scoring_model_name:
        Causal LM used to compute perplexity.
    reference_model_name:
        Reference LM for log-likelihood ratio computation.
    device:
        Torch device.
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
        """Return prediction dicts with keys ``id`` (empty), ``score``, ``label``.

        Parameters
        ----------
        texts:
            Raw text strings.

        Returns
        -------
        List of dicts ``{"score": float, "label": int}``.
        """
        results = []
        for text in texts:
            feats = self._ppl.extract(text)
            ppl = feats["perplexity"]
            if np.isnan(ppl):
                score = 0.5
            else:
                # Score ∈ [0, 1]: higher → more likely AI-generated
                score = float(np.clip(1.0 - ppl / self.threshold, 0.0, 1.0))
            results.append({"score": score, "label": int(score >= 0.5)})
        return results


class LLRDetector:
    """Zero-shot detector using the log-likelihood ratio (DetectGPT-style).

    A large positive LLR (scoring model assigns higher log-likelihood than
    the reference model) indicates the text is more ``in-distribution'' for
    the scoring (larger, instruction-tuned) LM, suggesting AI generation.

    Parameters
    ----------
    llr_threshold:
        LLR value above which text is classified as AI-generated.
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
        """Return prediction dicts with keys ``score`` and ``label``."""
        results = []
        for text in texts:
            feats = self._ppl.extract(text)
            llr = feats["llr"]
            if np.isnan(llr):
                score = 0.5
            else:
                # Sigmoid to map LLR to [0, 1]
                score = float(1.0 / (1.0 + np.exp(-llr)))
            results.append({"score": score, "label": int(score >= 0.5)})
        return results


class RobertaDetector:
    """Zero-shot detector using the RoBERTa-based OpenAI text detector.

    Wraps ``openai-community/roberta-base-openai-detector`` (or any compatible
    binary text-classification model).

    Parameters
    ----------
    model_name:
        HuggingFace model ID.
    device:
        Torch device (``-1`` for CPU, ``0`` for first GPU).
    batch_size:
        Inference batch size.
    """

    _FAKE_LABEL = "LABEL_1"  # "Fake" / AI-generated label

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
        """Return prediction dicts with keys ``score`` and ``label``."""
        outputs = self._pipe(texts, batch_size=self.batch_size)
        results = []
        for out in outputs:
            # Score is probability of the "Fake"/AI label
            if out["label"] == self._FAKE_LABEL:
                score = float(out["score"])
            else:
                score = 1.0 - float(out["score"])
            results.append({"score": score, "label": int(score >= 0.5)})
        return results
