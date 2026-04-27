"""
perplexity.py
-------------
Perplexity-based features for AI-generated text detection.

Methods implemented
~~~~~~~~~~~~~~~~~~~
* **GPT-2 perplexity** – lower perplexity under a causal LM suggests text
  that is more predictable, which is characteristic of LLM output.
* **Log-likelihood ratio (DetectGPT-style)** – difference in log-likelihood
  between a *scoring* model and a *reference* model.  Large positive values
  indicate the text is more likely under the scoring model than a generic
  reference, a signal of AI generation.
* **Burstiness** – variance of per-token log-probabilities.  Human text tends
  to have higher burstiness (alternating high and low surprise tokens) than
  uniformly fluent LLM output.
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
    """Compute perplexity and related features for a list of texts.

    Parameters
    ----------
    scoring_model_name:
        HuggingFace model ID for the primary causal LM.
    reference_model_name:
        HuggingFace model ID for the reference causal LM used in the
        log-likelihood ratio computation.
    max_length:
        Maximum number of tokens to consider per document.
    device:
        Torch device string (``"cuda"``, ``"cpu"``).  Defaults to CUDA if
        available.
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _token_log_probs(
        self,
        text: str,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
    ) -> List[float]:
        """Return per-token log-probabilities for *text*."""
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

        # Shift logits so each position predicts the next token
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_lps = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        return token_lps.squeeze(0).cpu().tolist()

    def _perplexity(self, log_probs: Sequence[float]) -> float:
        """Compute perplexity from a sequence of per-token log-probs."""
        if not log_probs:
            return float("nan")
        return math.exp(-sum(log_probs) / len(log_probs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict:
        """Extract all perplexity features for a single *text*.

        Returns
        -------
        dict with keys:

        * ``perplexity``         – GPT-2 perplexity (lower → more AI-like)
        * ``log_likelihood``     – mean log-probability under scoring model
        * ``llr``                – log-likelihood ratio (scoring − reference)
        * ``burstiness``         – variance of per-token log-probs
        * ``entropy``            – Shannon entropy of token log-prob distribution
        """
        score_lps = self._token_log_probs(
            text, self._scoring_tokenizer, self._scoring_model
        )
        ref_lps = self._token_log_probs(
            text, self._ref_tokenizer, self._ref_model
        )

        # Align lengths (may differ due to different vocabularies/tokenisers)
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
        """Extract features for a list of texts."""
        return [self.extract(t) for t in texts]
