"""
ensemble.py
-----------
Ensemble detector that combines multiple base detectors.

Strategy
~~~~~~~~
* **WeightedAverageEnsemble** – takes a weighted average of the ``score``
  fields from each component detector.
* **StackingEnsemble** – trains a meta-learner (logistic regression) on the
  base-detector scores as features.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class WeightedAverageEnsemble:
    """Combine detectors by weighted average of their AI-probability scores.

    Parameters
    ----------
    detectors:
        Dict mapping detector name → detector instance.  Each detector must
        implement a ``predict(texts: List[str]) -> List[dict]`` method.
    weights:
        Optional dict mapping detector name → float weight.  If not provided,
        uniform weights are used.  Weights need not sum to 1 (they are
        normalised internally).
    """

    def __init__(
        self,
        detectors: Dict[str, object],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.detectors = detectors
        names = list(detectors.keys())
        if weights is None:
            raw_weights = {n: 1.0 for n in names}
        else:
            raw_weights = weights
        total = sum(raw_weights.values())
        self.weights: Dict[str, float] = {n: w / total for n, w in raw_weights.items()}

    def predict(self, texts: List[str]) -> List[dict]:
        """Run all detectors and return weighted-average predictions."""
        n = len(texts)
        combined_scores = np.zeros(n)

        for name, detector in self.detectors.items():
            preds = detector.predict(texts)  # type: ignore[attr-defined]
            scores = np.array([p["score"] for p in preds])
            combined_scores += self.weights[name] * scores

        return [
            {"score": float(s), "label": int(s >= 0.5)}
            for s in combined_scores
        ]


class StackingEnsemble:
    """Meta-learner trained on base-detector scores.

    After calling :meth:`fit`, the stacking ensemble uses a logistic
    regression to produce final predictions from the base-detector scores.

    Parameters
    ----------
    detectors:
        Dict mapping detector name → detector instance.
    """

    def __init__(self, detectors: Dict[str, object]) -> None:
        self.detectors = detectors
        self._meta = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        self._scaler = StandardScaler()
        self._fitted = False

    def _base_scores(self, texts: List[str]) -> np.ndarray:
        """Return (N, K) matrix of base-detector scores."""
        cols = []
        for detector in self.detectors.values():
            preds = detector.predict(texts)  # type: ignore[attr-defined]
            cols.append([p["score"] for p in preds])
        return np.column_stack(cols)  # (N, K)

    def fit(self, texts: List[str], labels: List[int]) -> None:
        """Train the meta-learner on *texts* with ground-truth *labels*."""
        X = self._base_scores(texts)
        X_scaled = self._scaler.fit_transform(X)
        self._meta.fit(X_scaled, labels)
        self._fitted = True

    def predict(self, texts: List[str]) -> List[dict]:
        """Return stacked predictions."""
        if not self._fitted:
            raise RuntimeError("StackingEnsemble must be fit() before predict().")
        X = self._base_scores(texts)
        X_scaled = self._scaler.transform(X)
        probs = self._meta.predict_proba(X_scaled)
        return [
            {"score": float(p[1]), "label": int(p[1] >= 0.5)}
            for p in probs
        ]
