"""
classifier.py
-------------
Supervised fine-tuned classifier for AI-generated text detection.

Two variants are provided:

* **FineTunedClassifier** – fine-tunes a DeBERTa-v3-base (or similar)
  sequence-classification model directly on raw texts.
* **FeatureClassifier** – trains a lightweight gradient-boosted classifier
  on the hand-crafted feature vectors produced by the feature-extraction
  modules (useful when compute is limited).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import joblib


# ---------------------------------------------------------------------------
# Torch dataset wrapper
# ---------------------------------------------------------------------------

class _TextDataset(Dataset):
    def __init__(self, encodings: dict, labels: Optional[List[int]] = None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Fine-tuned sequence classifier
# ---------------------------------------------------------------------------

class FineTunedClassifier:
    """Fine-tune a pre-trained transformer for binary AI-text classification.

    Parameters
    ----------
    model_name:
        HuggingFace model ID (default: DeBERTa-v3-base).
    max_length:
        Maximum token length.
    output_dir:
        Directory to save checkpoints and the final model.
    num_labels:
        Number of output classes (default 2: human / AI).
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512,
        output_dir: str = "results/classifier",
        num_labels: int = 2,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = output_dir
        self.num_labels = num_labels
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _encode(self, texts: List[str]) -> dict:
        tokenizer = self._get_tokenizer()
        return tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None,
        )

    @staticmethod
    def _compute_metrics(eval_pred) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
            "roc_auc": roc_auc_score(labels, probs[:, 1]),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_texts: List[str],
        train_labels: List[int],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[int]] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
    ) -> None:
        """Fine-tune the model on *train_texts* / *train_labels*."""
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )

        train_enc = self._encode(train_texts)
        train_dataset = _TextDataset(train_enc, train_labels)

        eval_dataset = None
        if eval_texts is not None and eval_labels is not None:
            eval_enc = self._encode(eval_texts)
            eval_dataset = _TextDataset(eval_enc, eval_labels)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=0.01,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch" if eval_dataset else "no",
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="f1_macro",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        callbacks = (
            [EarlyStoppingCallback(early_stopping_patience=2)]
            if eval_dataset is not None
            else []
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=callbacks,
        )
        trainer.train()

    def predict(self, texts: List[str]) -> List[dict]:
        """Return prediction dicts with keys ``score`` and ``label``."""
        if self._model is None:
            raise RuntimeError("Model is not trained. Call fit() first.")
        enc = self._encode(texts)
        dataset = _TextDataset(enc)
        trainer = Trainer(model=self._model)
        output = trainer.predict(dataset)
        probs = torch.softmax(torch.tensor(output.predictions), dim=-1).numpy()
        results = []
        for p in probs:
            score = float(p[1])
            results.append({"score": score, "label": int(score >= 0.5)})
        return results

    def save(self, path: str) -> None:
        """Save the fine-tuned model and tokenizer to *path*."""
        if self._model is None:
            raise RuntimeError("No model to save.")
        Path(path).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._get_tokenizer().save_pretrained(path)

    def load(self, path: str) -> None:
        """Load a previously saved model from *path*."""
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSequenceClassification.from_pretrained(path)


# ---------------------------------------------------------------------------
# Lightweight feature-based classifier
# ---------------------------------------------------------------------------

class FeatureClassifier:
    """Gradient-boosted classifier trained on hand-crafted feature vectors.

    Useful as a fast baseline that does not require a GPU.

    Parameters
    ----------
    n_estimators:
        Number of boosting rounds.
    max_depth:
        Maximum tree depth.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
    ) -> None:
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                random_state=42,
            )),
        ])
        self._feature_names: Optional[List[str]] = None

    def fit(self, feature_matrix: np.ndarray, labels: List[int], feature_names: Optional[List[str]] = None) -> None:
        """Train on a pre-computed feature matrix."""
        self._feature_names = feature_names
        self._pipeline.fit(feature_matrix, labels)

    def predict(self, feature_matrix: np.ndarray) -> List[dict]:
        """Return prediction dicts with keys ``score`` and ``label``."""
        probs = self._pipeline.predict_proba(feature_matrix)
        return [
            {"score": float(p[1]), "label": int(p[1] >= 0.5)}
            for p in probs
        ]

    def save(self, path: str) -> None:
        """Persist the trained pipeline to *path* (joblib format)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    def load(self, path: str) -> None:
        """Load a pipeline from *path*."""
        self._pipeline = joblib.load(path)
