# PAN 2026 – Generated Plagiarism Detection

A complete system for the [PAN 2026 shared task on Generated Plagiarism Detection](https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html).

Given a text document (optionally paired with a source document), the system predicts whether it was **written by a human** (label `0`) or **AI-generated / AI-rewritten** (label `1`).

---

## Repository structure

```
.
├── data/
│   ├── raw/          ← place downloaded PAN corpora here
│   └── processed/    ← pre-processed / cached features
├── notebooks/
│   └── 01_exploratory.ipynb
├── results/          ← model checkpoints and prediction outputs
├── src/
│   ├── data_loader.py           ← load / save JSONL corpora
│   ├── evaluate.py              ← metrics (F1-macro, AUC-ROC, …)
│   ├── features/
│   │   ├── perplexity.py        ← GPT-2 perplexity & LLR features
│   │   ├── stylometric.py       ← TTR, sentence-length stats, etc.
│   │   └── embeddings.py        ← sentence-transformer features
│   └── models/
│       ├── zero_shot.py         ← PerplexityThreshold, LLR, RoBERTa detectors
│       ├── classifier.py        ← FineTunedClassifier (DeBERTa) + FeatureClassifier (GBM)
│       └── ensemble.py          ← WeightedAverage & Stacking ensembles
├── train.py          ← training CLI
├── predict.py        ← inference CLI
├── Dockerfile        ← TIRA / PAN submission container
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the PAN 2026 corpus

Register at https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html,
download the corpus, and place it in `data/raw/`:

```
data/raw/train.jsonl
data/raw/dev.jsonl
data/raw/test.jsonl   ← labels omitted
```

Expected JSONL schema (one JSON object per line):

```jsonc
{"id": "doc001", "text": "...", "label": 1}
{"id": "doc002", "text": "...", "source_text": "...", "label": 0}
```

### 3. Explore the data

```bash
jupyter notebook notebooks/01_exploratory.ipynb
```

### 4. Train a model

**Zero-shot baseline** (no training data required):
```bash
python train.py --train data/raw/train.jsonl --mode zeroshot \
    --output results/zeroshot_config.json
```

**Feature-based classifier** (fast, CPU-friendly):
```bash
python train.py --train data/raw/train.jsonl --dev data/raw/dev.jsonl \
    --mode features --output results/feature_clf.joblib
```

**Fine-tune DeBERTa** (requires GPU):
```bash
python train.py --train data/raw/train.jsonl --dev data/raw/dev.jsonl \
    --mode finetune --model microsoft/deberta-v3-base \
    --output results/deberta_model --epochs 3 --batch-size 8
```

### 5. Generate predictions

```bash
# Zero-shot
python predict.py --input data/raw/test.jsonl --mode zeroshot \
    --threshold 50.0 --output results/predictions.jsonl

# Feature classifier
python predict.py --input data/raw/test.jsonl --mode features \
    --model results/feature_clf.joblib --output results/predictions.jsonl

# Fine-tuned DeBERTa
python predict.py --input data/raw/test.jsonl --mode finetune \
    --model results/deberta_model --output results/predictions.jsonl

# RoBERTa zero-shot detector
python predict.py --input data/raw/test.jsonl --mode roberta \
    --output results/predictions.jsonl
```

Output format (JSONL):
```jsonc
{"id": "doc001", "score": 0.87, "label": 1}
{"id": "doc002", "score": 0.12, "label": 0}
```

---

## Detectors & models

| Mode | Class | Description |
|------|-------|-------------|
| `zeroshot` | `PerplexityThresholdDetector` | GPT-2 perplexity threshold |
| `zeroshot` | `LLRDetector` | Log-likelihood ratio (DetectGPT-style) |
| `roberta` | `RobertaDetector` | `openai-community/roberta-base-openai-detector` |
| `features` | `FeatureClassifier` | Gradient Boosting on stylometric + embedding features |
| `finetune` | `FineTunedClassifier` | DeBERTa-v3-base fine-tuned for sequence classification |
| ensemble | `WeightedAverageEnsemble` | Weighted combination of any detectors |
| ensemble | `StackingEnsemble` | Logistic-regression meta-learner on detector scores |

### Features

| Module | Features |
|--------|----------|
| `perplexity.py` | Perplexity, mean log-likelihood, LLR, burstiness, entropy |
| `stylometric.py` | TTR, CTTR, mean/std sentence & word length, punctuation density, hapax ratio, function-word ratio, sentence-length burstiness |
| `embeddings.py` | Document embedding (768-d), intra-document self-similarity, source–suspicious cosine similarity |

---

## Evaluation metrics

Primary metric: **macro-F1** (mirrors PAN official evaluation).  
Additional: accuracy, per-class F1, AUC-ROC, average precision.

```python
from src.evaluate import compute_metrics, print_report

metrics = compute_metrics(y_true, y_pred, y_score)
print_report(y_true, y_pred, y_score)
```

---

## Docker / TIRA submission

```bash
# Build the image
docker build -t pan26-generated-plagiarism .

# Run locally (zeroshot mode)
docker run --rm \
  -v $(pwd)/data/raw:/input \
  -v $(pwd)/results:/output \
  -e DETECTOR_MODE=zeroshot \
  -e PPL_THRESHOLD=50.0 \
  pan26-generated-plagiarism

# Run with a fine-tuned model
docker run --rm \
  -v $(pwd)/data/raw:/input \
  -v $(pwd)/results/deberta_model:/model \
  -v $(pwd)/results:/output \
  -e DETECTOR_MODE=finetune \
  -e MODEL_PATH=/model \
  pan26-generated-plagiarism
```

---

## Citation

If you use this code, please cite the PAN 2026 overview paper (to be published)
and the relevant shared-task description.
