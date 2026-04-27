# PAN 2026 – 生成式剽窃检测

本项目是 [PAN 2026 生成式剽窃检测共享任务](https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html) 的完整系统实现。

给定一篇文本（可选地与原始文档配对），系统将预测该文本是**人类撰写**（标签 `0`）还是**由 AI 生成 / AI 改写**（标签 `1`）。

---

## 项目结构

```
.
├── data/
│   ├── raw/          ← 存放下载的 PAN 语料库
│   └── processed/    ← 预处理 / 缓存的特征
├── notebooks/
│   └── 01_exploratory.ipynb
├── results/          ← 模型检查点与预测输出
├── src/
│   ├── data_loader.py           ← 加载 / 保存 JSONL 语料库
│   ├── evaluate.py              ← 评估指标（F1-macro、AUC-ROC 等）
│   ├── features/
│   │   ├── perplexity.py        ← GPT-2 困惑度与 LLR 特征
│   │   ├── stylometric.py       ← TTR、句子长度统计等文体特征
│   │   └── embeddings.py        ← 句子 Transformer 嵌入特征
│   └── models/
│       ├── zero_shot.py         ← 困惑度阈值、LLR、RoBERTa 检测器
│       ├── classifier.py        ← FineTunedClassifier（DeBERTa）+ FeatureClassifier（GBM）
│       └── ensemble.py          ← 加权平均集成与 Stacking 集成
├── train.py          ← 训练命令行入口
├── predict.py        ← 推理命令行入口
├── Dockerfile        ← TIRA / PAN 提交容器
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 下载 PAN 2026 语料库

请在 https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html 注册并下载语料库，
将其放入 `data/raw/`：

```
data/raw/train.jsonl
data/raw/dev.jsonl
data/raw/test.jsonl   ← 测试集不含标签
```

期望的 JSONL 格式（每行一个 JSON 对象）：

```jsonc
{"id": "doc001", "text": "...", "label": 1}
{"id": "doc002", "text": "...", "source_text": "...", "label": 0}
```

### 3. 数据探索

```bash
jupyter notebook notebooks/01_exploratory.ipynb
```

### 4. 训练模型

**零样本基线**（无需训练数据）：
```bash
python train.py --train data/raw/train.jsonl --mode zeroshot \
    --output results/zeroshot_config.json
```

**特征分类器**（速度快，适合 CPU）：
```bash
python train.py --train data/raw/train.jsonl --dev data/raw/dev.jsonl \
    --mode features --output results/feature_clf.joblib
```

**微调 DeBERTa**（需要 GPU）：
```bash
python train.py --train data/raw/train.jsonl --dev data/raw/dev.jsonl \
    --mode finetune --model microsoft/deberta-v3-base \
    --output results/deberta_model --epochs 3 --batch-size 8
```

### 5. 生成预测

```bash
# 零样本模式
python predict.py --input data/raw/test.jsonl --mode zeroshot \
    --threshold 50.0 --output results/predictions.jsonl

# 特征分类器
python predict.py --input data/raw/test.jsonl --mode features \
    --model results/feature_clf.joblib --output results/predictions.jsonl

# 微调后的 DeBERTa
python predict.py --input data/raw/test.jsonl --mode finetune \
    --model results/deberta_model --output results/predictions.jsonl

# RoBERTa 零样本检测器
python predict.py --input data/raw/test.jsonl --mode roberta \
    --output results/predictions.jsonl
```

输出格式（JSONL）：
```jsonc
{"id": "doc001", "score": 0.87, "label": 1}
{"id": "doc002", "score": 0.12, "label": 0}
```

---

## 检测器与模型

| 模式 | 类名 | 说明 |
|------|------|------|
| `zeroshot` | `PerplexityThresholdDetector` | GPT-2 困惑度阈值检测器 |
| `zeroshot` | `LLRDetector` | 对数似然比检测器（DetectGPT 风格）|
| `roberta` | `RobertaDetector` | `openai-community/roberta-base-openai-detector` |
| `features` | `FeatureClassifier` | 基于文体特征 + 嵌入特征的梯度提升分类器 |
| `finetune` | `FineTunedClassifier` | 微调 DeBERTa-v3-base 序列分类模型 |
| ensemble | `WeightedAverageEnsemble` | 多检测器加权平均集成 |
| ensemble | `StackingEnsemble` | 以逻辑回归为元学习器的 Stacking 集成 |

### 特征说明

| 模块 | 特征 |
|------|------|
| `perplexity.py` | 困惑度、平均对数似然、LLR、突发性、熵 |
| `stylometric.py` | TTR、CTTR、句子 / 词语长度均值与标准差、标点密度、Hapax 比率、功能词比率、句长突发性 |
| `embeddings.py` | 文档嵌入向量（768 维）、文档内句子自相似度、原文-可疑文余弦相似度 |

---

## 评估指标

主要指标：**宏平均 F1**（与 PAN 官方评估一致）。  
附加指标：准确率、各类别 F1、AUC-ROC、平均精度。

```python
from src.evaluate import compute_metrics, print_report

metrics = compute_metrics(y_true, y_pred, y_score)
print_report(y_true, y_pred, y_score)
```

---

## Docker / TIRA 提交

```bash
# 构建镜像
docker build -t pan26-generated-plagiarism .

# 本地运行（零样本模式）
docker run --rm \
  -v $(pwd)/data/raw:/input \
  -v $(pwd)/results:/output \
  -e DETECTOR_MODE=zeroshot \
  -e PPL_THRESHOLD=50.0 \
  pan26-generated-plagiarism

# 使用微调模型运行
docker run --rm \
  -v $(pwd)/data/raw:/input \
  -v $(pwd)/results/deberta_model:/model \
  -v $(pwd)/results:/output \
  -e DETECTOR_MODE=finetune \
  -e MODEL_PATH=/model \
  pan26-generated-plagiarism
```

---

## 引用

如果您使用了本代码，请引用 PAN 2026 任务综述论文（待发表）及相关共享任务描述。
