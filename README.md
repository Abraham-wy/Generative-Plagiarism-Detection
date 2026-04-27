# PAN 2026 – 生成式剽窃检测

> 本项目是 [PAN 2026 生成式剽窃检测共享任务](https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html) 的完整系统实现，支持多种检测策略，从零样本基线到深度微调模型，一站式解决"这段文字是人写的还是 AI 生成/改写的？"这一核心问题。

---

## 目录

1. [项目简介](#项目简介)
2. [核心原理](#核心原理)
3. [环境要求](#环境要求)
4. [项目结构](#项目结构)
5. [快速开始（Step-by-Step 教程）](#快速开始step-by-step-教程)
   - [Step 1：克隆项目并安装依赖](#step-1克隆项目并安装依赖)
   - [Step 2：准备数据](#step-2准备数据)
   - [Step 3：数据探索（可选）](#step-3数据探索可选)
   - [Step 4：训练模型](#step-4训练模型)
   - [Step 5：生成预测](#step-5生成预测)
   - [Step 6：评估结果](#step-6评估结果)
6. [检测器与模型详解](#检测器与模型详解)
7. [特征工程详解](#特征工程详解)
8. [集成策略](#集成策略)
9. [Docker / TIRA 提交](#docker--tira-提交)
10. [常见问题 FAQ](#常见问题-faq)
11. [引用](#引用)

---

## 项目简介

**生成式剽窃检测**（Generated Plagiarism Detection）是学术诚信领域的新兴挑战：随着 ChatGPT、GPT-4、Claude 等大型语言模型的普及，学生和作者越来越容易使用 AI 生成或改写文本来冒充自己的原创内容。

本系统的任务定义：

> 给定一篇可疑文本（可选地附带原始参考文档），判断该文本是**人类撰写**（标签 `0`）还是**由 AI 生成 / AI 改写**（标签 `1`）。

本系统实现了从轻量级零样本方法到重量级微调 Transformer 的完整检测流水线，适合研究人员快速复现、参赛，以及在实际场景中部署。

---

## 核心原理

系统采用**多层次检测策略**，每一层都基于不同的理论假设：

### 1. 困惑度检测（Perplexity-based）

**原理**：AI 生成的文本通常对语言模型来说"困惑度更低"——因为生成模型倾向于选择高概率词汇，导致文本更"流畅"、更"可预测"。  
**做法**：用 GPT-2 计算待检测文本的困惑度（Perplexity），低于阈值则判定为 AI 生成。

### 2. 对数似然比检测（LLR / DetectGPT 风格）

**原理**：对文本做小幅扰动（改写），如果原文是 AI 生成的，其对数似然通常高于扰动后的版本（局部极大值特性）。  
**做法**：计算原文与扰动文本的对数似然之差（LLR），正值倾向于 AI 生成。

### 3. 文体特征分类（Stylometric Features）

**原理**：AI 生成文本与人类写作在词汇丰富度、句子长度分布、标点使用等文体特征上存在统计差异。  
**做法**：提取 TTR、句长方差、标点密度等特征，训练梯度提升分类器。

### 4. 语义嵌入特征（Embedding Features）

**原理**：利用句子 Transformer 获取语义表示，AI 生成文本内部句子的语义相似度通常较高（缺乏人类写作的多样性）。  
**做法**：计算文档嵌入向量、句内自相似度，以及原文与可疑文的余弦相似度。

### 5. 微调 Transformer 分类器（Fine-tuned DeBERTa）

**原理**：直接在标注数据上微调预训练语言模型，让模型学习区分人类写作与 AI 写作的深层语言模式。  
**做法**：在 PAN 数据集上微调 DeBERTa-v3-base 进行序列分类。

### 6. 集成（Ensemble）

将多个检测器的得分通过加权平均或 Stacking 方式融合，进一步提升鲁棒性。

---

## 环境要求

| 依赖 | 版本要求 |
|------|---------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1.0 |
| CUDA（可选，微调模式需要） | ≥ 11.8 |
| 内存（RAM） | ≥ 16 GB（推荐 32 GB） |
| 显存（微调模式） | ≥ 16 GB |
| 磁盘 | ≥ 20 GB（模型权重 + 数据） |

---

## 项目结构

```
.
├── data/
│   ├── raw/          ← 存放下载的 PAN 语料库（train/dev/test.jsonl）
│   └── processed/    ← 预处理 / 缓存的特征（自动生成）
├── notebooks/
│   └── 01_exploratory.ipynb   ← 数据探索笔记本
├── results/          ← 训练输出：模型检查点与预测文件
├── src/
│   ├── __init__.py
│   ├── data_loader.py           ← 加载 / 保存 JSONL 语料库
│   ├── evaluate.py              ← 评估指标（F1-macro、AUC-ROC 等）
│   ├── features/
│   │   ├── __init__.py
│   │   ├── perplexity.py        ← GPT-2 困惑度与 LLR 特征
│   │   ├── stylometric.py       ← TTR、句子长度统计等文体特征
│   │   └── embeddings.py        ← 句子 Transformer 嵌入特征
│   └── models/
│       ├── __init__.py
│       ├── zero_shot.py         ← 困惑度阈值、LLR、RoBERTa 检测器
│       ├── classifier.py        ← FineTunedClassifier（DeBERTa）+ FeatureClassifier（GBM）
│       └── ensemble.py          ← 加权平均集成与 Stacking 集成
├── train.py          ← 训练命令行入口
├── predict.py        ← 推理命令行入口
├── Dockerfile        ← TIRA / PAN 提交容器
└── requirements.txt  ← Python 依赖列表
```

---

## 快速开始（Step-by-Step 教程）

### Step 1：克隆项目并安装依赖

```bash
# 1. 克隆仓库
git clone https://github.com/Abraham-wy/Generative-Plagiarism-Detection.git
cd Generative-Plagiarism-Detection

# 2. 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. 安装所有依赖
pip install -r requirements.txt

# 4. 下载 NLTK 数据（首次运行需要）
python -c "import nltk; [nltk.download(p) for p in ('punkt', 'punkt_tab', 'stopwords')]"
```

> **提示**：如果网络较慢，可以使用国内镜像加速安装：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

### Step 2：准备数据

#### 2.1 下载 PAN 2026 官方数据集

前往 [PAN 2026 官网](https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html) 注册账号并下载数据集，将文件放入 `data/raw/` 目录：

```
data/raw/
├── train.jsonl    ← 训练集（含标签）
├── dev.jsonl      ← 验证集（含标签）
└── test.jsonl     ← 测试集（不含标签）
```

#### 2.2 数据格式说明

每个 `.jsonl` 文件中每一行是一个 JSON 对象：

```jsonc
// 最简形式（仅含待检测文本）
{"id": "doc001", "text": "这是一段待检测的文字...", "label": 1}

// 完整形式（含原始参考文档，用于计算文本相似度）
{"id": "doc002", "text": "可疑文本内容...", "source_text": "原始参考文档...", "label": 0}
```

字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 文档唯一标识符 |
| `text` | string | 待检测的可疑文本 |
| `source_text` | string（可选） | 原始参考文档，用于相似度计算 |
| `label` | int | `0` = 人类撰写，`1` = AI 生成/改写 |

#### 2.3 使用自定义数据集

如果你想在自己的数据上运行，只需将数据整理成上述 JSONL 格式即可。例如，用 Python 生成示例数据：

```python
import json

samples = [
    {"id": "s001", "text": "人工智能是计算机科学的一个分支...", "label": 0},
    {"id": "s002", "text": "Artificial intelligence (AI) refers to...", "label": 1},
]

with open("data/raw/my_data.jsonl", "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
```

---

### Step 3：数据探索（可选）

启动 Jupyter Notebook 进行数据探索和可视化：

```bash
jupyter notebook notebooks/01_exploratory.ipynb
```

---

### Step 4：训练模型

系统提供三种训练模式，按资源需求从低到高排列：

#### 模式 A：零样本基线（无需 GPU，无需训练数据）

零样本模式不需要训练——它通过 GPT-2 困惑度对文本打分，并在训练集上寻找最优阈值：

```bash
python train.py \
    --train data/raw/train.jsonl \
    --mode  zeroshot \
    --output results/zeroshot_config.json
```

输出文件 `results/zeroshot_config.json` 示例：
```json
{
  "threshold": 47.3,
  "f1_macro": 0.782
}
```

#### 模式 B：特征分类器（推荐入门，CPU 友好）

基于文体特征 + 嵌入特征训练梯度提升（GBM）分类器，速度快、内存占用小：

```bash
python train.py \
    --train      data/raw/train.jsonl \
    --dev        data/raw/dev.jsonl \
    --mode       features \
    --output     results/feature_clf.joblib \
    --n-estimators 300 \
    --max-depth    4
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n-estimators` | 300 | GBM 树的数量，越大越慢但通常越准 |
| `--max-depth` | 4 | 每棵树的最大深度 |

#### 模式 C：微调 DeBERTa（效果最佳，需要 GPU）

在 PAN 数据集上微调预训练 DeBERTa-v3-base 模型：

```bash
python train.py \
    --train        data/raw/train.jsonl \
    --dev          data/raw/dev.jsonl \
    --mode         finetune \
    --model        microsoft/deberta-v3-base \
    --output       results/deberta_model \
    --epochs       3 \
    --batch-size   8 \
    --learning-rate 2e-5
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `microsoft/deberta-v3-base` | HuggingFace 模型 ID，首次运行会自动下载 |
| `--epochs` | 3 | 训练轮数，建议 3-5 轮 |
| `--batch-size` | 8 | 每批次样本数，显存不足时调小（如 4） |
| `--learning-rate` | 2e-5 | 学习率，通常无需修改 |

> **显存不足时的解决方案**：
> - 将 `--batch-size` 从 8 调整为 4 或 2
> - 使用更小的模型，如 `microsoft/deberta-v3-small`
> - 开启梯度检查点（gradient checkpointing）

---

### Step 5：生成预测

训练完成后，使用 `predict.py` 对测试集进行推理：

```bash
# 零样本模式（使用训练时确定的阈值）
python predict.py \
    --input     data/raw/test.jsonl \
    --mode      zeroshot \
    --threshold 47.3 \
    --output    results/predictions_zeroshot.jsonl

# 特征分类器
python predict.py \
    --input  data/raw/test.jsonl \
    --mode   features \
    --model  results/feature_clf.joblib \
    --output results/predictions_features.jsonl

# 微调 DeBERTa
python predict.py \
    --input  data/raw/test.jsonl \
    --mode   finetune \
    --model  results/deberta_model \
    --output results/predictions_deberta.jsonl

# RoBERTa 零样本检测器（无需训练，直接调用预训练检测模型）
python predict.py \
    --input  data/raw/test.jsonl \
    --mode   roberta \
    --output results/predictions_roberta.jsonl
```

**输出文件格式**（每行一个 JSON）：

```jsonc
{"id": "doc001", "score": 0.87, "label": 1}   // score 越高，越可能是 AI 生成
{"id": "doc002", "score": 0.12, "label": 0}   // score 越低，越可能是人类撰写
```

---

### Step 6：评估结果

在有标签的验证集上评估模型性能：

```python
from src.data_loader import load_jsonl
from src.evaluate import compute_metrics, print_report

# 加载真实标签
dev_records = load_jsonl("data/raw/dev.jsonl")
y_true = [r["label"] for r in dev_records]

# 加载预测结果
pred_records = load_jsonl("results/predictions_features.jsonl")
y_pred  = [r["label"] for r in pred_records]
y_score = [r["score"] for r in pred_records]

# 打印完整评估报告
print_report(y_true, y_pred, y_score)
```

输出示例：

```
=== Dev-set Evaluation (FeatureClassifier) ===
Accuracy : 0.8920
F1-macro : 0.8875
AUC-ROC  : 0.9412
Avg Prec : 0.9301

              precision    recall  f1-score   support
   Human (0)     0.91      0.88      0.89       520
   AI    (1)     0.87      0.90      0.88       480
```

主要指标说明：

| 指标 | 说明 |
|------|------|
| **F1-macro** | 宏平均 F1，PAN 官方主指标，对类别不平衡鲁棒 |
| Accuracy | 准确率 |
| AUC-ROC | ROC 曲线下面积，衡量排序能力 |
| Avg Precision | 平均精度（PR 曲线下面积） |

---

## 检测器与模型详解

| 模式 | 类名 | 所在文件 | 说明 |
|------|------|---------|------|
| `zeroshot` | `PerplexityThresholdDetector` | `src/models/zero_shot.py` | 用 GPT-2 计算困惑度，低于阈值则判为 AI 生成 |
| `zeroshot` | `LLRDetector` | `src/models/zero_shot.py` | 对数似然比检测器，DetectGPT 风格 |
| `roberta` | `RobertaDetector` | `src/models/zero_shot.py` | 调用 `openai-community/roberta-base-openai-detector`，无需训练 |
| `features` | `FeatureClassifier` | `src/models/classifier.py` | 文体特征 + 嵌入特征 → 梯度提升分类器（GBM） |
| `finetune` | `FineTunedClassifier` | `src/models/classifier.py` | 在标注数据上微调 DeBERTa-v3-base |
| ensemble | `WeightedAverageEnsemble` | `src/models/ensemble.py` | 多检测器输出的加权平均融合 |
| ensemble | `StackingEnsemble` | `src/models/ensemble.py` | 以逻辑回归为元学习器的 Stacking 集成 |

---

## 特征工程详解

### `perplexity.py` — 困惑度特征

| 特征名 | 说明 |
|--------|------|
| `perplexity` | GPT-2 计算的文本困惑度，AI 生成文本通常更低 |
| `mean_log_likelihood` | 每个 token 的平均对数似然 |
| `llr` | 对数似然比（LLR），用于 DetectGPT 风格检测 |
| `burstiness` | 困惑度在句子间的突发性/方差，人类写作方差更大 |
| `entropy` | token 概率分布的熵 |

### `stylometric.py` — 文体特征

| 特征名 | 说明 |
|--------|------|
| `ttr` | 词汇类型-词符比（Type-Token Ratio），衡量词汇丰富度 |
| `cttr` | 修正后的 TTR |
| `mean_sent_len` | 句子平均长度（词数） |
| `std_sent_len` | 句子长度标准差，AI 生成文本通常更均匀 |
| `mean_word_len` | 词汇平均长度（字符数） |
| `punct_density` | 标点符号密度 |
| `hapax_ratio` | 只出现一次的词的比率，人类写作通常更高 |
| `function_word_ratio` | 功能词比率 |
| `sent_len_burstiness` | 句子长度的突发性 |

### `embeddings.py` — 嵌入特征

| 特征名 | 说明 |
|--------|------|
| 文档嵌入向量（768 维） | `all-mpnet-base-v2` 模型的句子级嵌入 |
| `self_sim` | 文档内句子之间的平均余弦相似度，AI 文本通常更高 |
| `source_sim` | 可疑文本与原始参考文档的余弦相似度（仅在有 source_text 时有效） |

---

## 集成策略

集成方法位于 `src/models/ensemble.py`，可将多个检测器的输出融合以提升性能：

```python
from src.models.ensemble import WeightedAverageEnsemble, StackingEnsemble

# 加权平均集成（无需训练）
ensemble = WeightedAverageEnsemble(
    detectors=[detector1, detector2, detector3],
    weights=[0.3, 0.3, 0.4]   # 权重之和为 1
)
preds = ensemble.predict(texts)

# Stacking 集成（需要在验证集上训练元学习器）
stacking = StackingEnsemble(
    detectors=[detector1, detector2, detector3]
)
stacking.fit(val_texts, val_labels)
preds = stacking.predict(test_texts)
```

---

## Docker / TIRA 提交

如果你要将系统提交到 PAN 2026 官方评估平台（TIRA），请使用 Docker：

```bash
# 1. 构建 Docker 镜像
docker build -t pan26-generated-plagiarism .

# 2. 本地测试（零样本模式）
docker run --rm \
  -v $(pwd)/data/raw:/input \
  -v $(pwd)/results:/output \
  -e DETECTOR_MODE=zeroshot \
  -e PPL_THRESHOLD=50.0 \
  pan26-generated-plagiarism

# 3. 使用预训练特征分类器
docker run --rm \
  -v $(pwd)/data/raw:/input \
  -v $(pwd)/results:/output \
  -v $(pwd)/results/feature_clf.joblib:/model/clf.joblib \
  -e DETECTOR_MODE=features \
  -e MODEL_PATH=/model/clf.joblib \
  pan26-generated-plagiarism

# 4. 使用微调 DeBERTa 模型
docker run --rm \
  -v $(pwd)/data/raw:/input \
  -v $(pwd)/results/deberta_model:/model \
  -v $(pwd)/results:/output \
  -e DETECTOR_MODE=finetune \
  -e MODEL_PATH=/model \
  pan26-generated-plagiarism
```

**Docker 环境变量说明**：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `DETECTOR_MODE` | `zeroshot` | 检测模式：`zeroshot` / `features` / `finetune` / `roberta` |
| `MODEL_PATH` | `""` | 模型文件路径（features/finetune 模式必填） |
| `PPL_THRESHOLD` | `50.0` | 困惑度阈值（zeroshot 模式使用） |
| `INPUT_FILE` | `/input/test.jsonl` | 输入文件路径 |
| `OUTPUT_FILE` | `/output/predictions.jsonl` | 输出文件路径 |

---

## 常见问题 FAQ

**Q1：运行时提示 `CUDA out of memory`，怎么办？**

> 减小 `--batch-size`（例如从 8 改为 2），或换用更小的基础模型（如 `microsoft/deberta-v3-small`）。

**Q2：零样本模式需要哪些模型权重？**

> 零样本模式会自动从 HuggingFace 下载 `gpt2` 模型权重（约 500 MB）。若网络受限，可提前手动下载并设置 `TRANSFORMERS_CACHE` 环境变量。

**Q3：可以检测中文文本吗？**

> 目前困惑度特征基于 GPT-2（英文），对中文效果有限。建议中文场景改用 `roberta` 模式或将基础模型替换为支持中文的预训练模型（如 `hfl/chinese-roberta-wwm-ext`）。

**Q4：没有 GPU，能不能运行微调模式？**

> 可以，但速度会非常慢（CPU 微调 DeBERTa 通常需要数小时）。推荐使用 `--mode features` 或 `--mode roberta` 进行无 GPU 部署。

**Q5：如何复现最佳性能？**

> 推荐使用集成策略：先分别训练 `features` 和 `finetune` 模型，再用 `WeightedAverageEnsemble` 融合两者的预测得分。

**Q6：输出的 `score` 是什么含义？**

> `score` 是模型预测为 AI 生成（标签 `1`）的置信度分数，范围 [0, 1]。`score > 0.5` 时 `label` 为 `1`（AI 生成），否则为 `0`（人类撰写）。

---

## 引用

如果你在研究中使用了本代码，请引用 PAN 2026 任务综述论文（待发表）及相关共享任务描述：

```bibtex
@inproceedings{pan26-generated-plagiarism,
  title     = {{PAN} 2026: Generated Plagiarism Detection},
  booktitle = {Working Notes of {CLEF} 2026},
  year      = {2026},
  note      = {待发表 / to appear}
}
```
