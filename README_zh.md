# PAN 生成式剽窃检测 —— 中文项目说明

> 英文版说明请参见 [README.md](README.md)

本仓库是针对 **PAN 2026 生成式剽窃检测（Generated Plagiarism Detection）**任务的完整解决方案框架，涵盖文本对齐检测（Text Alignment）和文档检索（Retrieval）两个子任务。本文档将解释仓库中每个文件的用途及其相互关系。

---

## 目录

- [任务背景](#任务背景)
- [仓库结构总览](#仓库结构总览)
- [根目录文件说明](#根目录文件说明)
- [src/ 源代码包说明](#src-源代码包说明)
- [docs/ 文档目录说明](#docs-文档目录说明)
- [其他目录说明](#其他目录说明)
- [各组件关系图](#各组件关系图)
- [快速上手](#快速上手)

---

## 任务背景

**PAN 生成式剽窃检测**要求系统判断：给定一对文档（可疑文档 + 来源文档），可疑文档中哪些段落是由 LLM（大语言模型）基于来源文档自动生成（改写/复述）的。

任务分为两个子任务：

| 子任务 | 描述 | 关键脚本 |
|--------|------|----------|
| **文本对齐（Text Alignment）** | 给定已配对的文档，找出可疑段落与来源段落的具体字符偏移量 | `main.py` |
| **文档检索（Retrieval）** | 给定可疑文档，从大型语料库中检索最可能的来源文档（top-1000） | `retrieve.py` |

---

## 仓库结构总览

```text
Generative-Plagiarism-Detection/
│
├── main.py                          # 【核心】文本对齐基线系统（离线运行，Docker 入口）
├── train.py                         # 分类器训练入口
├── predict.py                       # 分类器推理入口
├── retrieve.py                      # 文档检索系统（BM25 + 密集重排序）
│
├── requirements.txt                 # main.py 依赖（最精简版，适合 Docker 部署）
├── requirements.retrieval-lite.txt  # retrieve.py 依赖（检索子任务所需额外包）
│
├── Dockerfile                       # main.py 的 Docker 容器配置
├── Dockerfile.lite                  # 轻量版 Docker 容器配置
├── .dockerignore                    # Docker 构建时忽略的文件
├── .gitignore                       # Git 忽略文件
│
├── src/                             # 【核心源代码包】
│   ├── __init__.py
│   ├── data_loader.py               # 数据集加载与保存工具
│   ├── evaluate.py                  # 评估指标计算工具
│   ├── features/                    # 特征提取模块
│   │   ├── __init__.py
│   │   ├── embeddings.py            # 句子嵌入特征
│   │   ├── perplexity.py            # 困惑度特征（GPT-2 基础）
│   │   └── stylometric.py          # 文体统计特征
│   └── models/                      # 分类模型模块
│       ├── __init__.py
│       ├── classifier.py            # 监督分类器（DeBERTa 微调 + 梯度提升）
│       ├── ensemble.py              # 集成检测器
│       └── zero_shot.py             # 零样本检测器
│
├── docs/                            # 文档目录
│   ├── BEGINNER_PROJECT_GUIDE.md    # 零基础入门指南（中文）
│   └── PAN26_TASK_CHECKLIST.md     # 任务核查清单
│
├── notebooks/
│   └── 01_exploratory.ipynb         # 探索性数据分析笔记本
│
├── data/
│   └── raw/                         # 原始数据集（.jsonl 格式，需自行下载）
│
├── results/                         # 训练结果与模型输出（运行后自动生成）
│
├── test-data/                       # 本地测试用小型数据集
├── pan26-spot-check-dataset/        # PAN 官方 spot-check 数据集
├── pan26-spot-check-output/         # spot-check 运行输出
├── pan26-official-baseline-output/  # PAN 官方基线的参考输出
└── official-baseline-pyterrier/     # PAN 官方 PyTerrier 基线代码（参考用）
```

---

## 根目录文件说明

### `main.py` —— 文本对齐基线系统（最重要！）

**这是 Docker 容器的入口点，也是提交给 TIRA 平台的核心程序。**

功能：给定已配对的文档，用句子嵌入找出可疑文档与来源文档中的匹配片段，输出标准 PAN XML 格式的检测结果。

**运行方式：**
```bash
python main.py /path/to/dataset /path/to/output
```

**工作流程：**
1. 读取 `pairs` 文件，获取所有文档对
2. 将每篇文档拆分为带字符偏移量的"句子窗口"
3. 用 `all-MiniLM-L6-v2` 句子 Transformer 对窗口编码
4. 计算可疑窗口与来源窗口之间的余弦相似度
5. 相似度超过阈值（默认 0.80）的窗口对记为"剽窃检测"
6. 合并相邻检测结果，输出 PAN 格式 XML 文件

**特点：** 完全离线运行，无需 PyTerrier、Java 或外部 API。

---

### `train.py` —— 分类器训练入口

用于训练 AI 文本检测分类器（针对判断单篇文档是否为 AI 生成的二分类任务）。

支持三种训练模式（通过 `--mode` 参数选择）：

| 模式 | 描述 | 资源需求 |
|------|------|----------|
| `features` | 提取文体特征 + 嵌入特征，训练梯度提升分类器 | CPU 即可 |
| `finetune` | 微调 DeBERTa-v3-base 进行序列分类 | 需要 GPU |
| `zeroshot` | 基于困惑度阈值，无需训练，只寻找最优阈值 | CPU 即可 |

**示例：**
```bash
# 训练特征分类器
python train.py --train data/raw/train.jsonl --dev data/raw/dev.jsonl --mode features --output results/feature_clf.joblib

# 微调 DeBERTa
python train.py --train data/raw/train.jsonl --dev data/raw/dev.jsonl --mode finetune --output results/deberta_model
```

---

### `predict.py` —— 分类器推理入口

使用已训练好的模型对测试集进行预测。支持与 `train.py` 相同的四种模式（`features`、`finetune`、`zeroshot`、`roberta`）。

**示例：**
```bash
# 使用特征分类器预测
python predict.py --input data/raw/test.jsonl --model results/feature_clf.joblib --mode features --output results/predictions.jsonl

# 使用零样本困惑度检测
python predict.py --input data/raw/test.jsonl --mode zeroshot --output results/predictions.jsonl
```

输出格式为 JSONL，每行包含 `id`、`score`（AI 生成概率）、`label`（0=人类，1=AI）。

---

### `retrieve.py` —— 文档检索系统

针对检索子任务：给定可疑文档（查询），在大型语料库中找出最可能的来源文档。

**三级检索流水线：**
1. **多子查询 BM25**：将可疑文档拆分为多个子查询片段，分别用 BM25 检索
2. **RRF 融合**：用 Reciprocal Rank Fusion 合并多组检索结果，得到候选集（默认 200 篇）
3. **密集向量重排序**：用 `all-MiniLM-L6-v2` 对候选集重排，最终保留 top-1000

输出：TREC run 格式（gzip 压缩），可直接提交评测。

**示例：**
```bash
python retrieve.py \
    --dataset /path/to/data/dir \
    --output output/ \
    --index /tmp/indexes
```

---

### `requirements.txt` —— `main.py` 的最小依赖

```
torch==2.2.2
sentence-transformers==2.7.0
numpy>=1.26.0,<2.0
```

专为 Docker 部署设计，保持镜像体积最小。

---

### `requirements.retrieval-lite.txt` —— 检索子任务的额外依赖

`retrieve.py`、`train.py`、`predict.py` 所需的额外包，包括：
- `pandas`、`nltk`、`jsonlines` —— 数据处理
- `tira`、`python-terrier`、`ir-datasets` —— TIRA 平台和 PyTerrier 检索框架
- `click` —— CLI 工具

---

### `Dockerfile` —— `main.py` 的 Docker 容器

基于 `python:3.10-slim`，构建时自动下载 `all-MiniLM-L6-v2` 模型并存储到 `/models/`，使容器可**完全离线运行**。

```bash
# 构建镜像
docker build -t pan-plagiarism-baseline .

# 运行
docker run --rm \
  -v /path/to/dataset:/input:ro \
  -v /path/to/output:/output \
  pan-plagiarism-baseline
```

---

### `Dockerfile.lite` —— 轻量版 Docker 容器

功能与 `Dockerfile` 类似，去掉了旧版检索工具等额外依赖，镜像更小。

---

## `src/` 源代码包说明

`src/` 包为 `train.py` 和 `predict.py` 提供所有核心逻辑。

### `src/data_loader.py` —— 数据集加载与保存

提供统一的数据 I/O 接口，支持多种格式：

| 函数 | 说明 |
|------|------|
| `load_jsonl(path)` | 读取 JSONL 文件，返回规范化记录列表 |
| `load_directory(text_dir, truth_path)` | 读取纯文本目录 + 可选标签文件 |
| `to_dataframe(records)` | 转换为 pandas DataFrame |
| `save_predictions(predictions, output_path)` | 将预测结果保存为 JSONL |
| `load_predictions(path)` | 加载之前保存的预测结果 |

**数据格式示例（JSONL）：**
```json
{"id": "doc1", "text": "The quick brown fox ...", "label": 1}
{"id": "doc2", "text": "...", "source_text": "...", "label": 0}
```

其中 `label=1` 表示 AI 生成，`label=0` 表示人类撰写，`source_text` 为可选的原始文档（用于配对特征提取）。

---

### `src/evaluate.py` —— 评估指标

计算与 PAN 官方评测对应的各项指标：

| 函数 | 说明 |
|------|------|
| `compute_metrics(y_true, y_pred, y_score)` | 计算准确率、宏 F1、AUC-ROC、AP 等 |
| `print_report(y_true, y_pred, y_score)` | 打印格式化评估报告（含混淆矩阵） |
| `optimal_threshold(y_true, y_score)` | 在验证集上搜索最优决策阈值 |

**主要指标：** PAN 以**宏平均 F1（F1-macro）**为主要指标，兼顾人类文本和 AI 文本的分类性能。

---

### `src/features/` —— 特征提取模块

#### `src/features/stylometric.py` —— 文体统计特征

从原始文本计算语言风格特征，无需深度学习模型，速度极快：

| 特征 | 说明 |
|------|------|
| `ttr` | 词型词例比（类型/词例数） |
| `cttr` | 修正词型词例比 |
| `mean_sent_len` | 句子平均词数 |
| `std_sent_len` | 句子长度标准差 |
| `mean_word_len` | 词语平均字符数 |
| `std_word_len` | 词语长度标准差 |
| `punct_density` | 标点密度（标点数/总字符数） |
| `hapax_ratio` | 只出现一次的词占比（词汇多样性） |
| `func_word_ratio` | 功能词（停用词）比例 |
| `sent_len_burstiness` | 句子长度突发性（方差/均值²）|

**核心思想：** AI 文本往往句子结构更均匀，TTR 较低，功能词比例稳定。

#### `src/features/embeddings.py` —— 句子嵌入特征

使用 `sentence-transformers`（默认 `all-MiniLM-L6-v2`）计算语义嵌入特征：

| 特征 | 说明 |
|------|------|
| `embedding` | 文档级嵌入向量（384 维） |
| `self_sim` | 文档内句子嵌入的平均两两余弦相似度（自相似度） |
| `source_sim` | 可疑文档与原始文档嵌入之间的余弦相似度（配对模式） |

**核心思想：** AI 生成文本的内句自相似度更高（语义均匀）；与来源文档相似度高说明存在生成式改写。

#### `src/features/perplexity.py` —— 困惑度特征

使用 GPT-2 等因果语言模型计算困惑度相关特征：

| 特征 | 说明 |
|------|------|
| `perplexity` | GPT-2 困惑度（AI 文本更低） |
| `log_likelihood` | 评分模型下的平均对数概率 |
| `llr` | 对数似然比（DetectGPT 风格，正值→AI 生成） |
| `burstiness` | 每词元对数概率的方差（人类文本更高） |
| `entropy` | 词元对数概率分布的香农熵 |

---

### `src/models/` —— 分类模型模块

#### `src/models/classifier.py` —— 监督分类器

提供两种分类器：

**`FineTunedClassifier`** —— 微调 DeBERTa-v3-base（或其他 Transformer）进行序列分类
- 适合有 GPU 和足量标注数据的场景
- 使用 HuggingFace `Trainer` 训练，支持早停
- `fit()` 训练，`predict()` 推理，`save()/load()` 持久化

**`FeatureClassifier`** —— 基于手工特征的梯度提升分类器
- 适合 CPU 算力有限的场景
- 使用 scikit-learn `GradientBoostingClassifier` + `StandardScaler`
- 输入为 `stylometric.py` + `embeddings.py` 拼接的特征向量
- 模型以 `joblib` 格式保存

#### `src/models/zero_shot.py` —— 零样本检测器

无需训练，直接基于统计信号检测 AI 文本：

| 检测器 | 方法 | 特点 |
|--------|------|------|
| `PerplexityThresholdDetector` | 困惑度低于阈值 → AI 生成 | 基于 GPT-2，最简单 |
| `LLRDetector` | 对数似然比 > 阈值 → AI 生成 | DetectGPT 风格 |
| `RobertaDetector` | 调用 `roberta-base-openai-detector` | 专用检测模型 |

#### `src/models/ensemble.py` —— 集成检测器

组合多个基础检测器以提升性能：

| 集成方式 | 说明 |
|----------|------|
| `WeightedAverageEnsemble` | 各检测器得分的加权平均 |
| `StackingEnsemble` | 用逻辑回归作为元学习器，以基础检测器得分为特征 |

---

## `docs/` 文档目录说明

### `docs/BEGINNER_PROJECT_GUIDE.md` —— 零基础入门指南（中文）

面向第一次参加 PAN/TIRA 测评的同学，详细介绍：
- 任务到底要做什么（文本对齐子任务解释）
- 本地如何运行 baseline
- 如何用验证集测分
- 如何调参数
- 如何构建 Docker 镜像
- 如何提交到 TIRA 平台

**推荐第一次接触本项目时首先阅读这份文档。**

### `docs/PAN26_TASK_CHECKLIST.md` —— 任务核查清单

提交前的检查项目清单，确保所有步骤都已完成。

---

## 其他目录说明

### `notebooks/01_exploratory.ipynb` —— 探索性分析

Jupyter Notebook，用于数据集探索性分析（EDA）：查看数据分布、测试特征效果、可视化实验结果等。

### `data/raw/` —— 原始数据（需自行下载）

存放训练集、验证集、测试集的 JSONL 文件：
- `train.jsonl` —— 训练集
- `dev.jsonl` —— 验证集
- `test.jsonl` —— 测试集（无 label 字段）

### `results/` —— 模型与预测结果（自动生成）

运行 `train.py` 和 `predict.py` 后自动创建，保存训练好的模型文件和预测输出。

### `test-data/` / `pan26-spot-check-dataset/` —— 测试数据集

用于本地运行和验证 `main.py` 输出的小型数据集。

### `official-baseline-pyterrier/` —— 官方基线参考代码

PAN 官方提供的 PyTerrier 检索基线，供参考对比使用。

---

## 各组件关系图

```
                  ┌─────────────────────────────────────────┐
                  │          两个独立子任务                   │
                  └──────────────┬──────────────────────────┘
                                 │
          ┌──────────────────────┼──────────────────────────┐
          │                      │                          │
          ▼                      ▼                          ▼
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│  文本对齐子任务  │    │  检索子任务       │    │  AI文本检测（分类）   │
│  (Text Align)   │    │  (Retrieval)     │    │  (Classification)    │
└────────┬────────┘    └────────┬─────────┘    └──────────┬───────────┘
         │                      │                         │
         ▼                      ▼                         │
    ┌─────────┐          ┌───────────┐          ┌─────────┴───────────┐
    │ main.py │          │retrieve.py│          │  train.py           │
    │（核心） │          │           │          │  predict.py         │
    └─────────┘          └───────────┘          └─────────────────────┘
         │                      │                         │
    离线句子嵌入            BM25 + 密集                    │
    余弦相似度匹配          重排序（RRF）              ┌───┴────────────────┐
         │                      │                   │                    │
    输出 XML             输出 TREC run            src/features/      src/models/
    (PAN 格式)           (gzip 格式)          ┌──────┴──────────┐   ┌────┴──────────┐
                                             │  stylometric.py │   │classifier.py  │
                                             │  embeddings.py  │   │zero_shot.py   │
                                             │  perplexity.py  │   │ensemble.py    │
                                             └─────────────────┘   └───────────────┘
```

---

## 快速上手

### 场景一：运行文本对齐基线（`main.py`）

```bash
# 安装依赖
pip install -r requirements.txt

# 运行（输入目录需包含 pairs 文件、susp/ 和 src/ 子目录）
python main.py /path/to/dataset /path/to/output

# 调整相似度阈值（越低召回率越高，但误报也越多）
python main.py /path/to/dataset /path/to/output --threshold 0.78

# 输出 JSONL 格式（调试用）
python main.py /path/to/dataset /path/to/output --format jsonl
```

### 场景二：训练 AI 文本分类器（`train.py` + `predict.py`）

```bash
# 安装完整依赖
pip install -r requirements.retrieval-lite.txt

# 训练特征分类器（CPU 友好）
python train.py \
    --train data/raw/train.jsonl \
    --dev   data/raw/dev.jsonl \
    --mode  features \
    --output results/feature_clf.joblib

# 在测试集上预测
python predict.py \
    --input  data/raw/test.jsonl \
    --model  results/feature_clf.joblib \
    --mode   features \
    --output results/predictions.jsonl
```

### 场景三：运行文档检索系统（`retrieve.py`）

```bash
python retrieve.py \
    --dataset /path/to/data/dir \
    --output  output/ \
    --index   /tmp/indexes
```

### 场景四：Docker 部署（推荐用于 TIRA 提交）

```bash
docker build -t pan-plagiarism-baseline .
docker run --rm \
  -v /path/to/dataset:/input:ro \
  -v /path/to/output:/output \
  pan-plagiarism-baseline
```

---

## 常见问题

**Q：我应该从哪个文件开始看？**

A：先读 `docs/BEGINNER_PROJECT_GUIDE.md`（零基础中文指南），然后看 `main.py`。

**Q：`main.py` 和 `train.py`/`predict.py` 有什么区别？**

A：`main.py` 是**文本对齐**子任务的基线（给定文档对，找出对应段落），不需要训练。`train.py`/`predict.py` 是**AI 文本二分类**（判断单篇文档是否为 AI 生成），需要标注数据训练。

**Q：`src/` 里的代码什么时候用到？**

A：`train.py` 和 `predict.py` 会调用 `src/` 里的特征提取和模型代码。`main.py` 是独立实现，不依赖 `src/`。

**Q：我想提交到 TIRA 平台，提交哪个脚本？**

A：文本对齐子任务提交 `main.py`（通过 Docker），检索子任务提交 `retrieve.py`。

---

*更多详细信息请参见 [英文 README](README.md) 和 [零基础入门指南](docs/BEGINNER_PROJECT_GUIDE.md)。*
