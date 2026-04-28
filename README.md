# PAN 2026 – 生成式剽窃检测

> 本项目是 [PAN 2026 生成式剽窃检测共享任务](https://pan.webis.de/clef26/pan26-web/generated-plagiarism-detection.html) 的检索系统实现。  
> **核心任务**：给定由 LLM 自动生成的可疑文档，从语料库中检索出其生成来源，以 TREC run 格式输出结果。

---

## 目录

1. [任务说明](#任务说明)
2. [检索流水线原理](#检索流水线原理)
3. [环境要求](#环境要求)
4. [项目结构](#项目结构)
5. [快速开始（Step-by-Step 教程）](#快速开始step-by-step-教程)
   - [Step 1：安装依赖](#step-1安装依赖)
   - [Step 2：准备数据](#step-2准备数据)
   - [Step 3：运行检索](#step-3运行检索)
   - [Step 4：验证输出](#step-4验证输出)
   - [Step 5：TIRA 代码提交](#step-5tira-代码提交)
6. [Docker / TIRA 提交](#docker--tira-提交)
7. [常见问题 FAQ](#常见问题-faq)
8. [附录：AI 文本分类工具](#附录ai-文本分类工具)
9. [引用](#引用)

---

## 任务说明

**PAN 2026 生成式剽窃检测**是一个经典的信息检索任务：

- 每个**可疑文档（查询）** 均由未公开的 LLM 自动生成，该模型被指示基于**至少两篇源文档**综合撰写新的科学文本。
- 给定一批可疑文档和一个源文档语料库，目标是为每个可疑文档检索出**最多 1000 篇最可能的来源文档**。
- 结果以 **TREC run 格式**（gzip 压缩）提交。

### 输出格式

每行一条结果：

```
qid Q0 doc_id rank score system_tag
```

| 字段 | 说明 |
|------|------|
| `qid` | 可疑文档的查询 ID |
| `Q0` | 固定字符串，始终为 `Q0` |
| `doc_id` | 候选源文档的 ID |
| `rank` | 该文档在当前查询结果中的排名（从 0 开始） |
| `score` | 相关性得分（浮点数，**必须按降序排列**） |
| `system_tag` | 识别你的系统的标签字符串 |

> ⚠️ `trec_eval` 依赖 `score` 字段排序，而非 `rank` 字段。请确保得分严格降序，避免并列分数问题。

---

## 检索流水线原理

本系统在官方 BM25 基线基础上实现了三阶段改进流水线：

| 阶段 | 方法 | 改进说明 |
|------|------|---------|
| 1 | **多子查询 BM25** | 将长文档拆成 N 个片段，对每个片段独立检索，扩大初始召回范围（BM25 对极长查询效果会下降） |
| 2 | **RRF 融合** | Reciprocal Rank Fusion 将多组 BM25 结果融合为更稳定的候选集 |
| 3 | **密集向量重排序** | `sentence-transformers` 计算语义余弦相似度，对候选集重排序（捕捉 AI 改写后词面不同但语义相近的来源） |

### 对比官方基线

| 对比点 | 官方基线（baseline-pyterrier） | 本系统 |
|--------|-------------------------------|--------|
| 查询方式 | 单一全文查询 | 多子查询（默认 5 个） |
| 融合策略 | 无 | RRF |
| 语义理解 | 无（纯词袋） | sentence-transformers 重排序 |
| 改写型剽窃 | 难以检出 | 可通过语义相似度检出 |

---

## 环境要求

| 依赖 | 版本要求 |
|------|---------|
| Python | ≥ 3.10 |
| Java（PyTerrier 依赖） | ≥ 11 |
| PyTorch | ≥ 2.1.0 |
| CUDA（可选，重排序加速） | ≥ 11.8 |
| 内存（RAM） | ≥ 8 GB |
| 磁盘 | ≥ 10 GB（模型 + 索引） |

---

## 项目结构

```
.
├── data/
│   ├── raw/          ← 存放下载的 PAN 语料库
│   └── processed/    ← 预处理 / 缓存的特征（自动生成）
├── notebooks/
│   └── 01_exploratory.ipynb   ← 数据探索笔记本
├── results/          ← 检索结果输出目录
├── src/
│   ├── __init__.py
│   ├── data_loader.py           ← 加载 / 保存 JSONL 语料库（分类工具）
│   ├── evaluate.py              ← 评估指标（分类工具）
│   ├── features/
│   │   ├── perplexity.py        ← GPT-2 困惑度与 LLR 特征（分类工具）
│   │   ├── stylometric.py       ← 文体特征（分类工具）
│   │   └── embeddings.py        ← 句子 Transformer 嵌入特征
│   └── models/
│       ├── zero_shot.py         ← 零样本检测器（分类工具）
│       ├── classifier.py        ← DeBERTa / GBM 分类器（分类工具）
│       └── ensemble.py          ← 集成策略（分类工具）
├── retrieve.py       ← 检索入口（本任务核心脚本）
├── train.py          ← 分类器训练入口（附录工具）
├── predict.py        ← 分类器推理入口（附录工具）
├── Dockerfile        ← TIRA / PAN 提交容器
└── requirements.txt  ← Python 依赖列表
```

---

## 快速开始（Step-by-Step 教程）

### Step 1：安装依赖

```bash
git clone https://github.com/Abraham-wy/Generative-Plagiarism-Detection.git
cd Generative-Plagiarism-Detection

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate           # Linux / macOS
# .venv\Scripts\activate            # Windows

# 安装依赖
pip install -r requirements.txt
# 国内加速：pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载 NLTK 数据（首次运行需要）
python -c "import nltk; [nltk.download(p) for p in ('punkt', 'punkt_tab', 'stopwords')]"

# 确认 Java 已安装（PyTerrier 需要）
# Ubuntu/Debian: sudo apt install default-jdk-headless
# macOS:         brew install openjdk@17
java -version
```

---

### Step 2：准备数据

#### 方式一：使用 ir_datasets（TIRA 平台）

在 [TIRA](https://www.tira.io/) 注册后，通过 ir_datasets API 直接加载：

```python
from tira.third_party_integrations import ir_datasets

# spot-check 训练集（可用于本地调试）
dataset = ir_datasets.load("pan26-generated-plagiarism-detection/spot-check-dataset-20260227-training")

# 查看语料库（待检索的源文档）
for doc in dataset.docs_iter():
    print(doc.doc_id, doc.default_text()[:80])

# 查看查询（可疑文档）
for query in dataset.queries_iter():
    print(query.query_id, query.default_text()[:80])
```

#### 方式二：使用本地文件

从 [Zenodo](https://zenodo.org/records/19038846) 下载数据集，放入本地目录（如 `test-data/`）：

```
test-data/
├── corpus.jsonl.gz    ← 所有候选源文档（待检索）
└── queries.jsonl      ← 所有可疑文档（查询）
```

文件格式：

```jsonc
// corpus.jsonl.gz（每行）
{"doc_id": "123", "default_text": "arXiv 预印本正文..."}

// queries.jsonl（每行）
{"qid": "1", "query": "可疑文档的全文..."}
```

验证下载完整性：

```bash
md5sum test-data/*
# 期望输出：
# 31baa52aac61d768ad555112e4521082  test-data/corpus.jsonl.gz
# 3eb502962505ea4d22af6546c9286042  test-data/queries.jsonl
```

---

### Step 3：运行检索

#### 使用 ir_datasets（TIRA 数据集 ID）

```bash
python retrieve.py \
    --dataset pan26-generated-plagiarism-detection/spot-check-dataset-20260227-training \
    --output  output/ \
    --index   /tmp/indexes
```

#### 使用本地文件目录

```bash
python retrieve.py \
    --dataset test-data/ \
    --output  output/ \
    --index   /tmp/indexes
```

#### 纯 BM25 模式（关闭密集重排序，速度更快）

```bash
python retrieve.py \
    --dataset test-data/ \
    --output  output/ \
    --index   /tmp/indexes \
    --no-rerank
```

#### 全部参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | ir_datasets ID 或本地数据目录路径 |
| `--output` | 必填 | 输出目录（`run.txt.gz` 将写入此目录） |
| `--index` | 必填 | BM25 索引目录（首次运行自动构建） |
| `--n-sub-queries` | `5` | 每条查询拆分的子查询数量 |
| `--sub-query-tokens` | `64` | 每条子查询截断的最大词数 |
| `--bm25-top-k` | `200` | BM25 每条子查询的候选文档数 |
| `--final-top-k` | `1000` | 最终每条查询输出的文档数（≤1000） |
| `--rerank/--no-rerank` | `--rerank` | 是否启用 sentence-transformer 密集重排序 |
| `--rerank-model` | `all-MiniLM-L6-v2` | sentence-transformers 模型名称 |
| `--tag` | `pan26-retrieval` | TREC run 文件中的系统标识符 |

---

### Step 4：验证输出

```bash
# 查看前 5 行
zcat output/run.txt.gz | head -5
# 期望输出类似：
# 1 Q0 2301.12345 0 0.923456 pan26-retrieval
# 1 Q0 2207.09876 1 0.887123 pan26-retrieval
# ...

# 统计每个查询的检索数量
zcat output/run.txt.gz | awk '{print $1}' | sort | uniq -c

# 用 trec_eval 评估（需要 qrels 文件）
# gunzip -c output/run.txt.gz > /tmp/run.txt
# trec_eval -m map -m ndcg_cut.10 qrels.txt /tmp/run.txt
```

---

### Step 5：TIRA 代码提交

```bash
# 测试命令（--dry-run 不真正上传）
tira-cli code-submission \
    --path . \
    --task pan26-generated-plagiarism-detection \
    --dataset spot-check-dataset-20260227-training \
    --command 'python /app/retrieve.py --dataset $inputDataset --index /tmp/indexes --output $outputDir' \
    --dry-run

# 确认无误后去掉 --dry-run 正式提交
tira-cli code-submission \
    --path . \
    --task pan26-generated-plagiarism-detection \
    --dataset spot-check-dataset-20260227-training \
    --command 'python /app/retrieve.py --dataset $inputDataset --index /tmp/indexes --output $outputDir'
```

> **注意**：TIRA 平台上 `$inputDataset` 和 `$outputDir` 是自动注入的环境变量。

---

## Docker / TIRA 提交

```bash
# 构建镜像
docker build -t pan26-retrieval .

# 本地测试（使用本地数据目录）
docker run --rm \
  -v $(pwd)/test-data:/input \
  -v $(pwd)/output:/output \
  pan26-retrieval

# 关闭密集重排序（纯 BM25，速度更快）
docker run --rm \
  -v $(pwd)/test-data:/input \
  -v $(pwd)/output:/output \
  -e RETRIEVAL_RERANK=false \
  pan26-retrieval
```

**Docker 环境变量说明**：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RETRIEVAL_DATASET` | `/input` | ir_datasets ID 或本地数据目录路径 |
| `RETRIEVAL_INDEX` | `/tmp/indexes` | BM25 索引目录 |
| `RETRIEVAL_TAG` | `pan26-retrieval` | TREC run 系统标识符 |
| `RETRIEVAL_RERANK` | `true` | `true` 启用密集重排序，`false` 纯 BM25+RRF |

---

## 常见问题 FAQ

**Q1：运行时提示 Java 相关错误，怎么办？**

> PyTerrier 底层依赖 Terrier（Java 实现），需要安装 JDK ≥ 11：
> ```bash
> # Ubuntu / Debian
> sudo apt install default-jdk-headless
> # macOS
> brew install openjdk@17
> # 验证
> java -version
> ```

**Q2：检索输出的 `run.txt.gz` 能用 trec_eval 评估吗？**

> 可以。下载 `trec_eval` 后：
> ```bash
> gunzip -c output/run.txt.gz > /tmp/run.txt
> trec_eval -m map -m ndcg_cut.10 qrels.txt /tmp/run.txt
> ```
> PAN 2026 官方主要评估指标为 MAP（平均精度均值）。

**Q3：如何加快检索速度？**

> 1. 使用 `--no-rerank` 禁用密集重排序，仅使用 BM25+RRF（速度提升 3-5 倍）
> 2. 减少 `--n-sub-queries`（如从 5 改为 3）
> 3. 减少 `--bm25-top-k`（如从 200 改为 100）
> 4. 减少 `--final-top-k`（调试时可先用 100，提交时改回 1000）

**Q4：BM25 索引已经构建过，还需要重新构建吗？**

> 不需要。只要 `--index` 目录下存在 `data.properties` 文件，脚本会直接加载已有索引。  
> 只有在 `--index` 目录为空或不存在时才会重新构建。

**Q5：得分出现并列（tie）怎么办？**

> `trec_eval` 依赖 `score` 字段而非 `rank` 字段对文档排序。本系统使用连续浮点数（余弦相似度或 RRF 分）作为得分，理论上不存在完全并列。若使用自定义评分函数，请确保得分有区分度。

---

## 附录：AI 文本分类工具

> 本仓库同时包含一套独立的 AI 文本分类工具（`train.py` / `predict.py`），可用于判断给定文本是**人类撰写**还是 **AI 生成/改写**。  
> **这不是 PAN 2026 检索任务的一部分**，属于仓库的扩展研究工具。

### 检测原理

| 模式 | 方法 | 适用场景 |
|------|------|---------|
| `zeroshot` | GPT-2 困惑度阈值 | 无需训练数据，快速验证 |
| `features` | 文体 + 嵌入特征 → GBM | CPU 友好，训练快 |
| `finetune` | 微调 DeBERTa-v3-base | 效果最佳，需要 GPU |
| `roberta` | 预训练 RoBERTa 检测器 | 无需训练，开箱即用 |

### 训练

```bash
# 零样本（在训练集上寻找最优阈值）
python train.py --train data/raw/train.jsonl --mode zeroshot --output results/zeroshot_config.json

# 特征分类器
python train.py --train data/raw/train.jsonl --dev data/raw/dev.jsonl \
    --mode features --output results/feature_clf.joblib

# 微调 DeBERTa（需要 GPU）
python train.py --train data/raw/train.jsonl --dev data/raw/dev.jsonl \
    --mode finetune --model microsoft/deberta-v3-base --output results/deberta_model \
    --epochs 3 --batch-size 8 --learning-rate 2e-5
```

### 推理

```bash
python predict.py --input data/raw/test.jsonl --mode features \
    --model results/feature_clf.joblib --output results/predictions.jsonl
```

输出格式（每行一个 JSON）：

```jsonc
{"id": "doc001", "score": 0.87, "label": 1}   // label=1: AI 生成
{"id": "doc002", "score": 0.12, "label": 0}   // label=0: 人类撰写
```

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
