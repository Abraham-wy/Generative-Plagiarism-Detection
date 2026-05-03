# PAN26 Generated Plagiarism Detection — Retrieval Baseline

> 中文说明参阅 [README_zh.md](README_zh.md)

三路互补检索流水线：**BM25 + Dense 语义 + N-gram 字面** → 加权 RRF 融合 → 重排序。

## 架构

```
可疑文档
   │
   ├── 通路A: BM25 关键词 (权重 0.3, 兜底)
   ├── 通路B: Dense 语义向量 (权重 0.5, 主力)  
   └── 通路C: N-gram 字面匹配 (权重 0.2, 抓逐字抄袭)
   │
   ▼
  加权 RRF 融合
   │
   ▼
  Bi-encoder / Cross-encoder 精排
   │
   ▼
  top-1000 (TREC run 格式)
```

## 指标

| 指标 | 含义 |
|------|------|
| nDCG@10 | 前10结果的排序质量 |
| Recall@10 | 前10中召回的相关文档比例 |
| Recall@100 | 前100中召回的相关文档比例 |
| MRR | 第一个相关文档的排名倒数 |
| MAP | 平均精度（所有相关位置的平均） |
| P@5 | 前5结果的精度 |
| P@20 | 前20结果的精度 |

## 快速开始

```bash
# 纯 BM25（无子查询）
docker run --rm -v $(pwd)/data:/input:ro -v $(pwd)/output:/output \
  pan26-local --dataset /input --index /tmp/idx --output /output \
  --no-rerank --n-sub-queries 1

# Dense only
docker run --rm -v $(pwd)/data:/input:ro -v $(pwd)/output:/output \
  pan26-local --dataset /input --index /tmp/idx --output /output --dense-only

# N-gram only
docker run --rm -v $(pwd)/data:/input:ro -v $(pwd)/output:/output \
  pan26-local --dataset /input --index /tmp/idx --output /output --ngram-only

# 三路融合 + rerank（推荐）
docker run --rm -v $(pwd)/data:/input:ro -v $(pwd)/output:/output \
  pan26-local --dataset /input --index /tmp/idx --output /output \
  --fusion-ngram --w-bm25 0.3 --w-dense 0.5 --w-ngram 0.2

# 评测
python3 tools/evaluate_run.py --run output/run.txt.gz --qrels data/qrels.txt
```

## CLI 选项

| 选项 | 默认 | 说明 |
|------|------|------|
| `--dataset` | 必填 | 数据集目录或 ir_datasets ID |
| `--output` | 必填 | 输出目录 |
| `--index` | 必填 | 索引目录 |
| `--n-sub-queries` | 5 | BM25 子查询数 |
| `--bm25-top-k` | 200 | 每路子查询召回数 |
| `--final-top-k` | 1000 | 最终输出数 |
| `--rerank / --no-rerank` | rerank | bi-encoder 重排序 |
| `--dense-only` | off | 纯 dense 模式 |
| `--ngram-only` | off | 纯 n-gram 模式 |
| `--fusion-dense` | off | BM25+Dense 双路融合 |
| `--fusion-ngram` | off | 三路融合 |
| `--w-bm25` | 0.3 | BM25 融合权重 |
| `--w-dense` | 0.5 | Dense 融合权重 |
| `--w-ngram` | 0.2 | N-gram 融合权重 |
| `--cross-encoder` | off | 用 cross-encoder 精排 |

## 消融实验 (spot-check, 4 queries)

| 配置 | nDCG@10 | Recall@100 | MRR | MAP | P@5 |
|------|:--:|:--:|:--:|:--:|:--:|
| BM25 单路 | 1.0 | 1.0 | 1.0 | 1.0 | 0.20 |
| Dense 单路 | 1.0 | 1.0 | 1.0 | 1.0 | 0.20 |
| N-gram 单路 | 0.815 | 1.0 | 0.75 | 0.75 | 0.20 |
| BM25+Dense 融合 | 1.0 | 1.0 | 1.0 | 1.0 | 0.20 |
| 三路融合 | 1.0 | 1.0 | 1.0 | 1.0 | 0.20 |

*注：spot-check 仅 4 条查询，核心指标难以区分。正式评测以 TIRA 训练/测试集得分为准。*

## TIRA 提交

```bash
# 镜像提交
tira-cli code-submission --path . --task pan26-generated-plagiarism-detection \
  --command 'python /app/retrieve.py --dataset $inputDataset --index /tmp/indexes --output $outputDir'

# 直接上传结果
tira-cli upload --directory output --dataset generated-plagiarism-detection --system YOUR-RUN-NAME
```
