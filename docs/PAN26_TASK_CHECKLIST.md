# PAN 2026 生成式剽窃检测任务清单

## 你具体要做什么

- 主任务是源文档检索：给定 LLM 生成的可疑科学文档，从 `corpus.jsonl.gz` 中找出它参考或剽窃的源文档。
- 最终产物是 `run.txt.gz`，每行必须是 TREC run 格式：

```text
qid Q0 doc_id rank score tag
```

- 每个 `qid` 最多返回 1000 个源文档，`score` 必须对同一 `qid` 降序排列。
- 官方 2026 页面当前列出的重点指标是 `nDCG@10`、`Recall@10`、`Recall@100`，所以优化时要同时关注前 10 排名质量和前 100 召回。

## 需要知道什么

- 官方 baseline 是 PyTerrier BM25 检索系统，不需要监督训练。运行 baseline 时会构建 BM25 索引，这一步是“建索引”，不是训练模型。
- 本仓库的核心脚本是 `retrieve.py`。`train.py` 和 `predict.py` 是 AI 文本二分类附录工具，不是本检索任务主线。
- 正式测试集的 qrels 在截止前隐藏，所以本地不能完整复现最终榜单分数；本地重点是流程正确、格式正确、参数实验可记录。
- TIRA 会注入 `$inputDataset` 和 `$outputDir`，提交命令要让 `retrieve.py` 读取前者、把 `run.txt.gz` 写入后者。

## 推荐执行顺序

1. 安装依赖并确认 Java 可用。

```bash
pip install -r requirements.txt
java -version
```

2. 先跑快速 BM25+RRF 模式，确认数据、索引和输出格式。

```bash
python retrieve.py --dataset test-data --output output --index /tmp/indexes --no-rerank --force
python tools/validate_run.py output/run.txt.gz
```

3. 再跑默认增强模式，加入 dense rerank。

```bash
python retrieve.py --dataset test-data --output output --index /tmp/indexes --force
python tools/validate_run.py output/run.txt.gz
```

4. 做召回参数网格实验。

```bash
python tools/run_retrieval_experiments.py \
  --dataset test-data \
  --output-root experiments \
  --n-sub-queries 3,5,8 \
  --sub-query-tokens 64,128 \
  --bm25-top-k 200,500,1000
```

5. 抽查 top-k 结果，记录错误类型。

```bash
python tools/analyze_run.py output/run.txt.gz --top-k 10 --output output/analysis.json
```

6. 如果有 qrels，再计算官方方向指标。

```bash
python tools/evaluate_run.py --run output/run.txt.gz --qrels qrels.txt
```

7. TIRA dry-run 通过后再正式提交。

```bash
tira-cli code-submission \
  --path . \
  --task pan26-generated-plagiarism-detection \
  --dataset spot-check-dataset-20260227-training \
  --command 'python /app/retrieve.py --dataset $inputDataset --index /tmp/indexes --output $outputDir' \
  --dry-run
```

## 优化思路

- 先保召回：调大 `--n-sub-queries`、`--sub-query-tokens`、`--bm25-top-k`，重点看更多真实源文档是否进入 top 100 或候选集。
- 再保排序：对比不同 `--rerank-model`，目标是让真实源文档排进 top 10，从而提升 `nDCG@10` 和 `Recall@10`。
- 每次实验都保留输出目录和 `metadata.json`，不要只记一个线上分数；要知道是哪组参数带来的变化。
- 错误分析按类型记录：词面召回失败、语义相似但源错、长文档截断、公式/数字/引用线索没有被利用、rerank 把正确源文档压低。

## 当前默认选择

- 快速调试：`--no-rerank`
- 默认提交候选：`--rerank --rerank-model all-MiniLM-L6-v2`
- 更强 rerank 备选：`multi-qa-mpnet-base-dot-v1`
- 推荐召回实验网格：
  - `--n-sub-queries`: `3,5,8`
  - `--sub-query-tokens`: `64,128`
  - `--bm25-top-k`: `200,500,1000`
