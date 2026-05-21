# PAN26 Task 4 源检索阶段——总结与心得

## 当前进度

### 1. 数据准备
- PAN25 完整数据转换为 PAN26 格式：60,592 文档 / 42,444 查询 / 43,140 qrels
- PAN14 跨域测试数据：2,920 文档 / 98 查询（ID mapping 待解决）

### 2. 核心方法演进

**第一阶段：纯 BM25（baseline）**
- 倒排索引 + BM25 评分，top-100 query terms
- 标准检索速度极快（800q / 17s）
- R@10 ≈ 0.88，MRR ≈ 0.77

**第二阶段：Chunk Retrieval（E5 语义匹配）**
- 将候选文档切分为 256-token chunks
- E5-base-v2 编码 + 余弦相似度
- Coverage-aware scoring: 0.5*top1 + 0.3*top5_mean + 0.2*coverage
- 提升显著但计算成本高，需预编码 277K chunks

**第三阶段：Query Segmentation（段落分割 + 溯源投票）**
- 将查询文档按段落拆分为语义块
- 每个查询块独立 BM25 检索
- Coverage-weighted source voting 聚合
- 这是"显式溯源图建模"的核心

### 3. 关键性能优化

Chunk BM25 检索最初慢 100 倍（1.7-3s/chunk vs 0.02s/query），根因：
- Chunk 缺少稀有词汇 → 选中 term 的 posting list 极长（覆盖 30K+ 文档）
- 修复：max_df=5000 过滤 + max_terms=20 + heapq.nlargest
- 修复后：0.15s/chunk（20x 加速）

### 4. 最终指标得分

**800 queries（seed=20260520）**

| Metric | Standard BM25 | Paragraph-Segmented BM25 | Improvement |
|--------|--------------|--------------------------|-------------|
| R@10   | 0.9200       | **0.9700**               | +5.4%       |
| R@100  | 0.9700       | **0.9912**               | +2.2%       |
| nDCG@10| 0.8122       | **0.9242**               | +13.8%      |
| MRR    | 0.7771       | **0.9092**               | +17.0%      |

**1200 queries（验证集中）**

| Metric | Standard BM25 | Paragraph-Segmented BM25 | Improvement |
|--------|--------------|--------------------------|-------------|
| R@10   | 0.8825       | **0.9692**               | +9.8%       |
| R@100  | 0.9517       | **0.9908**               | +4.1%       |
| nDCG@10| 0.7850       | **0.9234**               | +17.6%      |
| MRR    | 0.7532       | **0.9086**               | +20.6%      |

### 5. TIRA 提交

- Team: fosuai
- Task: pan26-generated-plagiarism-detection
- Software: oscillating-list
- Docker: python:3.11-slim + numpy only（极简，~200MB）
- 方法：段落分割 BM25 + 溯源投票（纯 BM25，无 E5 依赖）

## 心得与反思

### 方法层面

1. **Query Segmentation 的本质是溯源图建模**。传统检索将 query 视为整体，但抄袭检测中 query（可疑文档）由多个不同来源拼接而成。将 query 拆分为语义段落，每个段落独立检索，再通过 source voting 聚合——这本质上是在重建"这个文档的各部分分别来自哪里"的溯源图。

2. **简单的 coverage-weighted voting 非常有效**。Score(s) = Σ(chunk_scores) * (1 + coverage_ratio)。被更多 query chunk 同时命中的文档得分更高，这天然奖励了"全局一致"的来源候选。

3. **BM25 在这个任务上意外的强**。与 E5 语义匹配相比，BM25 的词法匹配在学术文本抄袭场景中有天然优势——抄袭文本保留了原文的关键词。E5 只在 chunk-level 检索中补充了语义泛化能力，但词法精确匹配是基础。

### 工程层面

4. **Posting list 遍历是 BM25 性能瓶颈**。60K 文档的倒排索引中，常见词可能有 50K+ 的 posting list entries。解决方案不是优化数据结构，而是**抑制常见词**（max_df 过滤）——这同时提升了速度和精度（常见词区分度低）。

5. **macOS 休眠问题**。长任务在笔记本上不可靠，应使用 `caffeinate` 或迁移到 Linux 服务器。`time.time()` 包含休眠时间会导致误导性的慢速报告。

6. **指标分母的陷阱**。训练集有 43,140 个 qrels 条目，但子集测试只有 800 个查询——如果指标除以 43,140 而非 800，会得到 0.0x 的虚假低分。始终用 `n = sum(1 for qid in qrels if qid in run)`。

### 下一步

- 在 Linux 服务器上跑全量 42K queries 的分段检索
- 探索更复杂的 query 分段策略（semantic split by section headers）
- 将分段检索结果输入 text alignment 阶段（第二阶段）
- 在 PAN14 跨域数据上验证泛化性（需解决 ID mapping）
