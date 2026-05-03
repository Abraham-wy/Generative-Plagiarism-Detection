#!/usr/bin/env python3
"""
retrieve.py
-----------
PAN 2026 生成式剽窃检测 —— 检索子任务入口。

任务描述
~~~~~~~~
给定一批由 LLM 自动生成的可疑文档（查询），在语料库中检索最多 1000 篇
最可能作为其生成来源的文档，并以 TREC run 格式输出结果。

检索策略（三级流水线）
~~~~~~~~~~~~~~~~~~~~~
1. **一阶段 BM25（多子查询 + RRF）**
   - 将可疑文档按句子分段，从中均匀抽取若干子查询片段
   - 对每个子查询单独用 BM25 检索 top-k 文档
   - 使用 Reciprocal Rank Fusion（RRF）融合多组结果，得到候选集（默认 200）

2. **二阶段密集向量重排序（Dense Re-ranking，可选）**
   - 用 sentence-transformers（all-MiniLM-L6-v2）将可疑文档与候选文档编码为向量
   - 计算余弦相似度，按相似度重排前 200 候选，最终取 top-1000

3. **输出**
   - TREC run 格式（gzip 压缩）：run.txt.gz

用法
~~~~
本地文件（corpus.jsonl.gz + queries.jsonl）::

    python retrieve.py \\
        --dataset /path/to/data/dir \\
        --output  output/ \\
        --index   /tmp/indexes

ir_datasets（TIRA 平台或本地已安装）::

    python retrieve.py \\
        --dataset pan26-generated-plagiarism-detection/spot-check-dataset-20260227-training \\
        --output  output/ \\
        --index   /tmp/indexes

关闭密集重排序（纯 BM25 模式，速度最快）::

    python retrieve.py \\
        --dataset /path/to/data/dir \\
        --output  output/ \\
        --index   /tmp/indexes \\
        --no-rerank

TIRA 代码提交::

    tira-cli code-submission \\
        --path . \\
        --task pan26-generated-plagiarism-detection \\
        --dataset spot-check-dataset-20260227-training \\
        --command '/retrieve.py --dataset $inputDataset --index /tmp/indexes --output $outputDir' \\
        --dry-run
"""

from __future__ import annotations

import gzip
import logging
import os
from pathlib import Path
from typing import Iterator, List, NamedTuple, Optional

import click
import nltk
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK 数据（首次运行自动下载）
# ---------------------------------------------------------------------------
for _pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.tokenize import sent_tokenize  # noqa: E402  (after nltk download)

# ---------------------------------------------------------------------------
# 简单数据结构
# ---------------------------------------------------------------------------

class Document(NamedTuple):
    doc_id: str
    text: str

    def default_text(self) -> str:  # ir_datasets 兼容接口
        return self.text


class Query(NamedTuple):
    query_id: str
    text: str

    def default_text(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _load_local_dataset(data_dir: Path):
    """
    从本地目录加载 corpus.jsonl.gz 和 queries.jsonl。
    返回一个鸭子类型对象，与 ir_datasets API 兼容。
    """
    import json

    corpus_path = data_dir / "corpus.jsonl.gz"
    queries_path = data_dir / "queries.jsonl"

    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus.jsonl.gz not found in {data_dir}")
    if not queries_path.exists():
        raise FileNotFoundError(f"queries.jsonl not found in {data_dir}")

    class LocalDataset:
        def docs_iter(self) -> Iterator[Document]:
            with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    yield Document(doc_id=str(obj["doc_id"]), text=str(obj["default_text"]))

        def queries_iter(self) -> Iterator[Query]:
            with open(queries_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    yield Query(query_id=str(obj["qid"]), text=str(obj["query"]))

    return LocalDataset()


def load_dataset(dataset_id_or_path: str):
    """
    智能加载数据集：
    - 如果参数是一个存在的本地路径（目录），直接读取 corpus.jsonl.gz + queries.jsonl
    - 否则，通过 tira.third_party_integrations.ir_datasets 加载（TIRA 平台或 ir_datasets 注册集）
    """
    local_path = Path(dataset_id_or_path)
    if local_path.is_dir():
        log.info("从本地目录加载数据集：%s", local_path)
        return _load_local_dataset(local_path)
    else:
        log.info("通过 ir_datasets 加载数据集：%s", dataset_id_or_path)
        from tira.third_party_integrations import ir_datasets
        return ir_datasets.load(dataset_id_or_path)


# ---------------------------------------------------------------------------
# BM25 索引与检索
# ---------------------------------------------------------------------------

def _get_or_build_index(ir_dataset, index_directory: Path):
    """
    若索引不存在则构建，否则直接加载。
    返回 PyTerrier Index 对象。
    """
    import pyterrier as pt

    index_directory = index_directory.resolve().absolute()

    if not (index_directory / "data.properties").exists():
        log.info("构建 BM25 索引到：%s", index_directory)
        index_directory.mkdir(parents=True, exist_ok=True)
        indexer = pt.IterDictIndexer(
            str(index_directory),
            overwrite=True,
            meta={"docno": 100, "text": 32768},
        )
        docs = (
            {"docno": d.doc_id, "text": d.default_text()}
            for d in ir_dataset.docs_iter()
        )
        indexer.index(docs)
        log.info("索引构建完成。")
    else:
        log.info("加载已有索引：%s", index_directory)

    return pt.IndexFactory.of(str(index_directory))


def _make_sub_queries(query_text: str, n_chunks: int = 5, max_tokens: int = 64) -> List[str]:
    """
    将长文档切分为若干最具区分力的子查询片段。

    策略
    ----
    1. 拆成句子，用 TF-IDF 风格打分：关键词密度 × 区分度
    2. 取 top-n 个最高分句子
    3. 截断到 max_tokens
    """
    from nltk.corpus import stopwords as nltk_stopwords

    sentences = sent_tokenize(query_text)
    if not sentences:
        return [query_text[:500]]

    if len(sentences) <= n_chunks:
        words = query_text.split()
        return [" ".join(words[:max_tokens])]

    try:
        _stopwords = set(nltk_stopwords.words("english"))
    except Exception:
        _stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
                      "and", "or", "for", "with", "on", "at", "by", "from", "that",
                      "this", "it", "as", "be", "been", "has", "have", "had", "not"}

    # Score each sentence: info_density × distinctiveness
    # info_density = meaningful_tokens / total_tokens
    # distinctiveness = average word length of meaningful tokens
    scored = []
    for i, sent in enumerate(sentences):
        tokens = [w.lower().strip(",.!?;:()[]\"'") for w in sent.split()]
        meaningful = [w for w in tokens if w not in _stopwords and len(w) > 1]
        if not meaningful:
            scored.append((i, 0.0, sent))
            continue
        info_density = len(meaningful) / max(len(tokens), 1)
        avg_len = sum(len(w) for w in meaningful) / max(len(meaningful), 1)
        distinctiveness = min(avg_len / 8.0, 1.0)  # normalize, cap at 1.0
        score = info_density * 0.6 + distinctiveness * 0.4 + (0.1 if i < 3 else 0.0)  # bonus for early sentences (often intro)
        scored.append((i, score, sent))

    # Select top n by score, preserve original order
    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([s[0] for s in scored[:n_chunks]])
    selected = [sentences[i] for i in top_indices]

    # Truncate each
    result = []
    for sent in selected:
        words = sent.split()
        result.append(" ".join(words[:max_tokens]))
    return result


def _bm25_retrieve(index, bm25, queries_df: pd.DataFrame, top_k: int = 200) -> pd.DataFrame:
    """
    对 queries_df 中的每条查询执行 BM25 检索，返回所有结果的 DataFrame。
    """
    import pyterrier as pt

    tokeniser = pt.java.autoclass(
        "org.terrier.indexing.tokenisation.Tokeniser"
    ).getTokeniser()

    queries_df = queries_df.copy()
    queries_df["query"] = queries_df["query"].apply(
        lambda q: " ".join(tokeniser.getTokens(q))
    )
    # 过滤空查询
    queries_df = queries_df[queries_df["query"].str.strip() != ""]

    if queries_df.empty:
        return pd.DataFrame(columns=["qid", "docno", "score", "rank"])

    bm25_top_k = bm25 % top_k
    return bm25_top_k(queries_df)


def _run_fusion_pipeline(
    ir_dataset,
    index_directory: Path,
    output_file: Path,
    query_texts: dict[str, str],
    n_sub_queries: int,
    sub_query_tokens: int,
    bm25_top_k: int,
    final_top_k: int,
    rerank: bool,
    rerank_model: str,
    system_tag: str,
    w_bm25: float,
    w_dense: float,
    w_ngram: float,
    cross_encoder: bool = False,
) -> None:
    """三路加权融合流水线：BM25 + Dense + N-gram 可选 → RRF → rerank。"""
    import pyterrier as pt

    # Build BM25 index
    index = _get_or_build_index(ir_dataset, index_directory)
    bm25_retriever = pt.terrier.Retriever(index, wmodel="BM25", num_results=bm25_top_k)

    # Load all docs once
    doc_texts_all: dict[str, str] = {}
    for d in ir_dataset.docs_iter():
        doc_texts_all[d.doc_id] = d.default_text()

    ranked_paths: list[pd.DataFrame] = []
    weights: list[float] = []

    # Path A: BM25 + sub-queries → RRF (internal)
    log.info("融合-路径A: BM25 多子查询 …")
    bm25_ranked: list[pd.DataFrame] = []
    for sub_idx in range(n_sub_queries):
        rows = []
        for qid, q_text in query_texts.items():
            sub_qs = _make_sub_queries(q_text, n_chunks=n_sub_queries, max_tokens=sub_query_tokens)
            sq = sub_qs[min(sub_idx, len(sub_qs) - 1)]
            rows.append({"qid": qid, "query": sq})
        sub_df = pd.DataFrame(rows)
        result = _bm25_retrieve(index, bm25_retriever, sub_df, top_k=bm25_top_k)
        if not result.empty:
            bm25_ranked.append(result)
    if bm25_ranked:
        bm25_fused = _reciprocal_rank_fusion(bm25_ranked, top_n=bm25_top_k * n_sub_queries)
        ranked_paths.append(bm25_fused)
        weights.append(w_bm25)

    # Path B: Dense
    log.info("融合-路径B: Dense 向量检索 …")
    dense_ranked = _dense_retrieve(query_texts, doc_texts_all, top_k=bm25_top_k, model_name=rerank_model)
    ranked_paths.append(dense_ranked)
    weights.append(w_dense)

    # Path C: N-gram (optional)
    if w_ngram > 0:
        log.info("融合-路径C: N-gram 字面匹配 …")
        ngram_ranked = _ngram_retrieve(query_texts, doc_texts_all, top_k=bm25_top_k)
        ranked_paths.append(ngram_ranked)
        weights.append(w_ngram)

    # Weighted RRF fusion
    log.info("加权 RRF 融合 (weights: BM25=%.1f Dense=%.1f Ngram=%.1f) …",
             w_bm25, w_dense, w_ngram)
    fused_df = _reciprocal_rank_fusion(ranked_paths, top_n=final_top_k, weights=weights)

    # Rerank (optional)
    if rerank:
        candidate_docnos = set(fused_df["docno"].tolist())
        doc_texts_subset = {doc_id: doc_texts_all.get(doc_id, "") for doc_id in candidate_docnos}
        if cross_encoder:
            log.info("使用 cross-encoder 精排 …")
            final_df = _cross_encoder_rerank(
                fused_df,
                query_texts=query_texts,
                doc_texts=doc_texts_subset,
                top_n=final_top_k,
            )
        else:
            log.info("加载候选文档文本用于密集重排序 …")
            final_df = _dense_rerank(
                fused_df,
                query_texts=query_texts,
                doc_texts=doc_texts_subset,
                top_n=final_top_k,
                model_name=rerank_model,
            )
    else:
        final_df = fused_df.sort_values("score", ascending=False)
        final_df = final_df.groupby("qid").head(final_top_k).reset_index(drop=True)
        final_df["rank"] = final_df.groupby("qid").cumcount()

    _write_trec_run(final_df, output_file, system_tag=system_tag)
    log.info("完成！共写出 %d 条结果。", len(final_df))


def _reciprocal_rank_fusion(
    ranked_lists: List[pd.DataFrame],
    k: int = 60,
    top_n: int = 200,
    weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Reciprocal Rank Fusion（RRF）融合多组排名结果。

    参数
    ----
    ranked_lists : List[DataFrame]
        每个 DataFrame 包含 qid、docno、rank 列。
    k : int
        RRF 公式中的平滑常数（通常取 60）。
    top_n : int
        每个 qid 保留的最大文档数。
    weights : List[float] or None
        每路检索的权重。None 表示等权重。

    返回
    ----
    融合后的 DataFrame（qid、docno、score、rank）。
    """
    from collections import defaultdict

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for w, df in zip(weights, ranked_lists):
        for _, row in df.iterrows():
            qid = str(row["qid"])
            docno = str(row["docno"])
            rank = int(row["rank"]) + 1
            scores[qid][docno] += w / (k + rank)

    rows = []
    for qid, doc_scores in scores.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        for rank_idx, (docno, score) in enumerate(sorted_docs[:top_n]):
            rows.append({"qid": qid, "docno": docno, "score": score, "rank": rank_idx})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# N-gram 字面匹配检索（独立通路，检测逐字抄袭）
# ---------------------------------------------------------------------------

def _build_ngram_index(doc_texts: dict[str, str], n: int = 5) -> dict[str, set[str]]:
    """构建 n-gram 倒排索引：ngram → {doc_id, ...}"""
    from collections import defaultdict
    index: defaultdict[str, set[str]] = defaultdict(set)
    for doc_id, text in doc_texts.items():
        text_lower = text.lower()
        for i in range(len(text_lower) - n + 1):
            ngram = text_lower[i:i + n]
            index[ngram].add(doc_id)
    return dict(index)


def _ngram_retrieve(
    query_texts: dict[str, str],
    doc_texts: dict[str, str],
    top_k: int = 200,
    n: int = 5,
) -> pd.DataFrame:
    """
    纯 n-gram 字面匹配检索：倒排索引 + Jaccard 重叠评分。
    """
    log.info("N-gram 检索：构建 %d-gram 倒排索引 …", n)
    ngram_index = _build_ngram_index(doc_texts, n=n)
    log.info("倒排索引包含 %d 个 n-gram。", len(ngram_index))

    rows = []
    for qid, q_text in query_texts.items():
        text_lower = q_text.lower()
        q_ngrams = set()
        for i in range(len(text_lower) - n + 1):
            q_ngrams.add(text_lower[i:i + n])

        if not q_ngrams:
            continue

        doc_scores: dict[str, float] = {}
        for ng in q_ngrams:
            for doc_id in ngram_index.get(ng, set()):
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0

        if not doc_scores:
            continue

        # Normalize by query n-gram count → Jaccard-like overlap ratio
        q_len = len(q_ngrams)
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k]):
            rows.append({
                "qid": qid,
                "docno": doc_id,
                "score": score / q_len,  # normalized overlap ratio
                "rank": rank,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 密集向量检索（独立通路，不依赖 BM25）
# ---------------------------------------------------------------------------

def _encode_chunks(text: str, model, chunk_size: int = 256, overlap: int = 64) -> np.ndarray:
    """将长文本分块编码后 mean pooling，返回单条 embedding。"""
    words = text.split()
    if len(words) <= chunk_size:
        return model.encode([" ".join(words)], show_progress_bar=False)[0]
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap
    emb = model.encode(chunks, show_progress_bar=False)
    return emb.mean(axis=0)


def _dense_retrieve(
    query_texts: dict[str, str],
    doc_texts: dict[str, str],
    top_k: int = 200,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    纯 dense 向量检索：全文档 chunk-based mean pooling 编码，余弦相似度召回 top-k。
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    log.info("Dense 检索：加载模型 %s", model_name)
    device = "cuda" if _cuda_available() else "cpu"
    if os.environ.get("PAN_MODEL"):
        model = SentenceTransformer(os.environ["PAN_MODEL"], device=device, local_files_only=True)
    else:
        model = SentenceTransformer(model_name, device=device)

    doc_ids = list(doc_texts.keys())
    log.info("编码 %d 篇文档 (chunk-based mean pooling) …", len(doc_ids))
    doc_embeddings = np.stack([_encode_chunks(doc_texts[d], model) for d in doc_ids])

    rows = []
    for qid, q_text in query_texts.items():
        q_emb = _encode_chunks(q_text, model).reshape(1, -1)
        sims = cosine_similarity(q_emb, doc_embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        for rank, idx in enumerate(top_idx):
            rows.append({
                "qid": qid,
                "docno": doc_ids[idx],
                "score": float(sims[idx]),
                "rank": rank,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-encoder 精排
# ---------------------------------------------------------------------------

def _cross_encoder_rerank(
    candidates_df: pd.DataFrame,
    query_texts: dict[str, str],
    doc_texts: dict[str, str],
    top_n: int = 1000,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> pd.DataFrame:
    """用 cross-encoder 对候选文档精细打分。"""
    from sentence_transformers import CrossEncoder

    model_path = os.environ.get("PAN_CROSS_ENCODER", model_name)
    log.info("Cross-encoder 精排：加载模型 %s", model_path)
    model = CrossEncoder(model_path)

    rows = []
    qids = candidates_df["qid"].unique()
    for qid in qids:
        cand = candidates_df[candidates_df["qid"] == qid].copy()
        q_text = query_texts.get(str(qid), "")
        q_words = q_text.split()
        q_text_trunc = " ".join(q_words[:256])

        doc_ids = cand["docno"].tolist()
        pairs = [(q_text_trunc, doc_texts.get(str(d), "")[:2048]) for d in doc_ids]
        scores = model.predict(pairs, show_progress_bar=False)

        cand["cross_score"] = scores
        cand = cand.sort_values("cross_score", ascending=False).head(top_n)
        for rank, (_, row) in enumerate(cand.iterrows()):
            rows.append({
                "qid": qid,
                "docno": row["docno"],
                "score": float(row["cross_score"]),
                "rank": rank,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 密集向量重排序
# ---------------------------------------------------------------------------

def _dense_rerank(
    candidates_df: pd.DataFrame,
    query_texts: dict[str, str],
    doc_texts: dict[str, str],
    top_n: int = 1000,
    batch_size: int = 128,
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    使用 sentence-transformers 对 BM25 候选集进行密集向量重排序。

    参数
    ----
    candidates_df : DataFrame
        BM25 / RRF 候选集（qid、docno、score、rank）。
    query_texts : dict[str, str]
        qid -> 查询文本映射。
    doc_texts : dict[str, str]
        docno -> 文档文本映射（仅候选文档）。
    top_n : int
        每个 qid 最终保留文档数。
    batch_size : int
        编码批次大小。
    model_name : str
        sentence-transformers 模型标识符。

    返回
    ----
    重排序后的 DataFrame（qid、docno、score、rank）。
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    log.info("加载 sentence-transformer 模型：%s", model_name)
    device = "cuda" if _cuda_available() else "cpu"
    if os.environ.get("PAN_MODEL"):
        model = SentenceTransformer(os.environ["PAN_MODEL"], device=device, local_files_only=True)
    else:
        model = SentenceTransformer(model_name, device=device)

    rows = []
    qids = candidates_df["qid"].unique()

    for qid in qids:
        cand = candidates_df[candidates_df["qid"] == qid].copy()
        q_text = query_texts.get(str(qid), "")

        doc_ids = cand["docno"].tolist()
        if not doc_ids:
            continue

        q_emb = _encode_chunks(q_text, model).reshape(1, -1)
        q_emb = q_emb / np.linalg.norm(q_emb)
        d_embs = np.stack([_encode_chunks(doc_texts.get(str(d), ""), model) for d in doc_ids])
        d_embs = d_embs / np.linalg.norm(d_embs, axis=1, keepdims=True)

        sims = cosine_similarity(q_emb, d_embs)[0]

        sorted_idx = np.argsort(sims)[::-1][:top_n]
        for rank_idx, idx in enumerate(sorted_idx):
            rows.append({
                "qid": qid,
                "docno": doc_ids[idx],
                "score": float(sims[idx]),
                "rank": rank_idx,
            })

    return pd.DataFrame(rows)


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 主流水线
# ---------------------------------------------------------------------------

def process_dataset(
    ir_dataset,
    index_directory: Path,
    output_directory: Path,
    n_sub_queries: int = 5,
    sub_query_tokens: int = 64,
    bm25_top_k: int = 200,
    final_top_k: int = 1000,
    rerank: bool = True,
    rerank_model: str = "all-MiniLM-L6-v2",
    system_tag: str = "pan26-retrieval",
    force: bool = False,
    dense_only: bool = False,
    ngram_only: bool = False,
    fusion_dense: bool = False,
    fusion_ngram: bool = False,
    w_bm25: float = 0.3,
    w_dense: float = 0.5,
    w_ngram: float = 0.2,
    cross_encoder: bool = False,
) -> None:
    """
    检索流水线：
    - 默认：BM25 多子查询 + RRF → （可选）密集重排序 → TREC 输出
    - dense_only / ngram_only：纯单路检索
    - fusion_dense / fusion_ngram：多路加权融合
    - cross_encoder：用 cross-encoder 替换 bi-encoder 精排
    """
    import pyterrier as pt

    output_file = output_directory / "run.txt.gz"
    if output_file.exists() and not force:
        log.info("输出文件已存在，跳过：%s", output_file)
        return
    if output_file.exists() and force:
        log.info("覆盖已有输出文件：%s", output_file)
        output_file.unlink()

    output_directory.mkdir(parents=True, exist_ok=True)

    # 1. 加载所有查询
    log.info("加载查询文档 …")
    query_records = list(ir_dataset.queries_iter())
    query_texts: dict[str, str] = {q.query_id: q.default_text() for q in query_records}
    log.info("共 %d 条查询。", len(query_records))

    # --- Dense-only 通路 ---
    if dense_only:
        log.info("纯 dense 向量检索模式 …")
        doc_texts_all: dict[str, str] = {}
        for d in ir_dataset.docs_iter():
            doc_texts_all[d.doc_id] = d.default_text()
        final_df = _dense_retrieve(
            query_texts=query_texts,
            doc_texts=doc_texts_all,
            top_k=final_top_k,
            model_name=rerank_model,
        )
        log.info("写出 TREC run 到：%s", output_file)
        _write_trec_run(final_df, output_file, system_tag=f"{system_tag}-dense")
        log.info("完成！共写出 %d 条结果。", len(final_df))
        return

    # --- N-gram-only 通路 ---
    if ngram_only:
        log.info("纯 n-gram 字面匹配检索模式 …")
        doc_texts_all: dict[str, str] = {}
        for d in ir_dataset.docs_iter():
            doc_texts_all[d.doc_id] = d.default_text()
        final_df = _ngram_retrieve(
            query_texts=query_texts,
            doc_texts=doc_texts_all,
            top_k=final_top_k,
        )
        log.info("写出 TREC run 到：%s", output_file)
        _write_trec_run(final_df, output_file, system_tag=f"{system_tag}-ngram")
        log.info("完成！共写出 %d 条结果。", len(final_df))
        return

    # --- Fusion 模式 ---
    if fusion_dense or fusion_ngram:
        _run_fusion_pipeline(
            ir_dataset=ir_dataset,
            index_directory=index_directory,
            output_file=output_file,
            query_texts=query_texts,
            n_sub_queries=n_sub_queries,
            sub_query_tokens=sub_query_tokens,
            bm25_top_k=bm25_top_k,
            final_top_k=final_top_k,
            rerank=rerank,
            rerank_model=rerank_model,
            system_tag=system_tag,
            w_bm25=w_bm25,
            w_dense=w_dense,
            w_ngram=w_ngram if fusion_ngram else 0.0,
            cross_encoder=cross_encoder,
        )
        return

    # 2. 构建/加载 BM25 索引
    index = _get_or_build_index(ir_dataset, index_directory)
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=bm25_top_k)

    # 3. 多子查询 BM25 检索
    log.info("执行多子查询 BM25 检索（n_sub_queries=%d, top_k=%d） …",
             n_sub_queries, bm25_top_k)
    all_ranked: List[pd.DataFrame] = []
    for sub_idx in range(n_sub_queries):
        rows = []
        for qid, q_text in query_texts.items():
            sub_qs = _make_sub_queries(q_text, n_chunks=n_sub_queries, max_tokens=sub_query_tokens)
            # 每次循环取第 sub_idx 条子查询（若不足则循环取最后一条）
            sq = sub_qs[min(sub_idx, len(sub_qs) - 1)]
            rows.append({"qid": qid, "query": sq})

        sub_df = pd.DataFrame(rows)
        result = _bm25_retrieve(index, bm25, sub_df, top_k=bm25_top_k)
        if not result.empty:
            all_ranked.append(result)

    if not all_ranked:
        log.warning("BM25 检索结果为空！")
        return

    # 4. RRF 融合
    log.info("RRF 融合 %d 组结果 …", len(all_ranked))
    fused_df = _reciprocal_rank_fusion(
        all_ranked,
        top_n=max(bm25_top_k * n_sub_queries, final_top_k),
    )

    # 5. 密集重排序（可选）
    if rerank:
        log.info("加载候选文档文本用于密集重排序 …")
        candidate_docnos = set(fused_df["docno"].tolist())
        doc_texts: dict[str, str] = {}
        for d in ir_dataset.docs_iter():
            if d.doc_id in candidate_docnos:
                doc_texts[d.doc_id] = d.default_text()
            if len(doc_texts) == len(candidate_docnos):
                break  # 全部候选已收集，提前停止

        missing_docs = len(candidate_docnos) - len(doc_texts)
        if missing_docs:
            log.warning("有 %d 个候选文档未能加载正文，将用空文本参与重排序。", missing_docs)

        log.info("对 %d 个候选文档进行密集重排序 …", len(doc_texts))
        final_df = _dense_rerank(
            fused_df,
            query_texts=query_texts,
            doc_texts=doc_texts,
            top_n=final_top_k,
            model_name=rerank_model,
        )
    else:
        # 不重排序，直接截断到 final_top_k
        final_df = fused_df.sort_values("score", ascending=False)
        final_df = final_df.groupby("qid").head(final_top_k).reset_index(drop=True)
        final_df["rank"] = final_df.groupby("qid").cumcount()

    # 6. 写出 TREC run 格式（gzip）
    log.info("写出 TREC run 到：%s", output_file)
    _write_trec_run(final_df, output_file, system_tag=system_tag)
    log.info("完成！共写出 %d 条结果。", len(final_df))


def _write_trec_run(
    df: pd.DataFrame,
    output_path: Path,
    system_tag: str = "pan26-retrieval",
) -> None:
    """
    将 DataFrame（qid、docno、score、rank）写成 TREC run 格式（gzip 压缩）。

    格式：qid Q0 docno rank score tag
    """
    # 按 qid 和 score 排序（trec_eval 依赖得分降序，不依赖 rank 字段）。
    # 排序后重新编号 rank，避免上游过滤或重排后出现 rank 与 score 顺序不一致。
    df = df.sort_values(["qid", "score", "docno"], ascending=[True, False, True]).copy()
    df["rank"] = df.groupby("qid").cumcount()

    lines = []
    for _, row in df.iterrows():
        qid = str(row["qid"])
        docno = str(row["docno"])
        rank = int(row["rank"])
        score = float(row["score"])
        lines.append(f"{qid} Q0 {docno} {rank} {score:.6f} {system_tag}\n")

    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="ir_datasets 数据集 ID（如 pan26-generated-plagiarism-detection/spot-check-dataset-20260227-training）"
         "或包含 corpus.jsonl.gz + queries.jsonl 的本地目录路径。",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="输出目录，run.txt.gz 将写入此目录。",
)
@click.option(
    "--index",
    type=click.Path(path_type=Path),
    required=True,
    help="BM25 索引存储目录（首次运行时自动构建）。",
)
@click.option(
    "--n-sub-queries",
    type=int,
    default=5,
    show_default=True,
    help="每条查询拆分的子查询数量（多子查询提升召回率）。",
)
@click.option(
    "--sub-query-tokens",
    type=int,
    default=64,
    show_default=True,
    help="每条子查询截断的最大词数。",
)
@click.option(
    "--bm25-top-k",
    type=int,
    default=200,
    show_default=True,
    help="BM25 每条子查询检索的候选文档数。",
)
@click.option(
    "--final-top-k",
    type=int,
    default=1000,
    show_default=True,
    help="最终每条查询输出的文档数（≤1000）。",
)
@click.option(
    "--rerank/--no-rerank",
    default=True,
    show_default=True,
    help="是否启用 sentence-transformer 密集重排序（禁用则为纯 BM25+RRF 模式）。",
)
@click.option(
    "--rerank-model",
    type=str,
    default="all-MiniLM-L6-v2",
    show_default=True,
    help="密集重排序使用的 sentence-transformers 模型名称。",
)
@click.option(
    "--tag",
    type=str,
    default="pan26-retrieval",
    show_default=True,
    help="TREC run 文件中的系统标识符（system tag）。",
)
@click.option(
    "--force",
    is_flag=True,
    help="如果输出目录中已存在 run.txt.gz，则覆盖它并重新运行。",
)
@click.option(
    "--dense-only",
    is_flag=True,
    help="仅使用 dense 向量检索（不经过 BM25），用于独立评测 dense 通路。",
)
@click.option(
    "--ngram-only",
    is_flag=True,
    help="仅使用 n-gram 字面匹配检索，用于独立评测 n-gram 通路。",
)
@click.option(
    "--fusion-dense",
    is_flag=True,
    help="BM25 + Dense 双路加权融合模式。",
)
@click.option(
    "--fusion-ngram",
    is_flag=True,
    help="在双路基础上加入 N-gram，三路加权融合模式。",
)
@click.option(
    "--w-bm25",
    type=float,
    default=0.3,
    show_default=True,
    help="融合时 BM25 的权重。",
)
@click.option(
    "--w-dense",
    type=float,
    default=0.5,
    show_default=True,
    help="融合时 Dense 的权重。",
)
@click.option(
    "--w-ngram",
    type=float,
    default=0.2,
    show_default=True,
    help="融合时 N-gram 的权重。",
)
@click.option(
    "--cross-encoder",
    is_flag=True,
    help="使用 cross-encoder 做最终精排（替代 bi-encoder 余弦相似度）。",
)
def main(
    dataset: str,
    output: Path,
    index: Path,
    n_sub_queries: int,
    sub_query_tokens: int,
    bm25_top_k: int,
    final_top_k: int,
    rerank: bool,
    rerank_model: str,
    tag: str,
    force: bool,
    dense_only: bool,
    ngram_only: bool,
    fusion_dense: bool,
    fusion_ngram: bool,
    w_bm25: float,
    w_dense: float,
    w_ngram: float,
    cross_encoder: bool,
) -> None:
    """PAN 2026 生成式剽窃检测 —— 改进检索系统。"""
    import pyterrier as pt
    if not pt.started():
        pt.init()

    ir_dataset = load_dataset(dataset)

    process_dataset(
        ir_dataset=ir_dataset,
        index_directory=index,
        output_directory=output,
        n_sub_queries=n_sub_queries,
        sub_query_tokens=sub_query_tokens,
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k,
        rerank=rerank,
        rerank_model=rerank_model,
        system_tag=tag,
        force=force,
        dense_only=dense_only,
        ngram_only=ngram_only,
        fusion_dense=fusion_dense,
        fusion_ngram=fusion_ngram,
        w_bm25=w_bm25,
        w_dense=w_dense,
        w_ngram=w_ngram,
        cross_encoder=cross_encoder,
    )


if __name__ == "__main__":
    main()
