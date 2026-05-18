"""
Query Segmentation Retrieval: split query into semantic/section chunks,
retrieve each independently with BM25, then aggregate via source voting.

Pure BM25 + numpy — no E5 dependency. Docker-compatible.

Three strategies:
  A. paragraph_split: split by paragraph breaks → independent BM25
  B. fixed_chunk: rolling window chunks → source voting
  C. semantic_split: detect section headers (Introduction/Method/etc.)

Usage:
  python scripts/query_segment_retrieve.py \
    --corpus data/pan25_retrieval/train/corpus.jsonl \
    --queries data/pan25_retrieval/train/queries.jsonl \
    --qrels data/pan25_retrieval/train/qrels.txt \
    --output data/run_train_segment.txt \
    --strategy paragraph_split
"""

import argparse, json, math, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class BM25Index:
    def __init__(self, k1=1.2, b=0.75):
        self.k1=k1; self.b=b
        self.doc_ids=[]; self.doc_lens=[]
        self.postings=defaultdict(list)
        self.N=0; self.avgdl=0.0; self.idf={}

    def index(self, path):
        t0=time.time()
        with open(path, encoding='utf-8') as f:
            for line in f:
                d=json.loads(line.strip())
                did=d.get("doc_id") or d.get("qid")
                tokens=tokenize(d.get("default_text") or "")
                self.doc_ids.append(did); self.doc_lens.append(len(tokens)); self.N+=1
                tf=defaultdict(int)
                for t in tokens: tf[t]+=1
                for term,freq in tf.items(): self.postings[term].append((self.N-1,freq))
        self.avgdl=np.mean(self.doc_lens) if self.doc_lens else 1.0
        for term,posting in self.postings.items():
            df=len(posting)
            self.idf[term]=math.log((self.N-df+0.5)/(df+0.5)+1.0)
        print(f"  Indexed {self.N} docs ({time.time()-t0:.0f}s)")

    def search(self, text, top_k=50, max_terms=50):
        tokens=tokenize(text)
        qtf=defaultdict(int)
        for t in tokens: qtf[t]+=1
        terms=sorted([(t,self.idf.get(t,0)) for t in qtf],key=lambda x:x[1],reverse=True)[:max_terms]
        scores=defaultdict(float)
        for term,idf_val in terms:
            if idf_val==0: continue
            for doc_idx,tf in self.postings.get(term,[]):
                dl=self.doc_lens[doc_idx]
                scores[doc_idx]+=idf_val*(tf*(self.k1+1))/(tf+self.k1*(1-self.b+self.b*dl/self.avgdl))
        ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:top_k]
        return [(self.doc_ids[idx],score) for idx,score in ranked if score>0]


# ---- Segmentation strategies ----

def paragraph_split(text, min_chars=800, max_chars=3000):
    """Split by paragraph breaks, merging small ones."""
    paras=[p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks=[]; buf=''
    for p in paras:
        if len(buf)+len(p)>max_chars and len(buf)>min_chars:
            chunks.append(buf.strip()); buf=p
        else:
            buf=buf+'\n\n'+p if buf else p
    if buf.strip() and len(buf.strip())>min_chars: chunks.append(buf.strip())
    return chunks if chunks else [text[:max_chars]]


def fixed_chunk(text, chunk_chars=2000, stride_chars=1000):
    """Rolling window chunks of ~chunk_chars characters."""
    if len(text)<=chunk_chars: return [text]
    chunks=[]
    for i in range(0,len(text)-chunk_chars//2,stride_chars):
        chunks.append(text[i:i+chunk_chars])
    return chunks[:20]  # max 20 chunks


def semantic_split(text):
    """Split by section headers: Introduction, Method, Results, Discussion, etc."""
    sections=re.split(
        r'\n\s*(?:\d+\.?\s*)?(?:A[Bb][Ss][Tt][Rr][Aa][Cc][Tt]|'
        r'[Ii][Nn][Tt][Rr][Oo][Dd][Uu][Cc][Tt][Ii][Oo][Nn]|'
        r'[Rr][Ee][Ll][Aa][Tt][Ee][Dd]\s*[Ww][Oo][Rr][Kk]|'
        r'[Mm][Ee][Tt][Hh][Oo][Dd]|'
        r'[Ee][Xx][Pp][Ee][Rr][Ii][Mm][Ee][Nn][Tt]|'
        r'[Rr][Ee][Ss][Uu][Ll][Tt]|'
        r'[Dd][Ii][Ss][Cc][Uu][Ss][Ss][Ii][Oo][Nn]|'
        r'[Cc][Oo][Nn][Cc][Ll][Uu][Ss][Ii][Oo][Nn]|'
        r'[Rr][Ee][Ff][Ee][Rr][Ee][Nn][Cc][Ee]|'
        r'[Aa][Cc][Kk][Nn][Oo][Ww][Ll][Ee][Dd][Gg])\s*\n',
        text, flags=re.IGNORECASE
    )
    chunks=[s.strip() for s in sections if len(s.strip())>500]
    return chunks[:15] if chunks else paragraph_split(text)


# ---- Source voting aggregation ----

def vote_aggregate(all_results, method='coverage_weighted'):
    """
    Aggregate multiple ranked lists into a single ranking.

    all_results: list of [(doc_id, score), ...] from each segment.

    method='coverage_weighted': score = sum(segment_scores) * (n_segments_matched / total_segments)
    method='rrf': reciprocal rank fusion
    method='max': take max score per doc
    """
    if method=='max':
        merged={}
        for results in all_results:
            for doc_id,score in results:
                merged[doc_id]=max(merged.get(doc_id,0),score)
        return sorted(merged.items(),key=lambda x:x[1],reverse=True)

    elif method=='coverage_weighted':
        n_total=len(all_results)
        merged_scores=defaultdict(float)
        merged_count=defaultdict(int)
        for results in all_results:
            seen=set()
            for doc_id,score in results:
                merged_scores[doc_id]+=score
                if doc_id not in seen:
                    merged_count[doc_id]+=1
                    seen.add(doc_id)
        # Coverage boost: sqrt(count/total) rewards docs matched by multiple segments
        final={}
        for doc_id in merged_scores:
            coverage=merged_count[doc_id]/n_total
            final[doc_id]=merged_scores[doc_id]*(1.0+coverage)
        return sorted(final.items(),key=lambda x:x[1],reverse=True)

    elif method=='rrf':
        K=60; merged=defaultdict(float)
        for results in all_results:
            for rank,(doc_id,_) in enumerate(results,1):
                merged[doc_id]+=1.0/(K+rank)
        return sorted(merged.items(),key=lambda x:x[1],reverse=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--corpus",type=Path,required=True)
    parser.add_argument("--queries",type=Path,required=True)
    parser.add_argument("--qrels",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--strategy",default="paragraph_split",
                        choices=["paragraph_split","fixed_chunk","semantic_split"])
    parser.add_argument("--aggregation",default="coverage_weighted",
                        choices=["coverage_weighted","rrf","max"])
    parser.add_argument("--top-k",type=int,default=10)
    args=parser.parse_args()

    # Load qrels
    qrels={}
    with open(args.qrels) as f:
        for line in f:
            qid,_,doc,rel=line.strip().split()
            if int(rel)>0: qrels[qid]=doc

    # Build index
    print("Building BM25 index...")
    idx=BM25Index()
    idx.index(args.corpus)

    # Load queries
    print(f"Loading queries...")
    qids,qtexts=[],[]
    with open(args.queries,encoding='utf-8') as f:
        for line in f:
            q=json.loads(line.strip())
            qids.append(q.get("qid") or q.get("query_id"))
            qtexts.append(q.get("query") or q.get("default_text") or "")
    print(f"  {len(qids)} queries")

    # Segment and retrieve
    split_fn={"paragraph_split":paragraph_split,"fixed_chunk":fixed_chunk,"semantic_split":semantic_split}[args.strategy]
    print(f"\nSegmenting + retrieving ({args.strategy}, {args.aggregation})...")
    t0=time.time()

    with open(args.output,"w",encoding='utf-8') as out:
        for qi,(qid,qtext) in enumerate(zip(qids,qtexts)):
            chunks=split_fn(qtext)
            if len(chunks)<=1:
                # Single segment: standard BM25
                results=idx.search(qtext,top_k=args.top_k)
            else:
                # Multi-segment: retrieve each, then vote
                all_results=[idx.search(c,top_k=50) for c in chunks]
                results=vote_aggregate(all_results,method=args.aggregation)[:args.top_k]

            for rank,(doc_id,score) in enumerate(results,1):
                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} segment\n")

            if (qi+1)%2000==0:
                print(f"  {qi+1}/{len(qids)} ({time.time()-t0:.0f}s)")

    elapsed=time.time()-t0
    print(f"Done: {len(qids)} queries in {elapsed:.0f}s")

    # Evaluate
    run10=defaultdict(list)
    with open(args.output) as f:
        for line in f:
            p=line.strip().split()
            if int(p[3])<=10: run10[p[0]].append(p[2])

    r10=r100=nd=mr=0
    for qid in qrels:
        rel=qrels[qid]; ranked=run10.get(qid,[])
        if not ranked: continue
        if rel in ranked[:10]: r10+=1
        if rel in ranked[:100]: r100+=1
        rels=[1 if d==rel else 0 for d in ranked[:10]]
        nd+=sum((2**r-1)/math.log2(i+2) for i,r in enumerate(rels))
        for i,d in enumerate(ranked[:10],1):
            if d==rel: mr+=1.0/i; break
    n=len(qrels)
    print(f"\n=== {args.strategy} + {args.aggregation} ===")
    print(f"R@10={r10/n:.4f}  R@100={r100/n:.4f}  nDCG@10={nd/n:.4f}  MRR={mr/n:.4f}")
    print(f"Output: {args.output}")


if __name__=="__main__":
    main()
