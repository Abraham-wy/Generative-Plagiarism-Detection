#!/usr/bin/env python3
"""
PAN26 Task 4 submission entry point: source retrieval with BM25.

Input:  $inputDataset/{corpus.jsonl.gz, queries.jsonl}
Output: $outputDir/run.txt (TREC format)
"""

import argparse, gzip, json, math, os, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class BM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1=k1; self.b=b
        self.doc_ids=[]; self.doc_lens=[]
        self.postings=defaultdict(list)
        self.N=0; self.avgdl=0.0; self.idf={}

    def index(self, path):
        t0=time.time()
        opener=gzip.open if str(path).endswith('.gz') else open
        with opener(path,'rt',encoding='utf-8',errors='replace') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                d=json.loads(line)
                did=d.get("doc_id") or d.get("qid")
                text=d.get("default_text") or ""
                tokens=tokenize(text)
                self.doc_ids.append(did); self.doc_lens.append(len(tokens)); self.N+=1
                tf=defaultdict(int)
                for t in tokens: tf[t]+=1
                for term,freq in tf.items(): self.postings[term].append((self.N-1,freq))
        self.avgdl=np.mean(self.doc_lens) if self.doc_lens else 1.0
        for term,posting in self.postings.items():
            df=len(posting)
            self.idf[term]=math.log((self.N-df+0.5)/(df+0.5)+1.0)
        print(f"  Indexed {self.N} docs ({time.time()-t0:.0f}s)")

    def search(self, text, top_k=10):
        tokens=tokenize(text)
        qtf=defaultdict(int)
        for t in tokens: qtf[t]+=1
        terms=sorted([(t,self.idf.get(t,0)) for t in qtf],key=lambda x:x[1],reverse=True)[:100]
        scores=defaultdict(float)
        for term,idf_val in terms:
            if idf_val==0: continue
            for doc_idx,tf in self.postings.get(term,[]):
                dl=self.doc_lens[doc_idx]
                scores[doc_idx]+=idf_val*(tf*(self.k1+1))/(tf+self.k1*(1-self.b+self.b*dl/self.avgdl))
        ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:top_k]
        return [(self.doc_ids[idx],score) for idx,score in ranked if score>0]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path(os.environ.get("inputDataset","data/pan26/test-dataset")))
    p.add_argument("--output",type=Path,default=Path(os.environ.get("outputDir","/tmp/pan26_output")))
    p.add_argument("--top-k",type=int,default=10)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)

    corp=args.input/"corpus.jsonl"
    if not corp.exists(): corp=args.input/"corpus.jsonl.gz"
    queries=args.input/"queries.jsonl"

    print(f"Corpus: {corp}")
    print(f"Queries: {queries}")

    idx=BM25()
    idx.index(corp)

    qids,qtexts=[],[]
    with open(queries,encoding='utf-8') as f:
        for line in f:
            q=json.loads(line.strip())
            qids.append(q.get("qid") or q.get("query_id"))
            qtexts.append(q.get("query") or q.get("default_text") or "")
    print(f"  {len(qids)} queries")

    out=args.output/"run.txt"
    t0=time.time()
    with open(out,'w',encoding='utf-8') as f:
        for i,(qid,qtext) in enumerate(zip(qids,qtexts)):
            for rank,(doc_id,score) in enumerate(idx.search(qtext,args.top_k),1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} bm25\n")
            if (i+1)%1000==0: print(f"  {i+1}/{len(qids)} ({time.time()-t0:.0f}s)")

    print(f"Done: {len(qids)} queries in {time.time()-t0:.0f}s")
    print(f"Output: {out}")

if __name__=="__main__":
    main()
