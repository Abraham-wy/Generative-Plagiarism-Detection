"""Full segmentation retrieval with progress tracking and auto-save."""
import json, re, math, time, sys
from collections import defaultdict
import numpy as np

def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())

print('Building BM25 index...', flush=True)
t0=time.time()
doc_ids=[]; doc_lens=[]
postings=defaultdict(list)
N=0
with open('data/pan25_retrieval/train/corpus.jsonl') as f:
    for line in f:
        d=json.loads(line.strip())
        did=d.get('doc_id') or d.get('qid')
        tokens=tokenize(d.get('default_text') or '')
        doc_ids.append(did); doc_lens.append(len(tokens)); N+=1
        tf=defaultdict(int)
        for t in tokens: tf[t]+=1
        for term,freq in tf.items(): postings[term].append((N-1,freq))
        if N%20000==0: print(f'  {N} docs ({time.time()-t0:.0f}s)', flush=True)
avgdl=np.mean(doc_lens)
idf={}
for term,posting in postings.items():
    df=len(posting)
    idf[term]=math.log((N-df+0.5)/(df+0.5)+1.0)
print(f'  {N} docs indexed ({time.time()-t0:.0f}s)', flush=True)

def search(text, top_k=30, max_terms=30):
    tokens=tokenize(text)
    qtf=defaultdict(int)
    for t in tokens: qtf[t]+=1
    terms=sorted([(t,idf.get(t,0)) for t in qtf],key=lambda x:x[1],reverse=True)[:max_terms]
    scores=defaultdict(float)
    for term,idf_val in terms:
        if idf_val==0: continue
        for doc_idx,tf in postings.get(term,[]):
            dl=doc_lens[doc_idx]
            scores[doc_idx]+=idf_val*(tf*2.2)/(tf+1.2*(0.25+0.75*dl/avgdl))
    ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:top_k]
    return [(doc_ids[idx],score) for idx,score in ranked if score>0]

def paragraph_split(text):
    paras=[p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks=[]; buf=''
    for p in paras:
        if len(buf)+len(p)>3000 and len(buf)>800:
            chunks.append(buf.strip()); buf=p
        else: buf=buf+'\n\n'+p if buf else p
    if buf.strip() and len(buf.strip())>800: chunks.append(buf.strip())
    return chunks if chunks else [text[:3000]]

# Load qrels
qrels={}
with open('data/pan25_retrieval/train/qrels.txt') as f:
    for line in f:
        qid,_,doc,rel=line.strip().split()
        if int(rel)>0: qrels[qid]=doc

# Load queries
qids=[]; qtexts=[]
with open('data/pan25_retrieval/train/queries.jsonl') as f:
    for line in f:
        q=json.loads(line.strip())
        qids.append(q.get('qid') or q.get('query_id'))
        qtexts.append(q.get('query') or '')

# Also load BM25 top-100 for R@100
bm25_100=defaultdict(set)
with open('data/bm25_top100_train.jsonl') as f:
    for line in f:
        q=json.loads(line.strip())
        for c in q['candidates']: bm25_100[q['qid']].add(c['doc_id'])

print(f'\nRetrieving {len(qids)} queries (segmented if >20K chars)...', flush=True)
t0=time.time()
segmented=0
OUT='data/run_train_segment_full.txt'

with open(OUT,'w') as out:
    for qi,(qid,qtext) in enumerate(zip(qids,qtexts)):
        is_long = len(qtext) > 20000
        if is_long:
            chunks=paragraph_split(qtext)
            if len(chunks)<=1:
                results=search(qtext,top_k=10)
            else:
                segmented+=1
                all_res=[search(c,top_k=30) for c in chunks]
                n_total=len(all_res)
                merged_scores=defaultdict(float)
                merged_count=defaultdict(int)
                for res in all_res:
                    seen=set()
                    for doc_id,score in res:
                        merged_scores[doc_id]+=score
                        if doc_id not in seen:
                            merged_count[doc_id]+=1; seen.add(doc_id)
                final={}
                for doc_id in merged_scores:
                    coverage=merged_count[doc_id]/n_total
                    final[doc_id]=merged_scores[doc_id]*(1.0+coverage)
                results=sorted(final.items(),key=lambda x:x[1],reverse=True)[:10]
        else:
            results=search(qtext,top_k=10)

        for rank,(doc_id,score) in enumerate(results,1):
            out.write(f'{qid} Q0 {doc_id} {rank} {score:.6f} segment\n')

        if (qi+1)%5000==0:
            elapsed=time.time()-t0
            print(f'  {qi+1}/{len(qids)} ({elapsed:.0f}s) seg={segmented}', flush=True)

elapsed=time.time()-t0
print(f'\nDone: {len(qids)} queries in {elapsed:.0f}s ({segmented} segmented)', flush=True)

# Evaluate
run10=defaultdict(list)
with open(OUT) as f:
    for line in f:
        p=line.strip().split()
        if int(p[3])<=10: run10[p[0]].append(p[2])

r10=r100=nd=mr=0; total=0
for qid in qrels:
    rel=qrels[qid]; ranked=run10.get(qid,[])
    if not ranked: continue; total+=1
    if rel in ranked[:10]: r10+=1
    if rel in bm25_100.get(qid,set()) or rel in ranked[:10]: r100+=1
    rels=[1 if d==rel else 0 for d in ranked[:10]]
    nd+=sum((2**r-1)/math.log2(i+2) for i,r in enumerate(rels))
    for i,d in enumerate(ranked[:10],1):
        if d==rel: mr+=1.0/i; break

n=max(total,1)
print(f'\n=== FINAL: Segmented BM25 ===')
print(f'R@10={r10/n:.4f}  R@100={r100/n:.4f}  nDCG@10={nd/n:.4f}  MRR={mr/n:.4f}')
print(f'Output: {OUT}')
