"""Public retrieval evaluations, isolated from application stores and credentials.

Run: python -u benchmarks/public_retrieval/run.py --dataset litsearch
Preparation and each query are checkpointed. No generation/judging API is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'output/benchmarks/public_retrieval_20261029'
SEED = 20261029
METHODS = ['bm25', 'dense', 'hybrid_rrf', 'hybrid_rerank_rrf']
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csvwrite(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def get(url):
    import requests
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def prepare_litsearch(out):
    if (out / 'prepared.json').exists():
        return
    import fsspec
    import pyarrow.parquet as pq
    meta = get('https://datasets-server.huggingface.co/parquet?dataset=princeton-nlp/LitSearch').json()
    save(out / 'source_parquet_manifest.json', meta)
    corpus = []
    for entry in meta['parquet_files']:
        if entry['config'] != 'corpus_clean':
            continue
        cache = out / ('papers_' + entry['filename'] + '.json')
        if cache.exists():
            rows = read(cache)
        else:
            print('READ title/abstract columns', entry['filename'], flush=True)
            with fsspec.open(entry['url'], block_size=1024*1024).open() as f:
                table = pq.read_table(f, columns=['corpusid', 'title', 'abstract'])
            rows = [{'id': str(r['corpusid']), 'title': r['title'] or '', 'text': r['abstract'] or ''}
                    for r in table.to_pylist()]
            save(cache, rows)
        corpus.extend(rows)
    entry = next(e for e in meta['parquet_files'] if e['config'] == 'query')
    with fsspec.open(entry['url'], block_size=1024*1024).open() as f:
        queries_raw = pq.read_table(f).to_pylist()
    save(out / 'queries_original.json', queries_raw)
    queries = [{'id': f'lit{i:04d}', 'query': q['query'], 'group': q['query_set'],
                'qrels': {str(d): 1 for d in q['corpusids']},
                'specificity': q['specificity'], 'quality': q['quality']}
               for i, q in enumerate(queries_raw, 1)]
    assert len(queries) == 597, len(queries)
    ids = {r['id'] for r in corpus}
    assert len(ids) == len(corpus)
    assert all(set(q['qrels']).issubset(ids) for q in queries)
    save(out / 'corpus.json', corpus)
    save(out / 'queries.json', queries)
    save(out / 'prepared.json', {'dataset': 'LitSearch', 'corpus': len(corpus), 'queries': len(queries),
         'representation': 'title: <title>. abstract: <abstract>',
         'labels': 'Original target-paper IDs, not exhaustive graded relevance.',
         'source': 'https://huggingface.co/datasets/princeton-nlp/LitSearch'})
    print('PREPARED', read(out / 'prepared.json'), flush=True)


def fuse(a, b, k=60):
    scores = {}
    for ranking in [a, b]:
        assert len(ranking) == len(set(ranking))
        for i, docid in enumerate(ranking, 1):
            scores[docid] = scores.get(docid, 0) + 1/(k+i)
    return sorted(scores, key=lambda d: (-scores[d], d))[:100]


def metrics(ranking, qrels, cutoff, threshold=1):
    top = ranking[:cutoff]
    relevant = {d for d, g in qrels.items() if g >= threshold}
    hits = [i for i, d in enumerate(top, 1) if d in relevant]
    gains = [(2**max(0, qrels.get(d, 0))-1)/math.log2(i+1) for i, d in enumerate(top, 1)]
    ideal = sum((2**g-1)/math.log2(i+1) for i, g in enumerate(
        sorted([max(0, v) for v in qrels.values()], reverse=True)[:cutoff], 1))
    linear_dcg = sum(max(0,qrels.get(d,0))/math.log2(i+1) for i,d in enumerate(top,1))
    linear_ideal = sum(g/math.log2(i+1) for i,g in enumerate(
        sorted([max(0,v) for v in qrels.values()],reverse=True)[:cutoff],1))
    return {f'MRR@{cutoff}': 1/hits[0] if hits else 0.,
            f'Hit@{cutoff}': float(bool(hits)),
            f'Recall@{cutoff}': len(hits)/len(relevant) if relevant else 0.,
            f'nDCG@{cutoff}': sum(gains)/ideal if ideal else 0.,
            f'nDCG_linear@{cutoff}': linear_dcg/linear_ideal if linear_ideal else 0.}


def evaluate(out):
    import numpy as np
    queries = read(out / 'queries.json')
    result = {q['id']: read(out / 'runs' / (q['id']+'.json')) for q in queries}
    rows = []
    for q in queries:
        for method in METHODS:
            ranking = result[q['id']]['rankings'][method]
            row = {'query_id': q['id'], 'group': q['group'], 'method': method,
                   'known_relevant': sum(g >= 1 for g in q['qrels'].values())}
            for cutoff in [5, 10, 20, 100]:
                row.update(metrics(ranking, q['qrels'], cutoff))
                if out.name == 'trec_rag_2024':
                    for key,value in metrics(ranking,q['qrels'],cutoff,threshold=2).items():
                        if key.startswith(('MRR','Hit','Recall')):row['rel2_'+key]=value
            row['judged_at_10'] = sum(d in q['qrels'] for d in ranking[:10])/10
            if q.get('umbrela_qrels') is not None:
                for key, value in metrics(ranking, q['umbrela_qrels'], 10).items():
                    row['umbrela_'+key] = value
            rows.append(row)
    # Keep consistent columns even if optional fields are absent on some queries.
    fields = sorted(set().union(*(r.keys() for r in rows)))
    csvwrite(out / 'metrics_per_query.csv', [{k:r.get(k,'') for k in fields} for r in rows])
    groups = {'all': queries}
    for group in sorted({q['group'] for q in queries}):
        groups[group] = [q for q in queries if q['group'] == group]
    if any(q['group'].startswith('manual_') for q in queries):
        groups['author_written'] = [q for q in queries if q['group'].startswith('manual_')]
    summary, paired = [], []
    for group, qs in groups.items():
        qids = {q['id'] for q in qs}
        selected = [r for r in rows if r['query_id'] in qids]
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(qs), size=(10000, len(qs)))
        for method in METHODS:
            values = [r for r in selected if r['method'] == method]
            for metric in [k for k in values[0] if '@' in k and not k.startswith('umbrela_')]:
                a = np.array([r[metric] for r in values])
                lo, hi = np.quantile(a[draws].mean(axis=1), [.025,.975])
                summary.append({'group':group,'method':method,'metric':metric,'queries':len(qs),
                                'mean':float(a.mean()),'ci_low':float(lo),'ci_high':float(hi)})
        for a, b in [('hybrid_rrf','bm25'),('hybrid_rrf','dense'),
                     ('hybrid_rerank_rrf','hybrid_rrf'),
                     ('hybrid_rerank_rrf','bm25'),('hybrid_rerank_rrf','dense')]:
            for metric in ['MRR@10','Hit@10','Recall@10','nDCG@10']:
                diff = np.array([next(r[metric] for r in selected if r['query_id']==q['id'] and r['method']==a)
                       -next(r[metric] for r in selected if r['query_id']==q['id'] and r['method']==b) for q in qs])
                lo,hi=np.quantile(diff[draws].mean(axis=1),[.025,.975])
                paired.append({'group':group,'comparison':a+' minus '+b,'metric':metric,
                               'mean_difference':float(diff.mean()),'ci_low':float(lo),'ci_high':float(hi)})
    csvwrite(out / 'metrics_summary.csv', summary)
    csvwrite(out / 'paired_differences.csv', paired)
    timing=[]
    for component in next(iter(result.values()))['timing_ms']:
        a=np.array([r['timing_ms'][component] for r in result.values()])
        timing.append({'component':component,'median_ms':float(np.median(a)),
                       'p95_ms':float(np.quantile(a,.95)),'mean_ms':float(a.mean())})
    csvwrite(out / 'latency_summary.csv',timing)
    export=out/'trec_format';export.mkdir(exist_ok=True)
    with (export/'qrels_primary.txt').open('w',encoding='utf-8') as f:
        for q in queries:
            for docid,grade in sorted(q['qrels'].items()):
                f.write(f"{q['id']} 0 {docid} {grade}\n")
    for method in METHODS:
        with (export/(method+'.run')).open('w',encoding='utf-8') as f:
            for q in queries:
                for rank,docid in enumerate(result[q['id']]['rankings'][method],1):
                    f.write(f"{q['id']} Q0 {docid} {rank} {1/rank:.10f} {method}\n")
    # Preserve exact source used for postprocessing alongside the results.
    (out/'scoring_source.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    report=['# Retrieval results', '', 'All configurations use identical corpus and queries.',
            'Reranking is applied to the dense top 100 before RRF (k=60).',
            'CIs: 10,000 paired query bootstrap resamples; no multiplicity adjustment.',
            'Unjudged results receive zero gain for conventional scoring; this is not a verified negative label.',
            'Recall measures recovery of known qrels, not exhaustive relevance.', '']
    for group in groups:
        report += [f'## {group}', '', '| Method | MRR@10 | Recall@10 | Hit@10 | nDCG@10 |',
                   '|---|---:|---:|---:|---:|']
        for method in METHODS:
            vals=[next(r['mean'] for r in summary if r['group']==group and r['method']==method and r['metric']==m)
                  for m in ['MRR@10','Recall@10','Hit@10','nDCG@10']]
            report.append('| '+method+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
        report.append('')
    (out/'RESULTS.md').write_text('\n'.join(report),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,metric in zip(axes,['Recall@10','nDCG@10']):
            vals=[next(r for r in summary if r['group']=='all' and r['method']==m and r['metric']==metric) for m in METHODS]
            means=np.array([r['mean'] for r in vals])
            ax.bar(range(4),means,color=['#8294a4','#536d82','#247a72','#d18b37'])
            ax.errorbar(range(4),means,yerr=[means-np.array([r['ci_low'] for r in vals]),np.array([r['ci_high'] for r in vals])-means],fmt='none',ecolor='#222222',capsize=4)
            ax.set_xticks(range(4),['BM25','Dense','RRF','RRF + rerank'],rotation=15)
            ax.set_ylabel(metric);ax.set_ylim(0,1);ax.spines[['top','right']].set_visible(False)
        fig.suptitle(out.name+' — query bootstrap 95% CIs')
        fig.tight_layout();fig.savefig(out/'retrieval_comparison.png',dpi=180)
        fig.savefig(out/'retrieval_comparison.pdf');plt.close(fig)
    except ImportError:
        print('matplotlib not available; CSV and Markdown reports retained',flush=True)
    save(out/'completed.json',{'queries':len(queries),'methods':METHODS,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


def run(out, dataset):
    import numpy as np
    import torch
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from flashrank import Ranker, RerankRequest
    import onnxruntime as ort
    corpus, queries = read(out/'corpus.json'), read(out/'queries.json')
    ids = [r['id'] for r in corpus]
    texts = [('title: '+r['title']+'. abstract: '+r['text']) if dataset=='litsearch'
             else (r['title']+'\n'+r['text']) for r in corpus]
    logical = hashlib.sha256(json.dumps(corpus,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    cache=Path.home()/'.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots'
    model_snapshot=sorted(cache.iterdir())[-1]
    model=SentenceTransformer(str(model_snapshot),device='cuda',local_files_only=True)
    model.max_seq_length=256
    client=chromadb.PersistentClient(path=str(out/'index'),settings=Settings(anonymized_telemetry=False))
    col=client.get_or_create_collection('public_retrieval',configuration={'hnsw':{
        'space':'cosine','ef_construction':100,'ef_search':100,'max_neighbors':16}})
    vectors_path=out/'embeddings.npy'
    build_start=time.perf_counter()
    if not vectors_path.exists():
        # Batches are independently resumable, avoiding lost work after interruption.
        parts=[]
        for offset in range(0,len(texts),2048):
            part=out/'embedding_parts'/f'{offset:07d}.npy'
            part.parent.mkdir(exist_ok=True)
            if not part.exists():
                vec=model.encode(texts[offset:offset+2048],batch_size=32,normalize_embeddings=True,show_progress_bar=False)
                np.save(part,vec)
            parts.append(np.load(part))
            print('EMBED',dataset,min(offset+2048,len(texts)),len(texts),flush=True)
        np.save(vectors_path,np.concatenate(parts))
    vectors=np.load(vectors_path,mmap_mode='r')
    if col.count()!=len(ids):
        for offset in range(0,len(ids),256):
            col.upsert(ids=ids[offset:offset+256],embeddings=vectors[offset:offset+256].tolist())
        assert col.count()==len(ids)
    print('INDEX READY',dataset,len(ids),round(time.perf_counter()-build_start,2),flush=True)
    bm_start=time.perf_counter()
    lexical=BM25Okapi([re.findall(r'\w+', (r['title']+' '+r['text']).lower()) for r in corpus])
    bm_build=time.perf_counter()-bm_start
    rankcache=ROOT/'src/src/models_cache'
    reranker=Ranker(model_name='ms-marco-MiniLM-L-12-v2',cache_dir=str(rankcache),max_length=512)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    reranker.session=ort.InferenceSession(str(rankcache/'ms-marco-MiniLM-L-12-v2/flashrank-MiniLM-L-12-v2_Q.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
    byid={r['id']:i for i,r in enumerate(corpus)}
    manifest={'dataset':dataset,'seed':SEED,'corpus':len(ids),'queries':len(queries),
              'corpus_sha256':logical,'model':'all-MiniLM-L6-v2','model_snapshot':model_snapshot.name,
              'embedding_max_length':256,'normalize_embeddings':True,'candidates':100,'rrf_k':60,
              'dense_similarity_floor':0.1,'reranker':'ms-marco-MiniLM-L-12-v2','reranker_max_length':512,
              'reranker_microbatch':8,'onnx_threads':[4,1],'rerank_placement':'dense branch before fusion',
              'hnsw':{'space':'cosine','ef_construction':100,'ef_search':100,'max_neighbors':16},
              'build_or_resume_seconds':time.perf_counter()-build_start,'bm25_build_seconds':bm_build,
              'gpu':torch.cuda.get_device_name(0),'platform':platform.platform(),
              'versions':{p:importlib.metadata.version(p) for p in ['torch','chromadb','sentence-transformers','flashrank','onnxruntime','rank-bm25']}}
    save(out/'run_manifest.json',manifest)
    def one(q):
        t=time.perf_counter();scores=lexical.get_scores(re.findall(r'\w+',q.lower()))
        top=np.argsort(-scores,kind='stable')[:100];lex=[ids[i] for i in top]
        bm_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter();v=model.encode([q],normalize_embeddings=True,show_progress_bar=False)
        torch.cuda.synchronize();embed_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter();found=col.query(query_embeddings=v.tolist(),n_results=100,include=['distances'])
        dense=[d for d,dist in zip(found['ids'][0],found['distances'][0]) if 1-float(dist)>=0.1]
        vector_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter();hybrid=fuse(lex,dense);fusion_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter();passages=[{'id':d,'text':texts[byid[d]]} for d in dense];rr=[]
        for offset in range(0,len(passages),8):
            rr.extend(reranker.rerank(RerankRequest(query=q,passages=passages[offset:offset+8])))
        rr.sort(key=lambda r:(-float(r['score']),r['id']))
        rerank_ms=(time.perf_counter()-t)*1000
        t=time.perf_counter();hybridrr=fuse(lex,[r['id'] for r in rr]);rf_ms=(time.perf_counter()-t)*1000
        return {'rankings':dict(zip(METHODS,[lex,dense,hybrid,hybridrr])),
                'timing_ms':{'bm25':bm_ms,'embedding':embed_ms,'vector':vector_ms,'fusion':fusion_ms,
                             'reranker':rerank_ms,'hybrid_rrf':bm_ms+embed_ms+vector_ms+fusion_ms,
                             'hybrid_rerank_rrf':bm_ms+embed_ms+vector_ms+rerank_ms+rf_ms}}
    print('WARMUP 3 queries',dataset,flush=True)
    for q in queries[:3]:one(q['query'])
    for i,q in enumerate(queries,1):
        path=out/'runs'/(q['id']+'.json')
        if path.exists():continue
        record=one(q['query']);record.update({'query_id':q['id'],'query':q['query']})
        save(path,record)
        print('QUERY',dataset,i,'/',len(queries),'rerank_ms',round(record['timing_ms']['reranker']),flush=True)
    evaluate(out)
    print('COMPLETE',dataset,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['litsearch','trec_rag_2024'],required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    out=BASE/args.dataset;out.mkdir(parents=True,exist_ok=True)
    if args.report_only:evaluate(out);return
    if args.dataset=='litsearch':prepare_litsearch(out)
    elif not (out/'prepared.json').exists():raise RuntimeError('Run prepare_trec.py first')
    if not args.prepare_only:run(out,args.dataset)


if __name__=='__main__':main()
