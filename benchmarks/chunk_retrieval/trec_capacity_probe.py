"""Bounded capacity probe on real TREC RAG segments; no production writes/API calls."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
OUT = ROOT / 'output/benchmarks/trec_capacity_probe_20261029'
OUT.mkdir(parents=True, exist_ok=True)
URL = 'https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2.1_doc_segmented.tar'
SAMPLE = OUT / 'segments.jsonl'
LIMIT = 50000

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2), encoding='utf-8')

def main():
    if not SAMPLE.exists():
        start = time.perf_counter()
        count = 0
        with urllib.request.urlopen(URL, timeout=60) as response:
            with tarfile.open(fileobj=response, mode='r|') as archive:
                with SAMPLE.open('w', encoding='utf-8') as target:
                    for member in archive:
                        if not member.isfile() or not member.name.endswith('.gz'):
                            continue
                        print('STREAM', member.name, flush=True)
                        with gzip.GzipFile(fileobj=archive.extractfile(member)) as stream:
                            for line in stream:
                                row = json.loads(line)
                                if not row.get('segment'):
                                    continue
                                target.write(json.dumps(row, ensure_ascii=False) + '\n')
                                count += 1
                                if count >= LIMIT:
                                    break
                        if count >= LIMIT:
                            break
        save('download.json', {'url': URL, 'sampling': 'first nonempty segments in first archive shard; not random/representative', 'segments': count, 'seconds': time.perf_counter()-start, 'bytes': SAMPLE.stat().st_size})
    rows = [json.loads(line) for line in SAMPLE.open(encoding='utf-8')]
    if len(rows) != LIMIT:
        raise RuntimeError('Incomplete sample; inspect before retry')
    import chromadb
    from chromadb.config import Settings
    import torch
    from sentence_transformers import SentenceTransformer
    from benchmarks.resource_memory import Memory
    from rank_bm25 import BM25Okapi
    cache = Path.home()/'.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots'
    start = time.perf_counter()
    model = SentenceTransformer(str(sorted(cache.iterdir())[-1]), device='cuda' if torch.cuda.is_available() else 'cpu', local_files_only=True)
    model.max_seq_length = 256
    model.encode(['warmup'], show_progress_bar=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    load_s = time.perf_counter()-start
    target = OUT/'index'
    if target.exists():
        raise RuntimeError('Refusing to overwrite probe index')
    client = chromadb.PersistentClient(path=str(target), settings=Settings(anonymized_telemetry=False))
    col = client.create_collection('trec_probe', configuration={'hnsw': {'space':'cosine','ef_construction':100,'ef_search':100,'max_neighbors':16}})
    checkpoints = []
    embed_s = insert_s = 0
    started = time.perf_counter()
    with Memory() as memory:
        for offset in range(0,len(rows),128):
            batch = rows[offset:offset+128]
            docs = [r['segment'] for r in batch]
            start = time.perf_counter()
            vectors = model.encode(docs,batch_size=32,show_progress_bar=False).tolist()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            embed_s += time.perf_counter()-start
            start = time.perf_counter()
            col.add(ids=[r['docid'] for r in batch],documents=docs,metadatas=[{'title':str(r.get('title','')),'source_url':str(r.get('url',''))} for r in batch],embeddings=vectors)
            insert_s += time.perf_counter()-start
            n = offset+len(batch)
            if n % 8192 == 0 or n == len(rows):
                point={'segments':n,'embedding_s':embed_s,'insert_s':insert_s,'elapsed_s':time.perf_counter()-started}
                checkpoints.append(point)
                print('BUILD',json.dumps(point),flush=True)
                save('checkpoints.json',checkpoints)
    build_memory = memory.fields()
    build_wall_s = time.perf_counter()-started
    with Memory() as bm_memory:
        start = time.perf_counter()
        tokens=[re.findall(r'\w+', (r.get('title','')+' '+r['segment']).lower()) for r in rows]
        bm=BM25Okapi(tokens)
        bm_build=time.perf_counter()-start
        timings=[]
        for query in ['what causes climate change','how does machine learning work','effects of exercise on blood pressure']:
            start=time.perf_counter()
            scores=bm.get_scores(re.findall(r'\w+',query.lower()))
            import numpy as np
            np.argsort(scores)[::-1][:100]
            timings.append(time.perf_counter()-start)
    disk=sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
    result={'segments':len(rows),'sample_sha256':hashlib.sha256(SAMPLE.read_bytes()).hexdigest(),'sampling':'first 50k segments; single shard, not randomized; timing/capacity only, not effectiveness benchmark','model_load_and_warmup_s':load_s,'embedding_s':embed_s,'insert_s':insert_s,'build_wall_s':build_wall_s,'checkpoints':checkpoints,'build_memory':build_memory,'bm25_memory':bm_memory.fields(),'bm25_build_s':bm_build,'bm25_query_s':timings,'disk_bytes_before_process_exit':disk,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'gpu_peak_allocated':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,'full_segments':113520750,'linear_embedding_hours':embed_s/len(rows)*113520750/3600,'linear_insert_hours':insert_s/len(rows)*113520750/3600,'linear_disk_bytes':disk/len(rows)*113520750,'limitations':'Full-index HNSW, RAM, disk I/O and thermal scaling not measured. No full-corpus completion guarantee. Download only partial archive; full download time not measured.'}
    save('results.json',result)
    print('RESULT',json.dumps(result),flush=True)

if __name__ == '__main__':
    main()
