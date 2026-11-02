"""Freeze TREC topics/judgments and prepare an explicitly reduced corpus.

No synthetic labels. Primary qrels are fully manual for 27 topics and UMBRELA
for 23 additional seeded topics. All available judged candidates for those
topics are retained, including grade 0. Random corpus distractors are unlabeled.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import time
from run import BASE, SEED, get, save, read

OUT=BASE/'trec_rag_2024'


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    sources={
        'topics.txt':'https://trec.nist.gov/data/rag/topics.rag24.test.txt',
        'conditions_qrels.txt':'https://trec.nist.gov/data/rag/2024-retrieval-conditions-qrels.txt',
        'umbrela_qrels.txt':'https://trec-rag.github.io/assets/txt/qrels.rag24.test-umbrela-all.txt'}
    for name,url in sources.items():
        path=OUT/name
        if not path.exists():path.write_text(get(url).text,encoding='utf-8')
    human=defaultdict(dict);manual_ids=set()
    for line in (OUT/'conditions_qrels.txt').read_text().splitlines():
        q,c,d,g=line.split()
        assert '#' in d, 'Expected segment IDs'
        if c=='1':
            manual_ids.add(q);human[q][d]=int(g)
    assert len(manual_ids)==27
    umbrella=defaultdict(dict)
    for line in (OUT/'umbrela_qrels.txt').read_text().splitlines():
        q,_,d,g=line.split();assert '#' in d;umbrella[q][d]=int(g)
    topics={}
    for line in (OUT/'topics.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        q,text=line.split('\t',1);topics[q]=text
    rng=random.Random(SEED)
    extra=sorted(rng.sample(sorted(set(umbrella)-manual_ids),23))
    queries=[]
    for q in sorted(manual_ids|set(extra)):
        queries.append({'id':q,'query':topics[q],
                        'group':'fully_manual_27' if q in manual_ids else 'umbrela_extra_23',
                        'qrels':human[q] if q in manual_ids else umbrella[q],
                        'umbrela_qrels':umbrella[q]})
    # Keep the union of every available assessment (human and UMBRELA), regardless of grade.
    needed=set()
    selected={q['id'] for q in queries}
    for q in selected:needed.update(umbrella[q]);needed.update(human[q])
    for line in (OUT/'conditions_qrels.txt').read_text().splitlines():
        q,_,d,_=line.split()
        if q in selected:needed.add(d)
    save(OUT/'queries.json',queries)
    save(OUT/'required_segment_ids.json',sorted(needed))
    save(OUT/'selection_manifest.json',{'seed':SEED,'queries':50,'fully_manual':sorted(manual_ids),
        'extra_umbrela':extra,'required_segments':len(needed),'minimum_corpus_size':64200,
        'sources':sources,'primary_labels':'27 fully manual topics + 23 UMBRELA topics; report separately',
        'corpus_selection':'union of all judged segments for selected topics, including grade 0, plus seeded random distractors',
        'limitation':'Reduced corpus derived from judgment pools; not official full-corpus evaluation. Unjudged distractors are not verified negatives.'})
    print('FROZEN',len(queries),'topics;',len(needed),'required segments',flush=True)
    return needed


def collect(allowed_missing=()):
    import requests
    required=freeze()
    source_manifest=OUT/'corpus_mirror_manifest.json'
    api=read(source_manifest) if source_manifest.exists() else get('https://huggingface.co/api/datasets/drexalt/msmarco-2.1-segmented').json()
    save(OUT/'corpus_mirror_manifest.json',api)
    revision=api['sha']
    files=sorted(x['rfilename'] for x in api['siblings'] if x['rfilename'].endswith('.json.gz'))
    assert len(files)==60, len(files)
    cache=OUT/'shard_samples';cache.mkdir(exist_ok=True)
    wanted={d.encode() for d in required}
    def shard(index,name):
        checkpoint=cache/(Path(name).name+'.json')
        if checkpoint.exists():return read(checkpoint)
        url=f'https://huggingface.co/datasets/drexalt/msmarco-2.1-segmented/resolve/{revision}/{name}'
        for attempt in range(4):
            found=[];distractors=[];count=0;rng=random.Random(SEED+index)
            started=time.perf_counter()
            try:
                # Avoid stale CDN redirect/cache routes on long-lived large-file transfers.
                with requests.get(url,params={'download':'true','transfer':f'{time.time_ns()}'},
                                  stream=True,timeout=(20,40)) as response:
                    response.raise_for_status()
                    with gzip.GzipFile(fileobj=response.raw) as stream:
                        for line in stream:
                            # Source JSON begins with docid. Avoid decoding >113M unused records.
                            docid=line.split(b'"',4)[3]
                            keep=docid in wanted
                            sampled=rng.random()<0.0012
                            if keep or sampled:
                                row=json.loads(line)
                                record={'id':row['docid'],'title':row.get('title',''),
                                        'text':row['segment'],'url':row.get('url','')}
                                assert record['id'].encode()==docid
                                if keep:found.append(record)
                                elif sampled:distractors.append(record)
                            count+=1
                            if count%10000==0 and time.perf_counter()-started>240:
                                raise TimeoutError('Shard transfer exceeded four-minute deadline; retry another CDN route')
                            if count%500000==0:
                                print('STREAM',index,'records',count,'required',len(found),flush=True)
                result={'shard':name,'count':count,'required':found,'random_candidates':distractors,
                        'seconds':time.perf_counter()-started,'source':url}
                save(checkpoint,result)
                print('SHARD DONE',index,'records',count,'required',len(found),'seconds',round(result['seconds']),flush=True)
                return result
            except Exception as exc:
                print('SHARD RETRY',index,attempt,type(exc).__name__,flush=True)
                if attempt==3:raise
                time.sleep(2**attempt)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(shard,i,name) for i,name in enumerate(files)]
        for future in as_completed(futures):results.append(future.result())
    corpus={r['id']:r for result in results for r in result['required']}
    missing=required-set(corpus)
    save(OUT/'missing_required_ids.json',sorted(missing))
    assert missing.issubset(set(allowed_missing)), f'{len(missing)} required passages missing without explicit exception; refusing partial evaluation'
    queries=read(OUT/'queries.json')
    for docid in missing:
        observed=[labels[docid] for q in queries for labels in [q['qrels'],q['umbrela_qrels']] if docid in labels]
        assert observed and all(g==0 for g in observed), 'Missing relevant evidence cannot use the zero-grade exception'
    if missing:
        save(OUT/'approved_missing_id_exception.json',{
            'approval':'Explicit user approval in conversation on 2026-10-29',
            'missing_ids':sorted(missing),'all_labels_zero_in_selected_topics':True,
            'qrels_unchanged':True,'action':'Omit unavailable ID from corpus requirement; retain all original qrels; add seeded distractor to maintain corpus size.'})
    candidates={r['id']:r for result in results for r in result['random_candidates'] if r['id'] not in corpus}
    rng=random.Random(SEED)
    extra=rng.sample(sorted(candidates),max(0,64200-len(corpus)))
    corpus.update({d:candidates[d] for d in extra})
    save(OUT/'corpus.json',[corpus[d] for d in sorted(corpus)])
    manifest=read(OUT/'selection_manifest.json')
    manifest.update({'corpus':len(corpus),'random_distractors':len(extra),'scanned_records':sum(r['count'] for r in results),
                     'found_assessed_segments':len(required)-len(missing),'approved_missing_ids':sorted(missing),
                     'mirror_revision':revision,'mirror':'drexalt/msmarco-2.1-segmented',
                     'random_sampling':'Independent Bernoulli p=.0012 per record with per-shard seeds, then fixed-size seeded sample of non-required candidates; independent of relevance grades and system scores.',
                     'completed_shards':len(results)})
    save(OUT/'prepared.json',manifest)
    print('PREPARED TREC',len(corpus),'segments',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collect',action='store_true')
    parser.add_argument('--allow-missing-zero-id',action='append',default=[])
    args=parser.parse_args()
    if args.collect:collect(args.allow_missing_zero_id)
    else:freeze()
