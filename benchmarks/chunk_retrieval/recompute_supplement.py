"""API-free analysis of the text-free revision supplement.
Also copied to the supplement root as REPRODUCE.py for standalone use.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

def main():
    default=Path(__file__).resolve().parent
    if not (default/'results').is_dir():default=default.parents[1]/'Report/supplement/current_evaluation'
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,default=default);args=parser.parse_args()
    root=args.bundle
    def rows(p):
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    ratings=rows(root/'results/human_generation_statistics_20261027/ratings_anonymized.csv')
    human_metrics=['Relevance','Accuracy','Completeness','Evidence / Citation','Clarity']
    a=np.full((18,12,5),np.nan)
    for row in ratings:
        q=int(row['query_id'][2:])-1;r=int(row['reviewer_id'][1:])-1
        assert np.isnan(a[q,r]).all();a[q,r]=[int(row[m]) for m in human_metrics]
    assert np.isfinite(a).all() and ((a>=1)&(a<=5)).all()
    rng=np.random.default_rng(20261027);draws=[]
    for _ in range(10000):
        q=rng.integers(0,18,18);r=rng.integers(0,12,12)
        draws.append(a[q][:,r,:].mean(axis=(0,1)))
    ci=np.quantile(draws,[.025,.975],axis=0)
    human={m:{'mean':float(a[:,:,i].mean()),'sample_sd':float(a[:,:,i].std(ddof=1)),
              'crossed_ci95':ci[:,i].tolist()} for i,m in enumerate(human_metrics)}
    expected=json.loads((root/'results/human_generation_statistics_20261027/statistics.json').read_text(encoding='utf8'))
    for e in expected['metrics']:
        s=human[e['metric']];assert np.allclose(s['crossed_ci95'],e['crossed_ci95'],rtol=0,atol=1e-12)
        assert abs(s['mean']-e['mean'])<1e-12 and abs(s['sample_sd']-e['sample_sd'])<1e-12
    out={'human_statistics':human,
         'verification':'All human means/SDs/crossed CIs match supplied results. No APIs, texts or private XLSX required.'}
    (root/'RECOMPUTED.json').write_text(json.dumps(out,indent=2),encoding='utf8')
    print(out['verification'])

if __name__=='__main__':main()
