"""Read the original Google Forms XLSX export without editing it.
Crossed bootstrap resamples questions and reviewers independently, retaining
the complete submatrix and all five metrics in each draw. No i.i.d. rating CI.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/benchmarks/human_generation_statistics_20261027'
SOURCE=OUT/'human_ratings_source.xlsx'
METRICS=['Relevance','Accuracy','Completeness','Evidence / Citation','Clarity']
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def load():
    if not SOURCE.exists():
        with (OUT/'ratings_anonymized.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        a=np.full((18,12,5),np.nan)
        for row in rows:
            q=int(row['query_id'][2:])-1;r=int(row['reviewer_id'][1:])-1
            assert np.isnan(a[q,r]).all(),'Duplicate query/reviewer pair'
            a[q,r]=[float(row[m]) for m in METRICS]
        assert a.shape==(18,12,5) and np.isfinite(a).all()
        assert np.all((a>=1)&(a<=5)&(a==a.astype(int)))
        return a,{}
    with zipfile.ZipFile(SOURCE) as z:
        ss=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',NS)] if 'xl/sharedStrings.xml' in z.namelist() else []
        rows=[]
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('m:sheetData/m:row',NS):
            cells={}
            for c in row.findall('m:c',NS):
                val=c.find('m:v',NS)
                value=val.text if val is not None else ''.join(c.find('m:is',NS).itertext()) if c.find('m:is',NS) is not None else ''
                if c.get('t')=='s': value=ss[int(value)]
                cells[re.sub(r'\d+','',c.get('r'))]=value
            rows.append(cells)
    def colnum(s):
        n=0
        for ch in s:n=n*26+ord(ch)-64
        return n
    headers=sorted(((k,v) for k,v in rows[0].items() if 'Question ' in v),key=lambda kv:colnum(kv[0]))
    assert len(headers)==90
    mapping=[]
    qa={}
    for key,label in headers:
        q=int(re.search(r'Question (\d+):',label).group(1))
        metric=next(m for m in METRICS if '['+m+' (' in label)
        mapping.append((key,q,METRICS.index(metric)))
        # Question 2 has a stray '3' before its label in the original form.
        qa[q]=label[label.index('Question '):label.rfind(' ['+metric+' (')].strip()
    scores=[]
    for row in rows[1:]:
        vals=[row.get(k,'') for k,q,m in mapping]
        if not any(v!='' for v in vals):continue
        assert all(v!='' for v in vals),'Incomplete reviewer; do not silently drop'
        a=np.empty((18,5))
        for (key,q,m),v in zip(mapping,vals):
            a[q-1,m]=float(v)
        assert np.all((a>=1)&(a<=5)&(a==a.astype(int)))
        scores.append(a)
    a=np.stack(scores,axis=1) # question, reviewer, metric
    assert a.shape==(18,12,5),a.shape
    return a,qa

def main():
    a,qa=load()
    rng=np.random.default_rng(20261027)
    cross=[];query=[]
    for _ in range(10000):
        q=rng.integers(0,18,18);r=rng.integers(0,12,12)
        cross.append(a[q][:,r,:].mean(axis=(0,1)))
        query.append(a[q].mean(axis=(0,1)))
    cross=np.asarray(cross);query=np.asarray(query)
    stats=[]
    original=[(4,1.065),(4.009,1.145),(3.361,1.220),(4.028,1.002),(4.139,.997)]
    for i,m in enumerate(METRICS):
        mean=float(a[:,:,i].mean());sd=float(a[:,:,i].std(ddof=1))
        stats.append({'metric':m,'n_ratings':216,'mean':mean,'sample_sd':sd,
            'crossed_ci95':np.quantile(cross[:,i],[.025,.975]).tolist(),
            'question_cluster_ci95':np.quantile(query[:,i],[.025,.975]).tolist(),
            'original_mean':original[i][0],'original_sd':original[i][1],
            'matches_original_rounded':round(mean,3)==original[i][0] and round(sd,3)==original[i][1]})
    with (OUT/'ratings_anonymized.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['query_id','reviewer_id',*METRICS])
        for q in range(18):
            for r in range(12):w.writerow([f'hq{q+1:02d}',f'r{r+1:02d}',*a[q,r].astype(int)])
    if qa:(OUT/'questions_answers.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf8')
    result={'source_url':'https://docs.google.com/spreadsheets/d/1SC_QiPMAWHAwoU4cEUZQigzGKjiIWosmIpZLqpvXFQ4/edit?gid=1165441521',
        'source_sha256':hashlib.sha256((SOURCE if SOURCE.exists() else OUT/'ratings_anonymized.csv').read_bytes()).hexdigest(),
        'shape':list(a.shape),'seed':20261027,'bootstrap_draws':10000,
        'main_ci':'percentile crossed bootstrap, independently resampling 18 questions and 12 reviewers; same draws across metrics',
        'sensitivity_ci':'question cluster bootstrap conditional on the observed panel',
        'limitations':'Approximate marginal intervals; selected question set and observed reviewer panel; not simultaneous coverage or a superiority test. No comparator generation system was rated.',
        'metrics':stats}
    (OUT/'statistics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for s in stats:print(json.dumps(s))

if __name__=='__main__':main()
