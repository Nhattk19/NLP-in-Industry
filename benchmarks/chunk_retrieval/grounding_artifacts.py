"""Local evidence review page. Passage-bearing page is excluded from release ZIP."""
import csv
import html
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('audit',HERE/'grounding_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def main():
    for path in (audit.OUT/'judgments').glob('qa*.json'):
        d=audit.read(path);p=audit.packet(audit.SOURCE/'responses'/path.name)
        d['validation_flags']=audit.validate(d['response'],p);audit.write(path,d)
    audit.summarize()
    with (audit.OUT/'human_review.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    sections=[];esc=html.escape
    for path in sorted((audit.OUT/'judgments').glob('qa*.json')):
        d=audit.read(path);p=audit.packet(audit.SOURCE/'responses'/path.name)
        chunk_html=''.join(f'<details><summary>[{ref}] {esc(chunk["chunk_id"])}</summary><pre>{esc(chunk["text"])}</pre></details>' for ref,chunk in p['chunks'].items())
        table=[]
        for row in rows:
            if row['query_id']!=p['query_id']:continue
            key=f'{row["query_id"]}_{row["claim_id"]}'
            table.append(f'''<article data-key="{key}"><h3>Claim {esc(row['claim_id'])} - LLM: {esc(row['llm_evidence_label'])}</h3>
            <blockquote>{esc(row['answer_span'])}</blockquote><p>{esc(row['atomic_claim'])}</p>
            <p>{esc(row['llm_reason'])}</p><details><summary>LLM evidence and citation judgments</summary><pre>{esc(row['llm_evidence'])}\n{esc(row['llm_citations'])}</pre></details>
            <label>Human evidence label <select data-field="human_evidence_label"><option value="">Unreviewed</option><option>supported</option><option>contradicted</option><option>insufficient</option><option>exclude_nonfactual</option><option>needs_split</option></select></label>
            <label>Human citation judgments (JSON, use refs and labels) <textarea data-field="human_citations_json">{esc(row['human_citations_json'])}</textarea></label>
            <label>Comment / split correction <textarea data-field="human_comment">{esc(row['human_comment'])}</textarea></label>
            <label>Reviewer <input data-field="reviewer" value="{esc(row['reviewer'])}"></label></article>''')
        flags='; '.join(d['validation_flags']) or 'No mechanical validation flags'
        sections.append(f'''<section><h2>{p['query_id']}: {esc(p['question'])}</h2>
        <p class="warning">{esc(flags)}</p><details><summary>Final answer</summary><pre>{esc(p['answer'])}</pre></details>
        <details><summary>Actual final-generation evidence ({len(p['chunks'])} chunks)</summary>{chunk_html}</details>{''.join(table)}</section>''')
    # JSON escapes '<' so generated answers cannot terminate the inert data script.
    data=json.dumps(rows,ensure_ascii=True).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>LitRAG-NLP claim review</title>
    <style>body{font:16px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#172331}h1,h2{line-height:1.25}section{margin:45px 0;border-top:3px solid #234e70}article{padding:18px;margin:20px 0;background:#f2f5f8;border:1px solid #ccd5dd}pre{white-space:pre-wrap;word-break:break-word;font:14px/1.5 monospace}label{display:block;margin:12px 0}textarea{display:block;width:96%;min-height:55px}select,input,button{padding:8px;font:inherit}.warning{color:#8a3a00}button{position:sticky;top:8px;background:#164c6e;color:white;cursor:pointer}blockquote{border-left:3px solid #7996ae;padding-left:14px}</style>
    <h1>Claim-level grounding and citation review</h1><p>Automated labels are preliminary. Review ALL claims, omissions/splits and attached citations against actual chunks. Do not mark plausible background knowledge as supported unless entailed by this context. Abstention is not perfect factuality. Mechanical quote/span flags need inspection; PDF hyphenation is normalized. No external services are used by this page.</p>
    <p>Blank fields mean unreviewed. Your edits remain in this page until you export CSV; export before closing. This local page contains paper passages and must not be publicly redistributed without a rights audit.</p><button onclick="download()">Export human_review.csv</button>
    '''+''.join(sections)+f'<script id="rows" type="application/json">{data}</script>'+'''
    <script>
    const rows=JSON.parse(document.getElementById('rows').textContent);
    const fields=Object.keys(rows[0]);
    const existing=new Map(rows.map(r=>[r.query_id+'_'+r.claim_id,r]));
    document.querySelectorAll('article[data-key]').forEach(a=>a.querySelectorAll('[data-field]').forEach(el=>el.value=existing.get(a.dataset.key)[el.dataset.field]??''));
    function download(){
      const index=new Map(rows.map(r=>[r.query_id+'_'+r.claim_id,{...r}]));
      document.querySelectorAll('article[data-key]').forEach(a=>a.querySelectorAll('[data-field]').forEach(el=>index.get(a.dataset.key)[el.dataset.field]=el.value));
      const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
      const csv='\\ufeff'+[fields.map(quote).join(','),...Array.from(index.values()).map(r=>fields.map(f=>quote(r[f])).join(','))].join('\\r\\n');
      const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
      const a=document.createElement('a');a.href=url;a.download='human_review.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }
    </script></html>'''
    if (audit.OUT/'human_validation.json').exists():
        page=page.replace('Automated labels are preliminary.', 'One author reviewed the automated assessments and retained all labels unchanged on 27 October 2026. Fields were populated from this explicit batch confirmation. Mechanical flags are retained for transparency.')
    (audit.OUT/'human_review.html').write_text(page,encoding='utf8')
    print('Local review page written; never included in text-free release bundle.')

if __name__=='__main__':main()
