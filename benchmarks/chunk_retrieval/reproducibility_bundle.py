"""Build a local text-free evaluation supplement; never upload it.
Full texts, passage-bearing prompts, credentials and reviewer timestamps excluded.
"""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Report/supplement/current_evaluation'

def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'benchmarks').rglob('*.py'))
    files += [ROOT/'data/src/eval/1_eval.py',ROOT/'requirements.txt']
    hashes={}
    for src in files:
        content=src.read_bytes()
        # Fail closed rather than publishing a source file containing a common key literal.
        assert not re.search(rb'AIza[0-9A-Za-z_-]{30,}|gh[pousr]_[0-9A-Za-z]{30,}',content),src.name
        relative=src.relative_to(ROOT)
        dest=OUT/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest)
        hashes[relative.as_posix()]=hashlib.sha256(content).hexdigest()
    write(OUT/'source_sha256.json',hashes)
    shutil.copy2(ROOT/'benchmarks/chunk_retrieval/recompute_supplement.py',OUT/'REPRODUCE.py')
    versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
    write(OUT/'installed_versions.json',versions)
    prompts=['# Exact source prompt templates\n\nExtracted without importing application modules. Expressions retain their runtime placeholders. Source line numbers refer to the accompanying snapshot.\n']
    for relative in ['src/agent/nodes/intent_classifier.py','src/agent/nodes/answer_generator.py',
                     'src/agent/nodes/result_evaluator.py','src/agent/nodes/external_searcher.py',
                     'src/web/core/rag.py','data/src/eval/1_eval.py']:
        src=(ROOT/relative).read_text(encoding='utf-8-sig');tree=ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and 'prompt' in t.id.lower() for t in node.targets):
                if isinstance(node.value,(ast.JoinedStr,ast.Constant)) and (isinstance(node.value,ast.JoinedStr) or isinstance(node.value.value,str)):
                    prompts.append(f'\n## {relative}:{node.lineno}\n\n```python\n{ast.get_source_segment(src,node)}\n```\n')
    (OUT/'PROMPT_TEMPLATES.md').write_text(''.join(prompts),encoding='utf8')
    subsets={
        'human_generation_statistics_20261027':['statistics.json','ratings_anonymized.csv'],
        'online_qa_cost_20261027':['manifest.json','queries.json','per_query.csv','llm_stage_summary.csv','local_stage_summary.csv','VALIDATION.json'],
        'grounding_audit_20261027':['protocol.json','summary.json','human_validation.json']}
    for folder,names in subsets.items():
        for name in names:
            src=ROOT/'output/benchmarks'/folder/name
            if src.exists():
                dest=OUT/'results'/folder/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    write(OUT/'AVAILABILITY.json',{'public_source_repository':'https://github.com/Nhattk19/NLP-in-Industry',
        'local_revision_supplement':'prepared locally; not uploaded by this script',
        'hosted_demo':'not supplied','software_license':'not selected; public visibility is not a reuse license',
        'raw_corpus':'not redistributed; item-level permissions audit incomplete',
        'fulltext_reconstruction':'Follow source URLs and verify per-version licenses before download/processing. Source documents and extraction may change; exact corpus reconstruction is not guaranteed.',
        'excluded':['credentials','PDFs','chunk text','vector databases','passage-bearing raw API prompts','original reviewer timestamps'],
        'public_retrieval':'Public qrels, rankings, metrics and configuration manifests are retained separately in Report/supplement/public_retrieval_20261029.'})
    targets=[p for p in OUT.rglob('*') if p.is_file() and p.name!='PACKAGE_SHA256.json']
    write(OUT/'PACKAGE_SHA256.json',{p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in targets})
    zip_path=ROOT/'Report/current_evaluation.zip'
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(OUT))
    print('files',len(targets)+1,'zip_bytes',zip_path.stat().st_size)

if __name__=='__main__':main()
