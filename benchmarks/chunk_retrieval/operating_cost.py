"""Instrument real QA on a disposable corpus copy. No production source changes.

python -u benchmarks/chunk_retrieval/operating_cost.py --limit 5
python -u benchmarks/chunk_retrieval/operating_cost.py --limit 30
Completed pilot responses are retained, never purchased twice. Timings exclude
index preparation/reset/model warmup. API settings/prompts remain the app's.
"""
from __future__ import annotations
import argparse
import csv
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
OUT = ROOT / "output/benchmarks/online_qa_cost_20261027"
RATE_IN, RATE_OUT, RATE_CACHE = .30, 2.50, .03
PRICE_URL = "https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def queries():
    """Fixed before any formal QA run; 18 base + 6 follow-up + 6 coverage probes."""
    bases = [
        ("What is masked language modeling in BERT, and how does it differ from autoregressive language modeling?", "Explain how its pretraining objective affects its use for text generation."),
        ("Compare statistical and neural machine translation regarding data requirements and error patterns.", "How do those differences affect translation for low-resource languages?"),
        ("How does a CRF layer help a neural named entity recognition model?", "What limitations does that approach have for nested entities?"),
        ("Compare extractive and abstractive text summarization, including factuality limitations.", "Which approach is more likely to introduce unsupported information, and why?"),
        ("How are dense retrievers trained for open-domain question answering?", "How does negative sampling affect this training?"),
        ("What are the main approaches to parameter-efficient fine-tuning of language models?", "How does LoRA reduce the number of trainable parameters?"),
        ("What is dependency parsing and how is its quality evaluated?", None),
        ("Explain coreference resolution and the main challenges in evaluating it.", None),
        ("How do contextual word representations differ from static word embeddings?", None),
        ("What methods help cross-lingual transfer when labeled training data are scarce?", None),
        ("Compare lexical and dense retrieval for scientific literature search.", None),
        ("How do text classification models handle class imbalance?", None),
        ("What methods are used to detect hallucinations in generated text?", None),
        ("How do knowledge distillation and pruning differ for compressing language models?", None),
        ("Explain the role of dialogue state tracking in task-oriented dialogue systems.", None),
        ("Compare automatic metrics for machine translation and their limitations.", None),
        ("How do neural models perform relation extraction from text?", None),
        ("What techniques improve robustness against textual adversarial examples?", None),
    ]
    probes = [
        "Which NLP papers published in October 2026 evaluate retrieval-augmented generation for low-resource languages? Summarize reported datasets and results, or state if evidence is unavailable.",
        "What do 2026 papers report about test-time adaptation for named entity recognition under domain shift? Include specific empirical findings if available.",
        "Which recent 2026 studies compare citation verification methods for scientific question answering? Describe their evaluation protocols and limitations.",
        "What evidence do papers published in 2026 provide on multimodal retrieval for scientific figures and text? Distinguish measured results from proposals.",
        "How have 2026 NLP studies evaluated agentic literature search against human research workflows? Identify concrete benchmarks and failure modes.",
        "Which 2026 papers evaluate speech-language models on low-resource Vietnamese tasks, and what datasets and limitations do they report?",
    ]
    result = []
    def add(question, category, parent=None):
        qid = f"qa{len(result)+1:03d}"
        result.append({"query_id": qid, "question": question, "category": category, "parent": parent})
        return qid
    # Pilot deliberately covers a base, a natural follow-up, comparison, and missing-coverage risk.
    parent = add(bases[0][0], "standalone")
    add(bases[0][1], "follow_up", parent)
    add(bases[1][0], "standalone")
    add(probes[0], "recent_coverage_probe")
    add(bases[2][0], "standalone")
    parent_ids = {0: "qa001", 1: "qa003", 2: "qa005"}
    for i in range(3,18):
        parent_ids[i] = add(bases[i][0], "standalone")
    for i in range(1,6):
        add(bases[i][1], "follow_up", parent_ids[i])
    for question in probes[1:]:
        add(question, "recent_coverage_probe")
    assert len(result) == 30
    return result


def snapshot_source():
    import chromadb
    from chromadb.config import Settings
    c = chromadb.PersistentClient(path=str(ROOT / "data/chroma_store_fulltext"),
        settings=Settings(anonymized_telemetry=False)).get_collection("papers_fulltext")
    rows = []
    for offset in range(0,c.count(),2000):
        r=c.get(limit=2000,offset=offset,include=["documents","metadatas"])
        rows.extend(zip(r["ids"],r["documents"],r["metadatas"]))
    digest=hashlib.sha256()
    for cid,doc,meta in sorted(rows):
        digest.update((json.dumps({"id":cid,"document":doc,"metadata":meta or {}},
            sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n").encode())
    return {"chunks":len(rows),"document_metadata_sha256":digest.hexdigest()}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT / "responses").mkdir(exist_ok=True)
    (OUT / "calls").mkdir(exist_ok=True)
    if not (OUT / "queries.json").exists():
        write(OUT / "queries.json",queries())
    if not (OUT / "source_before.json").exists():
        write(OUT / "source_before.json", snapshot_source())
    if not (OUT / "index_private").exists():
        print("Preparing disposable full-text corpus copy...",flush=True)
        shutil.copytree(ROOT / "data/chroma_store_fulltext", OUT / "index_private")


class BudgetStop(BaseException):
    pass


class Trace:
    def __init__(self,budget):
        self.qid="warmup"; self.calls=[]; self.spans=[]; self.active=None
        self.total_cost=0.; self.budget=budget
        for f in (OUT / "calls").glob("*.json"):
            self.total_cost += read(f).get("estimated_usd",0.)
        for f in (OUT / "diagnostic_pilot_reference_cycle" / "calls").glob("*.json"):
            self.total_cost += read(f).get("estimated_usd",0.)

    @contextmanager
    def span(self,stage):
        start=time.perf_counter()
        try:
            yield
        finally:
            self.spans.append({"query_id":self.qid,"stage":stage,
                "elapsed_ms":(time.perf_counter()-start)*1000})

    def timed(self,stage,fn):
        def wrapped(*a,**kw):
            with self.span(stage):
                return fn(*a,**kw)
        return wrapped


class MeasuredLLM:
    def __init__(self,model,stage,trace):
        self.model=model; self.stage=stage; self.trace=trace
        # Bounded transport timeout; preserve configured retry count, thinking, prompts and sampling.
        self.model.timeout=60

    def __getattr__(self,name):
        return getattr(self.model,name)

    def invoke(self,prompt,*args,**kwargs):
        tr=self.trace
        if tr.total_cost >= tr.budget-.02:
            raise BudgetStop("Observed usage is approaching the experiment budget")
        stage=self.stage
        if stage=="answer_generation" and "previous answer was cut off" in str(prompt).lower():
            stage="answer_continuation"
        if stage=="answer_generation" and "outside of the NLP/ML/DL/AI paper domain" in str(prompt):
            stage="ood_answer"
        index=len(list((OUT / "calls").glob(f"{tr.qid}_*.json")))+1
        record={"query_id":tr.qid,"call_index":index,"stage":stage,
            "model":self.model.model,"temperature":self.model.temperature,
            "max_output_tokens":self.model.max_output_tokens,"http_attempts":[],
            "prompt":str(prompt)}
        tr.active=record
        start=time.perf_counter()
        try:
            response=self.model.invoke(prompt,*args,**kwargs)
            usage=dict(response.usage_metadata or {})
            if not usage or not {"input_tokens","output_tokens","total_tokens"} <= usage.keys():
                raise ValueError("Missing usage metadata; cannot claim measured token cost")
            inp=int(usage["input_tokens"]); out=int(usage["output_tokens"])
            cached=int((usage.get("input_token_details") or {}).get("cache_read",0))
            reasoning=int((usage.get("output_token_details") or {}).get("reasoning",0))
            assert inp+out==int(usage["total_tokens"])
            assert reasoning<=out and cached<=inp
            cost=((inp-cached)*RATE_IN+cached*RATE_CACHE+out*RATE_OUT)/1e6
            record.update({"status":"ok","usage":usage,"input_tokens":inp,"output_tokens":out,
                "reasoning_tokens":reasoning,"visible_output_tokens":out-reasoning,
                "cached_tokens":cached,"total_tokens":int(usage["total_tokens"]),
                "estimated_usd":cost,"response":response.content,
                "response_metadata":response.response_metadata})
            tr.total_cost+=cost
            return response
        except Exception as exc:
            # Never print SDK exception text: some transport exceptions can contain request details.
            record.update({"status":"error","error_type":type(exc).__name__,
                           "error_code":getattr(exc,"code",None),"estimated_usd":0.,
                           "usage_unavailable":True})
            raise
        finally:
            record["elapsed_ms"]=(time.perf_counter()-start)*1000
            write(OUT / "calls" / f"{tr.qid}_{index:02d}_{stage}.json",record)
            tr.calls.append(record);tr.active=None
            print(f"LLM {tr.qid} {stage} {record['status']} {record['elapsed_ms']/1000:.2f}s "
                  f"usage_cost=${record.get('estimated_usd',0):.5f}",flush=True)


class MeasuredNode:
    def __init__(self,node,name,trace):
        self.node=node;self.name=name;self.trace=trace
    def __getattr__(self,name):
        return getattr(self.node,name)
    def __call__(self,*a,**kw):
        with self.trace.span("node_"+self.name):
            return self.node(*a,**kw)


def run(limit,budget):
    from src.runtime_config import load_runtime_config
    load_runtime_config()
    if os.getenv("GEMINI_MODEL","gemini-2.5-flash")!="gemini-2.5-flash":
        raise RuntimeError("Model differs from preregistered 2.5 Flash; explicit reconfiguration needed")
    prepare()
    import chromadb
    import torch
    from chromadb.config import Settings
    from google.genai import _api_client
    from src.agent.agent import PaperRAGAgent
    from src.bm25 import search_bm25_for_RAG as bm
    from src.chroma_fulltext import retrieve as dense
    from src.agent.nodes import external_searcher as external
    from src.agent.nodes import search_executor as search_module
    # Streamlit launches app.py with src/web on sys.path; reproduce that import context.
    sys.path.insert(0,str(ROOT / "src/web"))
    from src.web.core import rag as web_rag
    from langchain_google_genai import ChatGoogleGenerativeAI

    trace=Trace(budget)
    original_http=_api_client.BaseApiClient._request_once
    def http_request(client,request,*a,**kw):
        # Log status and duration only, never URL/headers/body or credential.
        entry={};start=time.perf_counter()
        try:
            response=original_http(client,request,*a,**kw)
            entry["status"]="ok"
            return response
        except Exception as e:
            entry.update({"status":"error","error_type":type(e).__name__,"code":getattr(e,"code",None)})
            raise
        finally:
            entry["elapsed_ms"]=(time.perf_counter()-start)*1000
            if trace.active is not None:
                trace.active["http_attempts"].append(entry)
    _api_client.BaseApiClient._request_once=http_request

    private=chromadb.PersistentClient(path=str(OUT / "index_private"),
        settings=Settings(anonymized_telemetry=False)).get_collection("papers_fulltext")
    source_ids=set(private.get(include=[]) ["ids"])
    # A resumed run may contain prior external additions. Recover the initial IDs from disk.
    if (OUT / "base_ids.json").exists():
        source_ids=set(read(OUT / "base_ids.json"))
    else:
        write(OUT / "base_ids.json",sorted(source_ids))
    bm.CHROMA_PATH=str(OUT / "index_private")
    dense.collection=private;dense.collection_error=None
    external.ExternalSearcher._get_chromadb_collection=lambda self:private
    search_module.apply_rrf_merge=trace.timed("retrieval_rrf",search_module.apply_rrf_merge)
    # Wrap the class function, not an instance's bound method: an instance-level
    # closure would retain its own searcher in a reference cycle after reload.
    bm.BM25ChunkSearcherForRAG.search=trace.timed("retrieval_bm25",bm.BM25ChunkSearcherForRAG.search)
    dense.rerank_chunks=trace.timed("retrieval_rerank",dense.rerank_chunks)
    external.ExternalSearcher._search_arxiv=trace.timed("arxiv_http_and_parse",external.ExternalSearcher._search_arxiv)
    external.ExternalSearcher._enrich_with_pdf_content=trace.timed("pdf_enrichment",external.ExternalSearcher._enrich_with_pdf_content)
    external.ExternalSearcher._ingest_to_chromadb=trace.timed("external_ingestion",external.ExternalSearcher._ingest_to_chromadb)

    init=time.perf_counter()
    agent=PaperRAGAgent()
    original_reload=agent.search_executor.reload_bm25_searcher
    def reload_index():
        with trace.span("bm25_reload"):
            original_reload()
    agent.search_executor.reload_bm25_searcher=reload_index
    agent.search_executor.chroma_search=trace.timed("retrieval_dense_including_rerank",agent.search_executor.chroma_search)
    for node,stage in [(agent.intent_classifier,"intent_classification"),
                       (agent.answer_generator,"answer_generation"),
                       (agent.result_evaluator,"answer_evaluation")]:
        node.llm=MeasuredLLM(node.llm,stage,trace)
    if agent.external_searcher.query_rewriter:
        agent.external_searcher.query_rewriter=MeasuredLLM(agent.external_searcher.query_rewriter,
            "arxiv_query_rewrite",trace)
    for attr in ["intent_classifier","search_executor","context_extractor","answer_generator",
                 "result_evaluator","external_searcher","response_formatter"]:
        setattr(agent,attr,MeasuredNode(getattr(agent,attr),attr,trace))
    agent.graph=agent._build_graph()
    web_rag.load_agent=lambda:agent
    rewriter=ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=os.environ["GOOGLE_API_KEY"],temperature=0)
    web_rag.load_query_rewriter=lambda:MeasuredLLM(rewriter,"conversation_rewrite",trace)

    # Local-only warmup: no generation, no paid warmup call.
    agent.search_executor({"intent":"global","query":"BERT masked language modeling","search_mode":"hybrid"})
    initialization=time.perf_counter()-init
    config={"model":"gemini-2.5-flash","max_context_tokens":agent.context_extractor.max_tokens,
        "generation_max_output_tokens":agent.answer_generator.llm.max_output_tokens,
        "generation_temperature":agent.answer_generator.llm.temperature,
        "candidate_pool_bm25":100,"dense_retrieve_pool":dense.RETRIEVAL_TOP_K,
        "reranker_enabled":dense.USE_RERANKER,"reranker_loaded":dense.load_reranker() is not None,
        "rrf_k":60,"max_external_iterations":2,"timeout_seconds_per_call":60,
        "sdk_max_retries_field":agent.answer_generator.llm.max_retries,
        "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads":torch.get_num_threads(),"source":read(OUT / "source_before.json"),
        "versions":{p:importlib.metadata.version(p) for p in
            ["langchain-google-genai","google-genai","langgraph","chromadb","torch","flashrank"]},
        "pricing":{"date":"2026-10-27","source":PRICE_URL,"input_usd_per_million":RATE_IN,
                   "output_including_reasoning_usd_per_million":RATE_OUT,"cached_input_usd_per_million":RATE_CACHE},
        "budget_cap_observed_usage_usd":budget,"initialization_plus_local_warmup_seconds":initialization,
        "protocol":"30 prespecified questions: 18 standalone, 6 follow-ups, 6 recent-coverage probes. "
          "First five are a pilot retained only when instrumentation remains valid. "
          "Sequential requests through actual web wrapper; one observation/query; no UI/network-browser overhead. "
          "Full process/index/model initialization, local warmup, corpus reset and BM25 reset excluded. "
          "Before each query remove all external IDs from a disposable copy and rebuild BM25 if needed. "
          "Logical contents reset; HNSW deletion/history may persist. "
          "Follow-up history uses the actual previous response, not a synthetic reference. "
          "Prompts, temperatures, thinking defaults, candidate pools and evaluator policy unchanged. "
          "60-second transport timeout; SDK retry configuration retained, HTTP attempts instrumented. "
          "LLM output_tokens already includes reasoning; never add reasoning twice. "
          "Listed-price usage estimate, not invoice. Failed calls without usage have unknown charges. "
          "Nested stage timings overlap and must not be summed across levels."}
    if (OUT / "manifest.json").exists():
        previous=read(OUT / "manifest.json")
        config["prior_initializations_seconds"]=previous.get("prior_initializations_seconds",[])+[
            previous["initialization_plus_local_warmup_seconds"]]
    write(OUT / "manifest.json",config)
    for case in read(OUT / "queries.json")[:limit]:
        target=OUT / "responses" / f"{case['query_id']}.json"
        if target.exists():
            print("SKIP cached",case["query_id"],flush=True);continue
        trace.qid="reset";trace.calls=[];trace.spans=[]
        current=set(private.get(include=[])["ids"])
        extra=current-source_ids
        if extra:
            private.delete(ids=sorted(extra))
            reload_index()
        assert private.count()==len(source_ids)
        history=[]
        if case["parent"]:
            parent=read(OUT / "responses" / f"{case['parent']}.json")
            history=[{"role":"user","content":parent["case"]["question"]},
                     {"role":"assistant","content":parent["output"]["answer"]}]
        trace.qid=case["query_id"];trace.calls=[];trace.spans=[]
        print("START",trace.qid,case["category"],flush=True)
        start=time.perf_counter()
        output=web_rag.run_agent_rag(case["question"],chat_history=history,chat_id=trace.qid)
        elapsed=(time.perf_counter()-start)*1000
        record={"case":case,"history":history,"end_to_end_ms":elapsed,"output":output,
            "calls":trace.calls,"spans":trace.spans,"initial_chunks":len(source_ids),
            "final_chunks":private.count(),"external_attempted":any(
                s["stage"]=="node_external_searcher" for s in trace.spans),
            "estimated_usd":sum(c.get("estimated_usd",0) for c in trace.calls)}
        write(target,record)
        print(f"DONE {trace.qid} {elapsed/1000:.2f}s calls={len(trace.calls)} "
              f"cost=${record['estimated_usd']:.5f} total=${trace.total_cost:.4f}",flush=True)
        summary()
        # Freeze evidence and stop after persistent API failures rather than silently measuring fallback strings.
        failed=[c for c in trace.calls if c["status"]!="ok"]
        if failed:
            write(OUT / "API_FAILURE.json",{"query_id":trace.qid,"failed_calls":len(failed),
                "error_types":[c.get("error_type") for c in failed]})
            raise RuntimeError("API failure recorded; inspect before continuing")
    verify=snapshot_source()
    before=read(OUT / "source_before.json")
    write(OUT / "source_verification.json",{"before":before,"after":verify,"unchanged":before==verify})
    assert before==verify
    summary()


def summary():
    import numpy as np
    rows=[];calls=[];spans=[]
    base_ids=set(read(OUT / "base_ids.json"))
    for path in sorted((OUT / "responses").glob("qa*.json")):
        r=read(path);cs=r["calls"];ss=r["spans"];calls.extend(cs);spans.extend(ss)
        rows.append({"query_id":r["case"]["query_id"],"category":r["case"]["category"],
            "end_to_end_ms":r["end_to_end_ms"],"logical_calls":len(cs),
            "http_attempts":sum(len(c["http_attempts"]) for c in cs),
            "failed_calls":sum(c["status"]!="ok" for c in cs),
            "input_tokens":sum(c.get("input_tokens",0) for c in cs),
            "output_tokens_including_reasoning":sum(c.get("output_tokens",0) for c in cs),
            "reasoning_tokens":sum(c.get("reasoning_tokens",0) for c in cs),
            "visible_output_tokens":sum(c.get("visible_output_tokens",0) for c in cs),
            "cached_input_tokens":sum(c.get("cached_tokens",0) for c in cs),
            "total_tokens":sum(c.get("total_tokens",0) for c in cs),
            "estimated_usd":r["estimated_usd"],"external_attempted":r["external_attempted"],
            "external_iterations":sum(s["stage"]=="node_external_searcher" for s in ss),
            "external_chunks_added":r["final_chunks"]-r["initial_chunks"],
            "external_chunks_cited":sum(s.get("chunk_id") not in base_ids for s in r["output"].get("sources",[]) if s.get("chunk_id")),
            "used_external_papers":r["output"].get("used_external_papers",False)})
    if not rows:return
    csv_write(OUT / "per_query.csv",rows)
    csv_write(OUT / "per_call.csv",[{k:v for k,v in c.items() if k not in
        ["prompt","response","response_metadata","usage","http_attempts"]} | {
            "http_attempts":len(c["http_attempts"]),
            "finish_reason":(c.get("response_metadata") or {}).get("finish_reason")} for c in calls])
    csv_write(OUT / "stage_spans.csv",spans)
    groups=[]
    for label,subset in [("all",rows),("no_external",[r for r in rows if not r["external_attempted"]]),
                         ("external_attempted",[r for r in rows if r["external_attempted"]])]+[
        (k,[r for r in rows if r["category"]==k]) for k in ["standalone","follow_up","recent_coverage_probe"]]:
        if not subset:continue
        groups.append({"group":label,"queries":len(subset),
            "median_e2e_seconds":float(np.median([r["end_to_end_ms"] for r in subset]))/1000,
            "p95_e2e_seconds":float(np.percentile([r["end_to_end_ms"] for r in subset],95))/1000,
            **{"mean_"+k:statistics.mean(r[k] for r in subset) for k in ["logical_calls","http_attempts",
                "input_tokens","output_tokens_including_reasoning","reasoning_tokens","visible_output_tokens",
                "cached_input_tokens","total_tokens","estimated_usd"]}})
    csv_write(OUT / "summary.csv",groups)
    llms=[]
    for stage in sorted({c["stage"] for c in calls}):
        subset=[c for c in calls if c["stage"]==stage and c["status"]=="ok"]
        if not subset:continue
        llms.append({"stage":stage,"calls":len(subset),"median_ms":float(np.median([c["elapsed_ms"] for c in subset])),
            "p95_ms":float(np.percentile([c["elapsed_ms"] for c in subset],95)),
            **{"mean_"+k:statistics.mean(c[k] for c in subset) for k in
                ["input_tokens","output_tokens","reasoning_tokens","estimated_usd"]}})
    csv_write(OUT / "llm_stage_summary.csv",llms)
    local=[]
    for stage in sorted({s["stage"] for s in spans}):
        subset=[s["elapsed_ms"] for s in spans if s["stage"]==stage]
        local.append({"stage":stage,"occurrences":len(subset),"median_ms":float(np.median(subset)),
            "p95_ms":float(np.percentile(subset,95))})
    csv_write(OUT / "local_stage_summary.csv",local)
    lines=["# Online QA operating-cost measurements", "",f"Completed queries: {len(rows)}/30. Successful LLM calls: {sum(c['status']=='ok' for c in calls)}. "
        f"Observed listed-price estimate: ${sum(r['estimated_usd'] for r in rows):.6f}.","",
        "| Group | n | E2E median / p95 seconds | Mean logical calls | Mean input / output tokens | Mean USD |",
        "|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f"| {g['group']} | {g['queries']} | {g['median_e2e_seconds']:.2f} / {g['p95_e2e_seconds']:.2f} | "
            f"{g['mean_logical_calls']:.2f} | {g['mean_input_tokens']:.1f} / {g['mean_output_tokens_including_reasoning']:.1f} | {g['mean_estimated_usd']:.6f} |")
    lines += ["", "## LLM stages (successful calls)","", "| Stage | Calls | Median / p95 ms | Mean input / output tokens |", "|---|---:|---:|---:|"]
    for g in llms:
        lines.append(f"| {g['stage']} | {g['calls']} | {g['median_ms']:.1f} / {g['p95_ms']:.1f} | {g['mean_input_tokens']:.1f} / {g['mean_output_tokens']:.1f} |")
    lines += ["", "## Local and external stages (inclusive spans)", "",
        "| Stage | Occurrences | Median / p95 ms |", "|---|---:|---:|"]
    for g in local:
        if not g["stage"].startswith("node_"):
            lines.append(f"| {g['stage']} | {g['occurrences']} | {g['median_ms']:.1f} / {g['p95_ms']:.1f} |")
    lines += ["", "## Protocol and limitations", "",read(OUT / "manifest.json")["protocol"],"",
        f"Pricing: [Gemini Developer API standard listed prices]({PRICE_URL}), referenced 2026-10-27. "
        "$0.30/M input, $2.50/M output including reasoning, $0.03/M cached input. Not invoice, free-credit accounting or hardware/power cost.",
        "", "- LLM output_tokens already includes reasoning in this installed LangChain SDK, verified against provider usage. Visible output and reasoning are also recorded separately.",
        "- End-to-end means server-side web-wrapper execution before rewrite through final normalized response, not browser rendering or time to first token.",
        "- The 30 queries are prespecified synthetic needs, not representative traffic. Follow-up questions depend on their parent answers. Probe mix is deliberate. Small-sample p95 is descriptive.",
        "- External_attempted is determined from actual node traces, not the final evaluator flag (which can change after regeneration). No search was forced or evaluator policy changed.",
        "- If arXiv fails or returns no papers, that is reported; those requests do not establish successful external-evidence acquisition.",
        "- Single observation per question; network/provider load and thermal conditions uncontrolled. Stage occurrences are nested, may repeat and must not be summed across inclusive levels.",
        "- Raw calls contain research query/passage prompts and responses but no credential or HTTP headers. HTTP hooks record attempts/status/duration only.",
        "- Setup connection diagnostics are excluded from formal QA cost; local warmup makes no LLM call. Corpus copied privately, never ingested into production.", ""]
    exclusion=OUT / "diagnostic_pilot_reference_cycle/EXCLUSION.json"
    if exclusion.exists():
        diag=read(exclusion)
        lines += ["## Excluded technical pilot", "",
            "An early instance-level timing wrapper could retain obsolete BM25 searchers. "
            "It was replaced with a class-level wrapper before the formal pilot. "
            "All earlier observations are archived separately and excluded from these formal summaries. "
            f"Recorded diagnostic calls: {diag['recorded_calls']}; observed diagnostic listed-price estimate: "
            f"${diag['listed_price_estimated_usd']:.6f}. An interrupted diagnostic query is not a formal observation. "
            "The experiment budget includes recorded diagnostic calls; unknown charges cannot be inferred from missing responses.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    write(OUT / "STATUS.json",{"completed_queries":len(rows),"planned_queries":30,
        "successful_llm_calls":sum(c["status"]=="ok" for c in calls),
        "failed_llm_calls":sum(c["status"]!="ok" for c in calls),
        "observed_usage_estimated_usd":sum(r["estimated_usd"] for r in rows)})


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--limit",type=int,default=5)
    parser.add_argument("--budget-usd",type=float,default=1.)
    parser.add_argument("--report-only",action="store_true")
    args=parser.parse_args()
    if args.report_only:summary()
    else:
        assert 1<=args.limit<=30 and 0<args.budget_usd<=1
        run(args.limit,args.budget_usd)
