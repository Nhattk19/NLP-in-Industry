"""Offline-only paper-nested index construction; no queries, LLM or production writes.

Run: python -u benchmarks/chunk_retrieval/scalability.py
Each phase has its own process so RAM and Chroma persistence are isolated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
SEED = 20261027


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_save(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def canonical(record):
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


from benchmarks.resource_memory import Memory


def source_records():
    import chromadb
    from chromadb.config import Settings
    collection = chromadb.PersistentClient(path=str(ROOT / "data/chroma_store_fulltext"),
        settings=Settings(anonymized_telemetry=False)).get_collection("papers_fulltext")
    records = []
    for offset in range(0, collection.count(), 2000):
        batch = collection.get(limit=2000, offset=offset, include=["documents", "metadatas"])
        for cid, doc, meta in zip(batch["ids"], batch["documents"], batch["metadatas"]):
            records.append({"id": cid, "document": doc, "metadata": meta or {}})
    records.sort(key=lambda r: r["id"])
    return collection, records


def fingerprint(records):
    digest = hashlib.sha256()
    for r in records:
        digest.update((canonical(r) + "\n").encode("utf-8"))
    return digest.hexdigest()


def export(out):
    collection, records = source_records()
    paper_ids = sorted({str(r["metadata"].get("paper_id", "")) for r in records})
    if "" in paper_ids:
        raise ValueError("Missing paper_id: cannot form paper-level samples")
    random.Random(SEED).shuffle(paper_ids)
    with (out / "corpus_snapshot.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    subsets = {}
    for pct in [25, 50, 100]:
        subset = paper_ids[:len(paper_ids) * pct // 100]
        save(out / f"papers_{pct}.json", subset)
        selected = set(subset)
        subsets[str(pct)] = {"papers": len(subset), "chunks": sum(
            str(r["metadata"]["paper_id"]) in selected for r in records)}
    config = {k: v for k, v in collection.configuration["hnsw"].items()}
    save(out / "manifest.json", {
        "seed": SEED, "source_collection": collection.name, "source_chunks": len(records),
        "source_papers": len(paper_ids), "source_document_metadata_sha256": fingerprint(records),
        "fingerprint_excludes_embeddings": True,
        "subsets": subsets, "hnsw": config,
        "embedding_model": "all-MiniLM-L6-v2", "embedding_batch_size": 32,
        "insert_batch_size": 128, "python": sys.version, "platform": platform.platform(),
        "versions": {p: importlib.metadata.version(p) for p in
            ["chromadb", "sentence-transformers", "torch", "rank-bm25", "psutil"]},
        "protocol": "Fresh paper-nested indexes; recompute embeddings from existing parsed chunks. "
            "Build excludes PDF acquisition/parsing/chunking and model download. Model load reported separately. "
            "Disk measured after the builder process exits. Offline construction measurements only.",
    })
    print("EXPORT", json.dumps(subsets), flush=True)


def load_subset(out, pct):
    papers = set(read(out / f"papers_{pct}.json"))
    with (out / "corpus_snapshot.jsonl").open(encoding="utf-8") as f:
        return [r for line in f if (r := json.loads(line))["metadata"]["paper_id"] in papers]


def build(out, pct):
    import chromadb
    import torch
    import psutil
    from chromadb.config import Settings
    from src.chroma_fulltext.ingest import load_embedding_model
    manifest = read(out / "manifest.json")
    target = out / f"index_{pct}"
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment index: {target}")
    records = load_subset(out, pct)
    start = time.perf_counter()
    model = load_embedding_model()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load = time.perf_counter() - start
    model.encode(["warmup scientific literature retrieval"], show_progress_bar=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    embedding_s = insert_s = 0.
    with Memory() as memory:
        start = time.perf_counter()
        client = chromadb.PersistentClient(path=str(target), settings=Settings(anonymized_telemetry=False))
        collection = client.create_collection("papers_fulltext", configuration={"hnsw": manifest["hnsw"]})
        for offset in range(0, len(records), 128):
            batch = records[offset:offset + 128]
            t = time.perf_counter()
            vectors = model.encode([r["document"] for r in batch], batch_size=32,
                show_progress_bar=False).tolist()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            embedding_s += time.perf_counter() - t
            t = time.perf_counter()
            collection.add(ids=[r["id"] for r in batch], documents=[r["document"] for r in batch],
                metadatas=[r["metadata"] for r in batch], embeddings=vectors)
            insert_s += time.perf_counter() - t
            if offset % 8192 == 0:
                print(f"BUILD {pct}% {min(offset+128,len(records))}/{len(records)} "
                      f"elapsed={time.perf_counter()-start:.1f}s", flush=True)
        elapsed = time.perf_counter() - start
        assert collection.count() == len(records)
    save(out / f"build_{pct}.json", {
        "percent": pct, **manifest["subsets"][str(pct)], "model_load_seconds": model_load,
        "embedding_seconds": embedding_s, "insert_seconds": insert_s,
        "embedding_and_index_build_seconds": elapsed, **memory.fields(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "embedding_max_seq_length": model.max_seq_length,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
        "system_ram_bytes": psutil.virtual_memory().total, "torch_threads": torch.get_num_threads(),
    })
    print(f"BUILD {pct}% complete {elapsed:.1f}s", flush=True)


def verify(out):
    _, records = source_records()
    expected = read(out / "manifest.json")["source_document_metadata_sha256"]
    actual = fingerprint(records)
    save(out / "source_verification.json", {"source_unchanged": actual == expected,
        "before_sha256": expected, "after_sha256": actual, "chunks": len(records)})
    assert actual == expected, "Source logical contents changed during experiment"


def report(out):
    rows = []
    for pct in [25, 50, 100]:
        row = read(out / f"build_{pct}.json")
        row["disk_bytes"] = sum(p.stat().st_size for p in (out / f"index_{pct}").rglob("*") if p.is_file())
        rows.append(row)
    csv_save(out / "build_summary.csv", rows)
    lines = ["# Offline index construction", "",
        "Measured embedding/index build time, process memory, and database size.",
        "No relevance judgments or query workload are used in these measurements.", "",
        "| Corpus | Papers | Chunks | Build seconds | Disk GB |",
        "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['percent']}% | {row['papers']} | {row['chunks']} | "
            f"{row['embedding_and_index_build_seconds']:.2f} | {row['disk_bytes']/1e9:.3f} |")
    lines += ["", "Build excludes acquisition, parsing, chunking, model download and startup.",
        "One build per level, in fixed order; results are specific to the recorded machine.",
        "Database sizes include text, metadata, embeddings, HNSW and database overhead."]
    (out / "BUILD_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out", default="output/benchmarks/corpus_scalability_20261027")
    parser.add_argument("--phase", choices=["export","build","verify","report"])
    parser.add_argument("--percent",type=int,choices=[25,50,100])
    args=parser.parse_args(); out=(ROOT/args.out).resolve()
    if not out.is_relative_to(ROOT / "output/benchmarks"):
        raise ValueError("Experiment output must be inside output/benchmarks")
    out.mkdir(parents=True,exist_ok=True)
    if args.phase:
        if args.phase == "build":
            globals()[args.phase](out,args.percent)
        else:
            globals()[args.phase](out)
        return
    def run(phase,pct=None):
        command=[sys.executable,"-u",str(Path(__file__).resolve()),"--out",str(out),"--phase",phase]
        if pct:
            command += ["--percent",str(pct)]
        with (out / f"{phase}_{pct or 'all'}.log").open("w",encoding="utf-8") as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                encoding="utf-8",errors="replace",env={**os.environ,"PYTHONIOENCODING":"utf-8"})
            for line in process.stdout:
                log.write(line); log.flush(); print(line,end="",flush=True)
            if process.wait():
                raise RuntimeError(f"Phase failed: {phase} {pct}; see log")
    run("export")
    for pct in [25,50,100]:
        run("build",pct)
    run("verify"); run("report")


if __name__ == "__main__":
    main()
