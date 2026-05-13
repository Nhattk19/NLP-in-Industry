# -*- coding: utf-8 -*-
"""
Chunk-aware ChromaDB retriever for full-text RAG.

This module is used by the agent pipeline. It retrieves chunk documents,
optionally reranks them, and returns chunk-level payloads that the agent can
feed into the context builder.
"""

from __future__ import annotations

import os
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer

try:
    from flashrank import Ranker, RerankRequest
except Exception:  # pragma: no cover - optional dependency
    Ranker = None
    RerankRequest = None


# Keep the same store as ingest.py
CHROMA_PATH = "./data/chroma_store_fulltext"
COLLECTION_NAME = "papers_fulltext"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = int(os.getenv("CHROMA_TOP_K", "10"))
RETRIEVAL_TOP_K = int(os.getenv("CHROMA_RETRIEVAL_TOP_K", "30"))
SIMILARITY_THRESHOLD = float(os.getenv("CHROMA_SIMILARITY_THRESHOLD", "0.1"))
USE_RERANKER = os.getenv("CHROMA_USE_RERANKER", "1").lower() not in {"0", "false", "no"}


def _safe_get_collection():
    print(f"[INIT] Connecting to ChromaDB at: {CHROMA_PATH}...")
    chroma_path = Path(CHROMA_PATH)
    if not chroma_path.exists():
        return None, f"ChromaDB was not found at `{CHROMA_PATH}`."

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(name=COLLECTION_NAME)
        print(
            f"[OK] Connected to Collection: '{COLLECTION_NAME}' (Total papers: {collection.count()})"
        )
        return collection, None
    except Exception as exc:
        return None, str(exc)


@lru_cache(maxsize=1)
def load_embedder():
    """Load the sentence-transformer used to embed queries."""
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
    )
    snapshot_dirs = sorted(cache_root.glob("*")) if cache_root.exists() else []
    if snapshot_dirs:
        return SentenceTransformer(
            str(snapshot_dirs[-1]),
            device=DEVICE,
            local_files_only=True,
        )
    return SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)


@lru_cache(maxsize=1)
def load_reranker():
    """Load the optional reranker used for chunk reranking."""
    if not USE_RERANKER or Ranker is None:
        return None
    try:
        return Ranker(model_name=RERANKER_MODEL)
    except Exception:
        return None


collection, collection_error = _safe_get_collection()


def _chunk_similarity(distance: float) -> float:
    return max(0.0, 1.0 - float(distance))


def retrieve_chunks(
    collection_obj,
    embedder,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    where: dict | None = None,
):
    """Retrieve raw chunks from ChromaDB."""
    query_embedding = embedder.encode([query]).tolist()

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            results = collection_obj.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                where=where,
            )

    if not results.get("ids") or not results["ids"][0]:
        return [], [], []

    docs, metas, dists = [], [], []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = _chunk_similarity(float(dist))
        if similarity < SIMILARITY_THRESHOLD:
            continue
        docs.append(doc or "")
        metas.append(meta or {})
        dists.append(float(dist))

    return docs, metas, dists


def rerank_chunks(
    query: str,
    docs: list[str],
    metas: list[dict],
    dists: list[float],
    reranker,
    top_k: int = TOP_K,
):
    """Rerank retrieved chunks if a reranker is available."""
    if not docs or reranker is None or RerankRequest is None:
        return docs[:top_k], metas[:top_k], dists[:top_k], []

    passages = []
    for index, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        passages.append(
            {
                "id": str(index),
                "text": doc,
                "meta": meta,
                "distance": dist,
            }
        )

    reranked = reranker.rerank(RerankRequest(query=query, passages=passages))

    final_docs, final_metas, final_dists, rerank_scores = [], [], [], []
    for passage in reranked[:top_k]:
        final_docs.append(passage["text"])
        final_metas.append(passage["meta"])
        final_dists.append(float(passage["distance"]))
        rerank_scores.append(float(passage["score"]))

    return final_docs, final_metas, final_dists, rerank_scores


def build_chunk_context(docs: list[str], metas: list[dict]) -> str:
    """Format retrieved chunks into a prompt-ready context string."""
    parts = []
    for index, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("source_url") or meta.get("paper_id") or "unknown"
        title = meta.get("title") or "Untitled"
        chunk_index = meta.get("chunk_index", "?")
        chunk_start = meta.get("chunk_start", "?")
        parts.append(
            f"[{index}] Source: {source} | Title: {title} | Chunk: {chunk_index} | Start: {chunk_start}\n{doc}"
        )
    return "\n\n---\n\n".join(parts)


def _format_result(doc: str, meta: dict, dist: float, rank: int, rerank_score: float | None = None):
    similarity = _chunk_similarity(dist)
    paper_id = str(meta.get("paper_id", ""))
    chunk_index = meta.get("chunk_index", -1)
    chunk_start = meta.get("chunk_start", -1)
    chunk_length = meta.get("chunk_length", len(doc or ""))

    result = {
        "rank": rank,
        "paper_id": paper_id,
        "title": meta.get("title", "Untitled"),
        "source_url": meta.get("source_url", ""),
        "chunk_id": f"{paper_id}_chunk_{int(chunk_index):04d}" if paper_id and chunk_index != -1 else meta.get("chunk_id", ""),
        "chunk_index": int(chunk_index) if chunk_index != -1 else -1,
        "chunk_start": int(chunk_start) if chunk_start != -1 else -1,
        "chunk_length": int(chunk_length) if chunk_length else len(doc or ""),
        "chunk_text": doc or "",
        "text": doc or "",
        "distance": float(dist),
        "similarity": float(similarity),
        "score": float(rerank_score) if rerank_score is not None else float(similarity),
    }

    # Keep a short snippet for debugging/UI if needed.
    result["snippet"] = (doc or "")[:500]
    return result


def search(
    query: str,
    top_k: int = TOP_K,
    *,
    retrieve_top_k: int = RETRIEVAL_TOP_K,
    where: dict | None = None,
    rerank: bool = True,
):
    """Search full-text chunks and return chunk-level results.

    This is the entry point used by src/agent/nodes/search_executor.py.
    """
    if collection is None:
        print(f"[WARN] ChromaDB collection not available: {collection_error}")
        return []

    try:
        embedder = load_embedder()
    except Exception as exc:
        print(f"[WARN] Failed to load embedder: {exc}")
        return []

    docs, metas, dists = retrieve_chunks(
        collection,
        embedder,
        query,
        top_k=retrieve_top_k,
        where=where,
    )

    rerank_scores: list[float] = []
    if rerank:
        reranker = load_reranker()
        docs, metas, dists, rerank_scores = rerank_chunks(
            query=query,
            docs=docs,
            metas=metas,
            dists=dists,
            reranker=reranker,
            top_k=top_k,
        )
    else:
        docs, metas, dists = docs[:top_k], metas[:top_k], dists[:top_k]

    output = []
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        rerank_score = rerank_scores[rank - 1] if rank - 1 < len(rerank_scores) else None
        output.append(_format_result(doc, meta, dist, rank, rerank_score=rerank_score))

    return output


def build_context_from_results(results: list[dict]) -> str:
    """Build a RAG context string directly from search results."""
    docs = [item.get("chunk_text") or item.get("text") or item.get("abstract", "") for item in results]
    metas = [
        {
            "paper_id": item.get("paper_id"),
            "title": item.get("title"),
            "source_url": item.get("source_url"),
            "chunk_index": item.get("chunk_index"),
            "chunk_start": item.get("chunk_start"),
        }
        for item in results
    ]
    return build_chunk_context(docs, metas)
