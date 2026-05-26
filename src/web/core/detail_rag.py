from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.nodes.answer_generator import AnswerGenerator
from src.agent.nodes.context_extractor import ContextExtractor
from src.agent.nodes.response_formatter import ResponseFormatter
from src.chroma_fulltext.retrieve import collection as fulltext_collection


@st.cache_resource(show_spinner=False)
def load_detail_rag_components():
    """Warm the minimal paper-specific RAG pipeline."""
    return ContextExtractor(), AnswerGenerator(), ResponseFormatter()


def paper_has_fulltext_chunks(paper_id: str) -> bool:
    """Check whether the current paper has any indexed full-text chunks."""
    return bool(_load_fulltext_chunks(paper_id))


def _load_fulltext_chunks(paper_id: str) -> list[dict]:
    """Load every indexed full-text chunk for one paper_id."""
    paper_id = str(paper_id or "").strip()
    if not paper_id or fulltext_collection is None:
        return []

    try:
        result = fulltext_collection.get(
            where={"paper_id": paper_id},
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    if not ids:
        return []

    chunks: list[dict] = []
    for index, chunk_id in enumerate(ids):
        meta = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        doc = documents[index] if index < len(documents) and documents[index] else ""
        chunk_text = doc or meta.get("chunk_text") or meta.get("text") or meta.get("snippet") or ""
        chunks.append(
            {
                "rank": index + 1,
                "paper_id": str(meta.get("paper_id") or paper_id).strip(),
                "title": meta.get("title", "Unknown Title"),
                "source_url": meta.get("source_url", ""),
                "chunk_id": chunk_id,
                "chunk_index": meta.get("chunk_index", -1),
                "chunk_start": meta.get("chunk_start", -1),
                "chunk_length": meta.get("chunk_length", len(chunk_text)),
                "chunk_text": chunk_text,
                "text": chunk_text,
                "similarity": 1.0,
                "score": 1.0,
                "source_score": 1.0,
            }
        )

    chunks.sort(key=lambda item: (int(item.get("chunk_index", 0) or 0), str(item.get("chunk_id", ""))))
    for index, chunk in enumerate(chunks, start=1):
        chunk["rank"] = index
    return chunks


def run_detail_rag(paper: dict, question: str) -> dict:
    """Run paper-specific RAG constrained to the current paper_id."""
    paper_id = str(paper.get("paper_id", "")).strip()
    question = (question or "").strip()

    if not paper_id:
        return {
            "success": False,
            "answer": "Paper ID is missing, so I cannot run paper-specific RAG.",
            "sources": [],
            "confidence": 0.0,
            "error": "Missing paper_id",
        }

    if not question:
        return {
            "success": False,
            "answer": "Please enter a question about this paper.",
            "sources": [],
            "confidence": 0.0,
            "error": "Empty question",
        }

    results = _load_fulltext_chunks(paper_id)

    if not results:
        return {
            "success": True,
            "answer": (
                "I could not find relevant chunks in this paper for that question. "
                "Try asking about the paper's method, experiments, results, or limitations."
            ),
            "sources": [],
            "confidence": 0.2,
            "error": "",
        }

    context_extractor, answer_generator, response_formatter = load_detail_rag_components()

    state = {
        "query": question,
        "intent": "global",
        "reranked_results": results,
        "execution_path": [],
        "context_max_tokens": None,
        "context_result_limit": None,
    }

    state = context_extractor(state)
    state = answer_generator(state)
    state = response_formatter(state)

    return {
        "success": True,
        "answer": state.get("final_answer") or state.get("initial_answer") or "",
        "sources": state.get("final_sources", []),
        "confidence": state.get("final_confidence", 0.0),
        "context_size": state.get("context_size", 0),
        "execution_path": state.get("execution_path", []),
        "error": "",
    }
