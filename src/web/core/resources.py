import os
import re
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import chromadb
import streamlit as st
from flashrank import Ranker
from rank_bm25 import BM25Okapi

from core.config import CHROMA_PATH, COLLECTION_NAME, RERANKER_MODEL

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\w+", str(text).lower())


@st.cache_resource(show_spinner="Loading ChromaDB...")
def init_chromadb():
    if not os.path.exists(CHROMA_PATH):
        return None

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME)


@st.cache_resource(show_spinner="Loading embedding model...")
def init_abstract_embedder():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            device=device,
            local_files_only=True,
        )
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


@st.cache_resource(show_spinner="Loading AI reranker...")
def init_reranker():
    return Ranker(model_name=RERANKER_MODEL)


@st.cache_resource(show_spinner="Building BM25 index...")
def init_bm25() -> tuple:
    if not os.path.exists(CHROMA_PATH):
        return None, []

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    all_docs = collection.get(include=["metadatas"])

    ids = all_docs.get("ids", [])
    metadatas = all_docs.get("metadatas", [])
    if not metadatas:
        return None, []

    corpus_metadata = []
    tokenized_corpus = []

    for paper_id, metadata in zip(ids, metadatas):
        document = metadata.copy() if metadata else {}
        document["paper_id"] = paper_id
        title = document.get("title", "")
        abstract = document.get("abstract", "")
        tokenized_corpus.append(tokenize(f"{title} {abstract}"))
        corpus_metadata.append(document)

    return BM25Okapi(tokenized_corpus), corpus_metadata


@st.cache_resource(show_spinner="Preloading search resources...")
def preload_search_resources():
    """Warm Chroma, reranker, and BM25 before any search page is used."""
    collection = init_chromadb()
    reranker = init_reranker()
    bm25_engine, bm25_metadata = init_bm25()
    return collection, reranker, bm25_engine, bm25_metadata
