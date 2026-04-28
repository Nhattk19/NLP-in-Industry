import os
import re

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from flashrank import Ranker
from rank_bm25 import BM25Okapi

from core.config import CHROMA_PATH, COLLECTION_NAME, RERANKER_MODEL


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\w+", str(text).lower())


@st.cache_resource(show_spinner="Loading ChromaDB...")
def init_chromadb():
    if not os.path.exists(CHROMA_PATH):
        return None

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)


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
