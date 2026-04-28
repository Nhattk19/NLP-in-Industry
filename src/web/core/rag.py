from pathlib import Path

import chromadb
import streamlit as st
import torch
from flashrank import Ranker, RerankRequest
from openai import OpenAI
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "./src/chroma_fulltext/chroma_store_fulltext"
COLLECTION_NAME = "papers_fulltext"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10
RETRIEVAL_TOP_K = 30
SIMILARITY_THRESHOLD = 0.1
OPENAI_MODEL = "gpt-4.1-nano"
EMPTY_ANSWER = "I don't know based on the provided documents."
API_KEY_FILE = Path(__file__).resolve().parent.parent / "pages" / "api_agent.txt"

SYSTEM_PROMPT = (
    "You are a strict, citation-focused assistant for a private knowledge base.\n"
    "RULES:\n"
    "1) Use ONLY the provided context to answer.\n"
    "2) The context is mostly English, but you must answer in Vietnamese by default unless the user explicitly asks for another language.\n"
    '3) If the answer is not clearly contained in the context, say: "I don\'t know based on the provided documents."\n'
    "4) Do NOT use outside knowledge, guessing, or web information.\n"
    "5) If applicable, cite sources as (source:page) using the metadata.\n"
    "6) Keep citations and source labels exactly as provided, but write the explanatory text in Vietnamese.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def load_api_key() -> str:
    try:
        return API_KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


@st.cache_resource(show_spinner=False)
def load_chroma():
    chroma_path = Path(CHROMA_PATH)
    if not chroma_path.exists():
        return None, f"ChromaDB was not found at `{CHROMA_PATH}`."

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection, None
    except Exception as exc:
        return None, str(exc)


@st.cache_resource(show_spinner=False)
def load_embedder():
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


@st.cache_resource(show_spinner=False)
def load_reranker():
    return Ranker(model_name=RERANKER_MODEL)


def init_rag_resources():
    api_key = load_api_key()
    collection, chroma_err = load_chroma()
    if chroma_err:
        return api_key, collection, chroma_err, None, None

    embedder = load_embedder()
    reranker = load_reranker()
    return api_key, collection, None, embedder, reranker


def retrieve_chunks(collection, embedder, query: str, top_k: int = RETRIEVAL_TOP_K):
    query_embedding = embedder.encode([query]).tolist()
<<<<<<< HEAD
    # Suppress ChromaDB internal logging
    import os
    from contextlib import redirect_stdout, redirect_stderr
    with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
=======
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9

    docs, metas, dists = [], [], []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = max(0.0, 1.0 - float(dist))
        if similarity < SIMILARITY_THRESHOLD:
            continue
        docs.append(doc)
        metas.append(meta)
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
    if not docs or reranker is None:
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


def build_context(docs: list[str], metas: list[dict]) -> str:
    parts = []
    for index, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("source_url") or meta.get("paper_id") or "unknown"
        title = meta.get("title") or "Untitled"
        chunk = meta.get("chunk_index", "?")
        parts.append(
            f"[{index}] Source: {source} | Title: {title} | Chunk: {chunk}\n{doc}"
        )
    return "\n\n---\n\n".join(parts)


def ask_openai(api_key: str, context: str, question: str) -> str:
    client = OpenAI(api_key=api_key)
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that follows instructions exactly.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def generate_answer(
    api_key: str,
    question: str,
    docs: list[str],
    metas: list[dict],
    dists: list[float],
    rerank_scores: list[float] | None = None,
):
    if not docs:
        return EMPTY_ANSWER, [], [], []

    context = build_context(docs, metas)
    try:
        answer = ask_openai(api_key, context, question)
        return answer, metas, dists, rerank_scores or []
    except Exception as exc:
        return f"API error: {exc}", [], [], []
