# -*- coding: utf-8 -*-
"""
Chunk-level BM25 searcher for RAG.

This module is separate from the legacy paper-level BM25 search used by the
UI search flow. It indexes chunked full-text documents so it can be paired with
the full-text Chroma retriever in the agent pipeline.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from src.config import QUERY_PATH, TOP_K


CHROMA_PATH = "./data/chroma_store_fulltext"
COLLECTION_NAME = "papers_fulltext"
OUTPUT_PATH_BM25_RAG = os.getenv(
    "OUTPUT_PATH_BM25_RAG",
    "./src/bm25/results_rag.json",
)

CHUNK_SIZE = int(os.getenv("RAG_BM25_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_BM25_CHUNK_OVERLAP", "200"))
MIN_FULLTEXT_WORDS = int(os.getenv("RAG_BM25_MIN_FULLTEXT_WORDS", "30"))
COLLECTION_BATCH_SIZE = int(os.getenv("RAG_BM25_COLLECTION_BATCH_SIZE", "1000"))

MARKDOWN_SEPARATORS = [
    r"\n#{1,6} ",
    r"```\n",
    r"\n\*\*\*+\n",
    r"\n---+\n",
    r"\n___+\n",
    r"\n\n",
    r"\n",
    r" ",
    r"",
]


@dataclass
class SplitPiece:
    text: str
    start: int


@dataclass
class Chunk:
    text: str
    start: int


def tokenize(text):
    """Simple word tokenization for BM25."""
    if not text:
        return []
    return re.findall(r"\w+", str(text).lower())


def load_queries(path):
    """Read queries.json file."""
    if not os.path.exists(path):
        print(f"X File not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data and isinstance(data[0], dict):
        return [item.get("query", item.get("original_query")) for item in data]
    return data


def normalize_text(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def trim_chunk(text, start):
    text = text.strip()
    if not text:
        return "", start

    leading = len(text) - len(text.lstrip())
    return text, start + leading


def split_with_separator(text, separator):
    if separator == "":
        return [SplitPiece(text=text, start=0)] if text else []

    parts = []
    last_index = 0

    for match in re.finditer(separator, text):
        end_index = match.end()
        if end_index > last_index:
            parts.append(SplitPiece(text=text[last_index:end_index], start=last_index))
        last_index = end_index

    if last_index < len(text):
        parts.append(SplitPiece(text=text[last_index:], start=last_index))

    if not parts and text:
        parts.append(SplitPiece(text=text, start=0))

    return parts


def recursive_split(text, absolute_start, separators):
    if len(text) <= CHUNK_SIZE:
        return [SplitPiece(text=text, start=absolute_start)] if text else []

    separator = ""
    next_separators = []

    for index, candidate in enumerate(separators):
        if candidate == "":
            separator = candidate
            next_separators = separators[index + 1 :]
            break
        if re.search(candidate, text):
            separator = candidate
            next_separators = separators[index + 1 :]
            break

    if separator == "":
        pieces = []
        for offset in range(0, len(text), CHUNK_SIZE):
            pieces.append(
                SplitPiece(
                    text=text[offset : offset + CHUNK_SIZE],
                    start=absolute_start + offset,
                )
            )
        return pieces

    pieces = []
    for part in split_with_separator(text, separator):
        if len(part.text) <= CHUNK_SIZE:
            pieces.append(SplitPiece(text=part.text, start=absolute_start + part.start))
        else:
            pieces.extend(
                recursive_split(
                    text=part.text,
                    absolute_start=absolute_start + part.start,
                    separators=next_separators,
                )
            )
    return pieces


def merge_splits(pieces):
    if not pieces:
        return []

    chunks = []
    current_text = pieces[0].text
    current_start = pieces[0].start

    for piece in pieces[1:]:
        if len(current_text) + len(piece.text) <= CHUNK_SIZE:
            current_text += piece.text
            continue

        chunk_text, chunk_start = trim_chunk(current_text, current_start)
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, start=chunk_start))

        overlap_text = current_text[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 else ""
        overlap_start = current_start + max(0, len(current_text) - len(overlap_text))
        current_text = overlap_text + piece.text
        current_start = overlap_start if overlap_text else piece.start

        while len(current_text) > CHUNK_SIZE:
            forced_text = current_text[:CHUNK_SIZE]
            forced_text, forced_start = trim_chunk(forced_text, current_start)
            if forced_text:
                chunks.append(Chunk(text=forced_text, start=forced_start))

            overlap_text = current_text[CHUNK_SIZE - CHUNK_OVERLAP : CHUNK_SIZE]
            current_start = current_start + CHUNK_SIZE - len(overlap_text)
            current_text = overlap_text + current_text[CHUNK_SIZE:]

    chunk_text, chunk_start = trim_chunk(current_text, current_start)
    if chunk_text:
        chunks.append(Chunk(text=chunk_text, start=chunk_start))

    return chunks


def split_document(full_text):
    normalized = normalize_text(full_text)
    pieces = recursive_split(
        text=normalized,
        absolute_start=0,
        separators=MARKDOWN_SEPARATORS,
    )
    return merge_splits(pieces)


def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _safe_get_collection():
    """Open the shared ChromaDB collection used by semantic search."""
    print(f"[BM25-RAG] Connecting to ChromaDB at: {CHROMA_PATH}...")
    chroma_path = Path(CHROMA_PATH)
    if not chroma_path.exists():
        return None, f"ChromaDB was not found at `{CHROMA_PATH}`."

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(name=COLLECTION_NAME)
        print(
            f"[BM25-RAG] Connected to Collection: '{COLLECTION_NAME}' "
            f"(Total chunks: {collection.count()})"
        )
        return collection, None
    except Exception as exc:
        return None, str(exc)


class BM25ChunkSearcherForRAG:
    """Chunk-level BM25 searcher for full-text RAG."""

    def __init__(self, data_path: str | None = None):
        self.data_path = data_path
        self.corpus_metadata = []
        self.corpus_texts = []
        self.collection = None
        self.collection_error = None
        tokenized_corpus = []

        self.collection, self.collection_error = _safe_get_collection()
        if self.collection is None:
            print(f"[BM25-RAG] ChromaDB unavailable: {self.collection_error}")
            self.bm25 = None
            print("[BM25-RAG] Index not ready.\n")
            return

        total_chunks = self.collection.count()
        print(
            f"[BM25-RAG] Loading {total_chunks} chunk records from ChromaDB "
            f"in batches of {COLLECTION_BATCH_SIZE}..."
        )

        try:
            for offset in range(0, total_chunks, COLLECTION_BATCH_SIZE):
                stored = self.collection.get(
                    include=["documents", "metadatas"],
                    limit=COLLECTION_BATCH_SIZE,
                    offset=offset,
                )

                ids = stored.get("ids") or []
                documents = stored.get("documents") or []
                metadatas = stored.get("metadatas") or []

                if not ids and not documents:
                    break

                for item_id, doc_text, meta in zip(ids, documents, metadatas):
                    meta = meta or {}
                    chunk_text = str(doc_text or "").strip()
                    if not chunk_text:
                        continue

                    title = str(meta.get("title") or "")
                    paper_id = str(meta.get("paper_id") or "")
                    source_url = str(meta.get("source_url") or "")
                    chunk_index = int(meta.get("chunk_index", -1))
                    chunk_start = int(meta.get("chunk_start", -1))
                    chunk_length = int(meta.get("chunk_length", len(chunk_text)))
                    chunk_id = str(meta.get("chunk_id") or item_id or "")

                    search_text = f"{title} {chunk_text}".strip()
                    tokenized_corpus.append(tokenize(search_text))
                    self.corpus_texts.append(search_text)
                    self.corpus_metadata.append(
                        {
                            "paper_id": paper_id,
                            "title": title,
                            "source_url": source_url,
                            "chunk_id": chunk_id or f"{paper_id}_chunk_{chunk_index:04d}",
                            "chunk_index": chunk_index,
                            "chunk_start": chunk_start,
                            "chunk_length": chunk_length,
                            "chunk_text": chunk_text,
                        }
                    )
        except Exception as exc:
            self.collection_error = str(exc)
            self.bm25 = None
            print(f"[BM25-RAG] Failed to read ChromaDB collection in batches: {exc}")
            print("[BM25-RAG] Index not ready.\n")
            return

        print(f"[BM25-RAG] Initializing BM25 Engine for {len(tokenized_corpus)} chunks...")
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        if self.bm25 is None:
            print("[BM25-RAG] No chunks found in ChromaDB.\n")
        else:
            print("[BM25-RAG] Index ready!\n")

    def search(self, query: str, top_k: int = 20):
        """Perform chunk-level lexical search."""
        if self.bm25 is None:
            return []

        tokenized_query = tokenize(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        scored_docs = [(i, float(score)) for i, score in enumerate(doc_scores) if score > 0]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_results = scored_docs[:top_k]
        output = []

        for rank, (doc_idx, score) in enumerate(top_results, start=1):
            meta = self.corpus_metadata[doc_idx]
            output.append(
                {
                    "rank": int(rank),
                    "paper_id": meta["paper_id"],
                    "title": meta["title"],
                    "source_url": meta["source_url"],
                    "chunk_id": meta["chunk_id"],
                    "chunk_index": meta["chunk_index"],
                    "chunk_start": meta["chunk_start"],
                    "chunk_length": meta["chunk_length"],
                    "chunk_text": meta["chunk_text"],
                    "text": meta["chunk_text"],
                    "score": round(score, 4),
                    "bm25_score": round(score, 4),
                }
            )

        return output


def main():
    queries = load_queries(QUERY_PATH)
    if not queries:
        return

    searcher = BM25ChunkSearcherForRAG()
    all_results = []

    print(f"🚀 [BM25-RAG] Bắt đầu tìm kiếm cho {len(queries)} queries (Top {TOP_K})...")

    for idx, query in enumerate(queries):
        print(f"[{idx+1:02d}/{len(queries)}] Đang tìm: '{query}'")
        search_results = searcher.search(query, top_k=TOP_K)
        all_results.append({"query": query, "results": search_results})

    os.makedirs(os.path.dirname(OUTPUT_PATH_BM25_RAG), exist_ok=True)
    with open(OUTPUT_PATH_BM25_RAG, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("\n🎉 HOÀN TẤT BM25 RAG SEARCH!")
    print(f"📂 File kết quả: {OUTPUT_PATH_BM25_RAG}")


if __name__ == "__main__":
    main()
