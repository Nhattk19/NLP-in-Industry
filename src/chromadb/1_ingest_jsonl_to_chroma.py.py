# 1_ingest_jsonl_to_chroma.py
# Encode documents (title + abstract) into embeddings and store them in ChromaDB for vector search (indexing stage)

import json
import os
import chromadb
from src.config import EMBEDDING_MODEL, DEVICE, TOP_K
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= CONFIG =================
JSONL_PATH = "./src/chromadb/0_data.jsonl"
COLLECTION_NAME = "papers_abstracts"
CHROMA_PATH = "./src/chromadb/chroma_store_abstracts"

BATCH_SIZE = 128

MIN_ABSTRACT_WORDS = 30  # ✅ safeguard nhẹ

# ================= INIT =================
client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}  # ✅ FIX: đảm bảo dùng cosine
)

model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

seen_ids = set()  # ✅ track duplicate


# ================= HELPERS =================
def build_document_text(record):
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()

    if not title and not abstract:
        return ""

    # normalize nhẹ
    title = " ".join(title.split())
    abstract = " ".join(abstract.split())

    # ✅ FIX: structured text thay vì repeat title
    if title and abstract:
        return f"title: {title}. abstract: {abstract}"
    elif title:
        return f"title: {title}"
    else:
        return f"abstract: {abstract}"


def build_metadata(record):
    authors = record.get("authors", [])
    author_names = ", ".join(
        a.get("name", "") for a in authors if a.get("name")
    )

    externalsid = record.get("externalsid", {}) or {}

    return {
        "paper_id": str(record.get("paper_id", "")),
        "title": str(record.get("title") or ""),
        "abstract": str(record.get("abstract") or ""),
        "venue": str(record.get("venue") or ""),
        "publication_date": str(record.get("publication_date") or ""),
        "is_survey": bool(record.get("is_survey", False)),
        "citation_count": int(record.get("citation_count", 0)),
        "nlp_score": int(record.get("nlp_score", 0)),
        "authors": author_names,
        "doi": str(externalsid.get("doi") or ""),
        "s2_url": str(externalsid.get("s2_url") or "")
    }


def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ================= MAIN =================
def main():
    total_lines = count_lines(JSONL_PATH)

    documents, metadatas, ids = [], [], []
    inserted = 0
    skipped_short = 0
    duplicate_count = 0

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        with tqdm(total=total_lines, desc="Reading") as read_bar, \
             tqdm(desc="Embedding", unit="batch") as embed_bar:

            for line_num, line in enumerate(f, start=1):
                read_bar.update(1)

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] Invalid JSON at line {line_num}")
                    continue

                abstract = (record.get("abstract") or "").strip()

                # ✅ FIX: safeguard abstract quá ngắn
                if len(abstract.split()) < MIN_ABSTRACT_WORDS:
                    skipped_short += 1
                    continue

                doc_text = build_document_text(record)
                if not doc_text:
                    continue

                paper_id = str(record.get("paper_id", f"line_{line_num}"))

                # ✅ FIX: detect duplicate
                if paper_id in seen_ids:
                    duplicate_count += 1
                    continue
                seen_ids.add(paper_id)

                documents.append(doc_text)
                metadatas.append(build_metadata(record))
                ids.append(paper_id)

                # ===== PROCESS BATCH =====
                if len(documents) >= BATCH_SIZE:
                    process_batch(documents, metadatas, ids, embed_bar)
                    inserted += len(ids)

                    documents, metadatas, ids = [], [], []

            # ===== FINAL BATCH =====
            if documents:
                process_batch(documents, metadatas, ids, embed_bar)
                inserted += len(ids)

    print("\n Done.")
    print(f"Inserted: {inserted}")
    print(f"Skipped (short abstract): {skipped_short}")
    print(f"Duplicates skipped: {duplicate_count}")


# ================= BATCH PROCESS =================
def process_batch(documents, metadatas, ids, embed_bar):
    try:
        embeddings = model.encode(
            documents,
            batch_size=32,
            show_progress_bar=False
        ).tolist()

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        embed_bar.update(1)

    except Exception as e:
        print(f"[ERROR] Batch failed: {e}")


# ================= ENTRY =================
if __name__ == "__main__":
    main()