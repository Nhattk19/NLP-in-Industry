import json
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

JSONL_PATH = "data/data_processed/papers_full.jsonl"
CHROMA_PATH = "./data/chroma_store_fulltext"
COLLECTION_NAME = "papers_fulltext"

BATCH_SIZE = 128
EMBED_BATCH_SIZE = 32
MIN_FULLTEXT_WORDS = 30
RESET_COLLECTION = True

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

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
STRIP_WHITESPACE = True
ADD_START_INDEX = True


@dataclass
class SplitPiece:
    text: str
    start: int


@dataclass
class Chunk:
    text: str
    start: int


def count_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def normalize_text(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def build_metadata(record, chunk_index, chunk):
    return {
        "paper_id": str(record.get("paper_id", "")),
        "title": str(record.get("title") or ""),
        "source_url": str(record.get("source_url") or ""),
        "chunk_index": int(chunk_index),
        "chunk_start": int(chunk.start) if ADD_START_INDEX else -1,
        "chunk_length": int(len(chunk.text)),
    }


def trim_chunk(text, start):
    if not STRIP_WHITESPACE:
        return text, start

    leading = len(text) - len(text.lstrip())
    trailing_trimmed = text.strip()
    return trailing_trimmed, start + leading


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


def load_collection(client):
    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_embedding_model():
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


def process_batch(collection, model, documents, metadatas, ids, embed_bar):
    embeddings = model.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    embed_bar.update(1)


def main():
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    collection = load_collection(client)
    model = load_embedding_model()

    total_lines = count_lines(JSONL_PATH)

    documents, metadatas, ids = [], [], []
    seen_ids = set()
    inserted_chunks = 0
    inserted_papers = 0
    skipped_short = 0
    duplicate_papers = 0
    empty_chunks = 0

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        with tqdm(total=total_lines, desc="Reading papers") as read_bar, tqdm(
            desc="Embedding", unit="batch"
        ) as embed_bar:
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

                paper_id = str(record.get("paper_id") or f"line_{line_num}")
                if paper_id in seen_ids:
                    duplicate_papers += 1
                    continue
                seen_ids.add(paper_id)

                full_text = normalize_text((record.get("full_text") or "").strip())
                if len(full_text.split()) < MIN_FULLTEXT_WORDS:
                    skipped_short += 1
                    continue

                chunks = split_document(full_text)
                if not chunks:
                    empty_chunks += 1
                    continue

                inserted_papers += 1

                for chunk_index, chunk in enumerate(chunks):
                    chunk_id = f"{paper_id}_chunk_{chunk_index:04d}"
                    documents.append(chunk.text)
                    metadatas.append(build_metadata(record, chunk_index, chunk))
                    ids.append(chunk_id)

                    if len(documents) >= BATCH_SIZE:
                        process_batch(collection, model, documents, metadatas, ids, embed_bar)
                        inserted_chunks += len(ids)
                        documents, metadatas, ids = [], [], []

            if documents:
                process_batch(collection, model, documents, metadatas, ids, embed_bar)
                inserted_chunks += len(ids)

    print("\nDone.")
    print(f"Inserted papers: {inserted_papers}")
    print(f"Inserted chunks: {inserted_chunks}")
    print(f"Skipped (short full_text): {skipped_short}")
    print(f"Skipped (duplicate papers): {duplicate_papers}")
    print(f"Skipped (no valid chunks): {empty_chunks}")
    print(f"Chroma path: {CHROMA_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunk count: {collection.count()}")


if __name__ == "__main__":
    main()
