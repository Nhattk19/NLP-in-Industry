import os
from contextlib import redirect_stdout, redirect_stderr

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from flashrank import RerankRequest

from core.config import TOP_K
from core.resources import tokenize


def vector_search(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    # Suppress ChromaDB internal logging
    with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

    if not results["ids"] or not results["ids"][0]:
        return []

    search_results = []
    for paper_id, metadata, distance in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        item = metadata.copy() if metadata else {}
        item["retrieve_score"] = round(float(distance), 4)
        item["paper_id"] = paper_id
        search_results.append(item)

    return search_results


def semantic_search(collection, reranker, query: str) -> list[dict]:
    search_results = vector_search(collection, query, top_k=TOP_K)
    if not search_results:
        return []

    passages = [
        {
            "id": str(item.get("paper_id")),
            "text": f"{item.get('title', '')}. {item.get('abstract', '')}",
            "original_data": item,
        }
        for item in search_results
    ]

    reranked = reranker.rerank(RerankRequest(query=query, passages=passages))

    final_results = []
    for passage in reranked:
        item = passage["original_data"]
        item["rerank_score"] = round(float(passage["score"]), 4)
        final_results.append(item)

    return final_results


def bm25_search(
    bm25_engine,
    bm25_metadata: list[dict],
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    tokenized_query = tokenize(query)
    document_scores = bm25_engine.get_scores(tokenized_query)

    scored_docs = [(index, float(score)) for index, score in enumerate(document_scores) if score > 0]
    scored_docs.sort(key=lambda item: item[1], reverse=True)

    final_results = []
    for doc_index, score in scored_docs[:top_k]:
        item = bm25_metadata[doc_index].copy()
        item["bm25_score"] = round(score, 4)
        final_results.append(item)

    return final_results


def hybrid_search(
    collection,
    reranker,
    bm25_engine,
    bm25_metadata: list[dict],
    query: str,
    top_k: int = 10,
    semantic_top_k: int = TOP_K,
    bm25_top_k: int = TOP_K,
    use_rerank: bool = True,
) -> list[dict]:
    if collection:
        if use_rerank and reranker:
            semantic_results = semantic_search(collection, reranker, query)[:semantic_top_k]
        else:
            semantic_results = vector_search(collection, query, top_k=semantic_top_k)
    else:
        semantic_results = []

    lexical_results = (
        bm25_search(bm25_engine, bm25_metadata, query, top_k=bm25_top_k)
        if bm25_engine
        else []
    )

    if not semantic_results and not lexical_results:
        return []

    rrf_scores = {}
    merged_items = {}
    rrf_k = 40

    for rank, item in enumerate(lexical_results, start=1):
        paper_id = str(item.get("paper_id", "")).strip()
        if not paper_id:
            continue
        rrf_scores[paper_id] = rrf_scores.get(paper_id, 0.0) + (1.0 / (rrf_k + rank))
        merged_items.setdefault(paper_id, item.copy())

    for rank, item in enumerate(semantic_results, start=1):
        paper_id = str(item.get("paper_id", "")).strip()
        if not paper_id:
            continue
        rrf_scores[paper_id] = rrf_scores.get(paper_id, 0.0) + (1.0 / (rrf_k + rank))
        if paper_id in merged_items:
            merged_items[paper_id].update(item)
        else:
            merged_items[paper_id] = item.copy()

    final_results = []
    for paper_id, score in sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        item = merged_items[paper_id].copy()
        item["hybrid_score"] = round(float(score), 4)
        item["_score_lbl_override"] = f"hybrid={score:.4f}"
        final_results.append(item)

    return final_results
