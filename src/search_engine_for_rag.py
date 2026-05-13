import os
import json
import subprocess
import shutil
import sys

# Disable ChromaDB telemetry before running modules
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

from src.config import OUTPUT_PATH_RETRIEVED, OUTPUT_PATH_RERANKED, OUTPUT_PATH_CHROMADB

# ================= BASE PATH =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ================= MODULE NAMES (KHÔNG dùng path nữa) =================
MODULE_BM25 = "src.bm25.search_bm25"
MODULE_RETRIEVE = "src.chromadb.retrieve"
MODULE_RERANK = "src.chromadb.rerank"
MODULE_GENERATE = "src.chromadb.generate_final"

# ================= OUTPUT PATH =================
LEXICAL_OUT = os.path.join(BASE_DIR, "src", "bm25", "results.json")
SEMANTIC_RAW_OUT = os.path.join(BASE_DIR, "src", "chromadb", "results.json")
SEMANTIC_FINAL_OUT = os.path.join(BASE_DIR, "src", "chromadb", "final_search_results.json")
FINAL_OUT = os.path.join(BASE_DIR, "src", "final_results.json")

# ================= CONFIG =================
MODE = "hybrid"
TOP_K = 10


def run_module(module_name):
    """Chạy module bằng python -m (đúng chuẩn package)"""
    print(f"\n▶️ Running module: {module_name}")
    
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=BASE_DIR  # đảm bảo chạy từ root
    )

    if result.returncode == 0:
        print(f"✅ Done: {module_name}")
        return True
    else:
        print(f"❌ Failed: {module_name}")
        return False


def _result_key(result: dict) -> str:
    """Return the best available identity key for a retrieval result."""
    if result.get("chunk_id"):
        return str(result["chunk_id"])

    paper_id = str(result.get("paper_id", ""))
    chunk_index = result.get("chunk_index", None)
    if chunk_index is not None and chunk_index != -1 and paper_id:
        return f"{paper_id}_chunk_{int(chunk_index):04d}"

    return paper_id


def _unique_by_result_key(results):
    """Keep the first occurrence for each chunk or paper key."""
    unique = []
    seen = set()

    for res in results:
        key = _result_key(res)
        if key in seen:
            continue
        seen.add(key)
        unique.append(res)

    return unique


def _merge_result_payload(base: dict, incoming: dict) -> dict:
    """Merge two result payloads while preserving chunk-rich fields."""
    merged = base.copy()

    for key, value in incoming.items():
        if key not in merged or merged[key] in ("", None, [], {}):
            merged[key] = value
            continue

        if key in {"chunk_text", "text", "document"} and value:
            merged[key] = value

    return merged


def apply_rrf_merge(lexical_results, semantic_results, top_k):
    """
    Generic RRF merging for in-memory result lists
    Returns merged results with RRF scoring
    """
    rrf_scores = {}
    paper_info = {}

    lexical_results = _unique_by_result_key(lexical_results)
    semantic_results = _unique_by_result_key(semantic_results)
    
    # Add lexical results
    for res in lexical_results:
        key = _result_key(res)
        rank = res.get("rank", lexical_results.index(res) + 1)
        rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (60 + rank))
        paper_info[key] = _merge_result_payload(paper_info.get(key, {}), res) if key in paper_info else res.copy()
    
    # Add semantic results
    for res in semantic_results:
        key = _result_key(res)
        rank = res.get("rank", semantic_results.index(res) + 1)
        rrf_scores[key] = rrf_scores.get(key, 0) + (1 / (60 + rank))
        paper_info[key] = _merge_result_payload(paper_info.get(key, {}), res) if key in paper_info else res.copy()
    
    # Sort by RRF score
    sorted_papers = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build result list
    merged_results = []
    for i, (key, score) in enumerate(sorted_papers[:top_k]):
        info = paper_info[key].copy()
        # Preserve the original source score before normalizing the final
        # hybrid ranking score into the canonical `score` field.
        info["source_score"] = info.get("score", info.get("rrf_score", 0))
        info["score"] = score
        info["rank"] = i + 1
        info["rrf_score"] = score
        merged_results.append(info)
    
    return merged_results


def combine_hybrid_rrf(lexical_file, semantic_file, output_file, top_k):
    print("\n🔀 Hybrid RRF merging...")

    with open(lexical_file, 'r', encoding='utf-8') as f:
        lex_data = json.load(f)

    with open(semantic_file, 'r', encoding='utf-8') as f:
        sem_data = json.load(f)

    sem_dict = {
        item.get("query", item.get("original_query", "")):
        item.get("results", item.get("retrieved_results", []))
        for item in sem_data
    }

    final_data = []

    for lex_item in lex_data:
        query = lex_item.get("query", "")
        lex_results = lex_item.get("results", [])
        sem_results = sem_dict.get(query, [])

        lex_results = _unique_by_result_key(lex_results)
        sem_results = _unique_by_result_key(sem_results)

        rrf_scores = {}
        paper_info = {}

        for res in lex_results:
            p_id = _result_key(res)
            rank = res["rank"]
            rrf_scores[p_id] = rrf_scores.get(p_id, 0) + (1 / (60 + rank))
            paper_info[p_id] = _merge_result_payload(paper_info.get(p_id, {}), res) if p_id in paper_info else res.copy()

        for i, res in enumerate(sem_results):
            p_id = _result_key(res)
            rank = res.get("rank", i + 1)
            rrf_scores[p_id] = rrf_scores.get(p_id, 0) + (1 / (60 + rank))
            paper_info[p_id] = _merge_result_payload(paper_info.get(p_id, {}), res) if p_id in paper_info else res.copy()

        sorted_papers = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        combined_results = []
        for i, (p_id, score) in enumerate(sorted_papers[:top_k]):
            info = paper_info[p_id]
            combined_results.append({
                "rank": i + 1,
                "paper_id": info.get("paper_id", p_id),
                "title": info.get("title", ""),
                "abstract": info.get("abstract", ""),
                "chunk_text": info.get("chunk_text", info.get("text", "")),
                "chunk_index": info.get("chunk_index", -1),
                "chunk_id": info.get("chunk_id", p_id),
                "score": round(score, 4),
                "rrf_score": round(score, 4),
                "source_score": round(float(info.get("score", 0) or 0), 4)
            })

        final_data.append({
            "query": query,
            "results": combined_results
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print("✅ Hybrid done.")


def main():
    print("=" * 60)
    print(f"🚀 SEARCH ENGINE | MODE: {MODE.upper()}")
    print("=" * 60)

    if MODE == "lexical":
        if run_module(MODULE_BM25) and os.path.exists(LEXICAL_OUT):
            shutil.copy(LEXICAL_OUT, FINAL_OUT)

    elif MODE == "semantic":
        s1 = run_module(MODULE_RETRIEVE)
        s2 = run_module(MODULE_RERANK) if s1 else False
        s3 = run_module(MODULE_GENERATE) if s2 else False

        if s3 and os.path.exists(SEMANTIC_FINAL_OUT):
            shutil.copy(SEMANTIC_FINAL_OUT, FINAL_OUT)

    elif MODE == "hybrid":
        run_module(MODULE_BM25)
        run_module(MODULE_RETRIEVE)

        if os.path.exists(LEXICAL_OUT) and os.path.exists(SEMANTIC_RAW_OUT):
            combine_hybrid_rrf(LEXICAL_OUT, SEMANTIC_RAW_OUT, FINAL_OUT, TOP_K)
        else:
            print("❌ Missing files for hybrid.")

    else:
        print("❌ Invalid MODE")


if __name__ == "__main__":
    main()
