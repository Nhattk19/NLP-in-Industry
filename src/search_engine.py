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


def apply_rrf_merge(lexical_results, semantic_results, top_k):
    """
    Generic RRF merging for in-memory result lists
    Returns merged results with RRF scoring
    """
    rrf_scores = {}
    paper_info = {}
    
    # Add lexical results
    for res in lexical_results:
        p_id = str(res["paper_id"])
        rank = res.get("rank", lexical_results.index(res) + 1)
        rrf_scores[p_id] = rrf_scores.get(p_id, 0) + (1 / (60 + rank))
        paper_info.setdefault(p_id, res)
    
    # Add semantic results
    for res in semantic_results:
        p_id = str(res["paper_id"])
        rank = res.get("rank", semantic_results.index(res) + 1)
        rrf_scores[p_id] = rrf_scores.get(p_id, 0) + (1 / (60 + rank))
        paper_info.setdefault(p_id, res)
    
    # Sort by RRF score
    sorted_papers = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build result list
    merged_results = []
    for i, (p_id, score) in enumerate(sorted_papers[:top_k]):
        info = paper_info[p_id].copy()
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

        rrf_scores = {}
        paper_info = {}

        for res in lex_results:
            p_id = str(res["paper_id"])
            rank = res["rank"]
            rrf_scores[p_id] = rrf_scores.get(p_id, 0) + (1 / (60 + rank))
            paper_info.setdefault(p_id, res)

        for i, res in enumerate(sem_results):
            p_id = str(res["paper_id"])
            rank = res.get("rank", i + 1)
            rrf_scores[p_id] = rrf_scores.get(p_id, 0) + (1 / (60 + rank))
            paper_info.setdefault(p_id, res)

        sorted_papers = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        combined_results = []
        for i, (p_id, score) in enumerate(sorted_papers[:top_k]):
            info = paper_info[p_id]
            combined_results.append({
                "rank": i + 1,
                "paper_id": p_id,
                "title": info.get("title", ""),
                "abstract": info.get("abstract", ""),
                "score": round(score, 4)
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