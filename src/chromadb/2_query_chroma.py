# 2_query_chroma.py
import json
import chromadb
from sentence_transformers import SentenceTransformer
import torch

# ================= CONFIG =================
CHROMA_PATH = "./src/chromadb/chroma_store_abstracts"
COLLECTION_NAME = "papers_abstracts"
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5
OUTPUT_PATH = "./src/chromadb/2_demo.json"

QUERIES = [
    "papers related to SOTA"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= INIT =================
print(f"Using device: {DEVICE}")

client = chromadb.PersistentClient(path=CHROMA_PATH)
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
collection = client.get_collection(name=COLLECTION_NAME)

print(f"Connected to collection: {COLLECTION_NAME}")

# ================= SEARCH FUNCTION =================
def search(query, top_k=5):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []

    for i in range(len(ids)):
        meta = metas[i]

        output.append({
            "paper_id": meta.get("paper_id"),
            "title": meta.get("title", ""),
            "abstract": meta.get("abstract", ""),
            "score": float(distances[i]),
            "doi": meta.get("doi", ""),
            "s2_url": meta.get("s2_url", "")
        })

    return output

def print_statistics(results):
    # average score
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Average score: {avg_score:.4f}")

    # have abstract vs no abstract
    with_abstract = sum(1 for r in results if r["abstract"])
    without_abstract = len(results) - with_abstract
    print(f"With abstract: {with_abstract}, Without abstract: {without_abstract}")
   

# ================= MAIN PIPELINE =================
def main():
    all_results = []

    for query in QUERIES:
        print(f"Processing query: {query}")

        results = search(query, TOP_K)

        all_results.append({
            "query": query,
            "results": results
        })

    # save file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to {OUTPUT_PATH}")
    print_statistics([r for q in all_results for r in q["results"]])


# ================= ENTRY =================
if __name__ == "__main__":
    main()