import os
import json
import re
from rank_bm25 import BM25Okapi
from src.config import DATA_PATH, QUERY_PATH, TOP_K, OUTPUT_PATH_BM25

# ================= HELPER FUNCTIONS =================
def tokenize(text):
    """Simple word tokenization for BM25 (lowercase, remove punctuation)"""
    if not text: return[]
    return re.findall(r'\w+', str(text).lower())

def load_queries(path):
    """Read queries.json file"""
    if not os.path.exists(path):
        print(f"X File not found: {path}")
        return[]
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data[0], dict):
        return[item.get("query", item.get("original_query")) for item in data]
    return data

# ================= BM25 SEARCHER CLASS =================
class BM25Searcher:
    def __init__(self, data_path):
        self.corpus_metadata = []
        tokenized_corpus =[]
        
        print(f"[BM25] Loading data from {os.path.basename(data_path)} to build index...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    # Combine Title and Abstract for search text
                    title = doc.get("title", "")
                    abstract = doc.get("abstract", "")
                    search_text = f"{title} {abstract}"
                    
                    # Tokenize and add to corpus
                    tokenized_corpus.append(tokenize(search_text))
                    
                    # Store only essential metadata to optimize memory
                    self.corpus_metadata.append({
                        "paper_id": str(doc.get("paper_id", "")),
                        "title": title,
                        "abstract": abstract
                    })
                except json.JSONDecodeError:
                    continue

        print(f"[BM25] Initializing BM25 Engine for {len(tokenized_corpus)} papers...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("[BM25] Index ready!\n")

    def search(self, query: str, top_k: int = 20):
        """Perform lexical search"""
        tokenized_query = tokenize(query)
        
        # Score all documents
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Keep only documents with score > 0
        scored_docs =[(i, float(score)) for i, score in enumerate(doc_scores) if score > 0]
        
        # Sort by BM25 score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Get Top K
        top_results = scored_docs[:top_k]
        
        output =[]
        for rank, (doc_idx, score) in enumerate(top_results, start=1):
            meta = self.corpus_metadata[doc_idx]
            
            # Format as 5-field output
            output.append({
                "rank": int(rank),
                "paper_id": meta["paper_id"],
                "title": meta["title"],
                "abstract": meta["abstract"],
                "score": round(score, 4)
            })
            
        return output

# ================= LUỒNG CHẠY CHÍNH =================
def main():
    queries = load_queries(QUERY_PATH)
    if not queries:
        return

    # Khởi tạo Engine
    searcher = BM25Searcher(DATA_PATH)
    all_results =[]
    
    print(f"🚀 [BM25] Bắt đầu tìm kiếm cho {len(queries)} queries (Top {TOP_K})...")

    for idx, query in enumerate(queries):
        print(f"[{idx+1:02d}/{len(queries)}] Đang tìm: '{query}'")
        
        # Lấy kết quả
        search_results = searcher.search(query, top_k=TOP_K)
        
        # Đóng gói chuẩn format: Chỉ gồm "query" và "results"
        all_results.append({
            "query": query,
            "results": search_results
        })

    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(OUTPUT_PATH_BM25), exist_ok=True)

    # Ghi file
    with open(OUTPUT_PATH_BM25 , "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("\n🎉 HOÀN TẤT LEXICAL SEARCH!")
    print(f"📂 File kết quả: {OUTPUT_PATH_BM25}")

if __name__ == "__main__":
    main()