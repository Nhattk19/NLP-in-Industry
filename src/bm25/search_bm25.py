import os
import json
import re
from rank_bm25 import BM25Okapi

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
DATA_PATH = "data/data_processed/final_cleaned_data.jsonl"  # File chứa dữ liệu đã làm sạch (JSONL)

# Trỏ đến file queries giống hệt như bản Vector Semantic
QUERY_PATH = "./src/queries.json"
# File output riêng cho BM25
OUTPUT_PATH = "./src/bm25/results_bm25.json" 

try:
    import config
    TOP_K = getattr(config, "TOP_K", 20)
except ImportError:
    TOP_K = 20

# ================= HÀM HỖ TRỢ =================
def tokenize(text):
    """
    Hàm tách từ (Tokenization) đơn giản cho BM25.
    - Chuyển thành chữ thường
    - Lấy các từ chứa chữ và số (bỏ dấu câu)
    """
    if not text: return[]
    return re.findall(r'\w+', str(text).lower())

def load_queries(path):
    """Đọc file ground_truth_queries.json chứa danh sách câu hỏi"""
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy file query tại: {path}")
        return[]
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data[0], dict):
        return[item.get("query", item.get("original_query")) for item in data]
    return data

# ================= CLASS BM25 SEARCHER =================
class BM25Searcher:
    def __init__(self, data_path):
        self.corpus_metadata =[]
        tokenized_corpus =[]
        
        print(f"📦 Đang load dữ liệu từ {os.path.basename(data_path)} để build BM25 Index...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy file: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    # Gộp Title và Abstract để tìm kiếm
                    title = doc.get("title", "")
                    abstract = doc.get("abstract", "")
                    search_text = f"{title} {abstract}"
                    
                    # Tokenize và đưa vào kho
                    tokenized_corpus.append(tokenize(search_text))
                    
                    # Lưu lại thông tin (metadata) để lát trả kết quả
                    self.corpus_metadata.append({
                        "paper_id": str(doc.get("paper_id", "")),
                        "title": title,
                        "abstract": abstract,
                        "venue": doc.get("venue", ""),
                        "year": str(doc.get("publication_date") or doc.get("year") or ""),
                        "s2_url": doc.get("externalsid", {}).get("s2_url", "")
                    })
                except json.JSONDecodeError:
                    continue

        print(f"⚙️  Đang khởi tạo BM25Okapi Engine cho {len(tokenized_corpus)} bài báo...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ BM25 Index đã sẵn sàng!\n")

    def search(self, query: str, top_k: int = 20):
        """Thực hiện tìm kiếm theo BM25"""
        tokenized_query = tokenize(query)
        
        # Lấy điểm số của query đối với toàn bộ bài báo trong kho
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Ghép cặp (Index, Score) và loại bỏ các bài có điểm = 0 (Không khớp chữ nào)
        scored_docs =[(i, float(score)) for i, score in enumerate(doc_scores) if score > 0]
        
        # Sắp xếp giảm dần theo Score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Cắt lấy Top K
        top_results = scored_docs[:top_k]
        
        output =[]
        for rank, (doc_idx, score) in enumerate(top_results, start=1):
            meta = self.corpus_metadata[doc_idx]
            
            output.append({
                "rank": rank,
                "paper_id": meta["paper_id"],
                "title": meta["title"],
                "abstract": meta["abstract"],
                "score": round(score, 4),      # ĐIỂM BM25: Càng cao càng tốt
                "venue": meta["venue"],
                "year": meta["year"],
                "s2_url": meta["s2_url"],
                "search_type": "bm25",         # Đánh dấu nguồn gốc
                "is_relevant": None
            })
            
        return output

# ================= LUỒNG CHẠY CHÍNH =================
def main():
    queries = load_queries(QUERY_PATH)
    if not queries:
        return

    # 1. Khởi tạo Engine
    searcher = BM25Searcher(DATA_PATH)
    
    all_results =[]
    print(f"🚀 Bắt đầu truy xuất BM25 cho {len(queries)} queries (Top {TOP_K})...")

    # 2. Tìm kiếm từng câu
    for idx, query in enumerate(queries):
        print(f"   [{idx+1:02d}/{len(queries)}] Đang tìm (BM25): '{query}'")
        
        search_results = searcher.search(query, top_k=TOP_K)
        
        query_record = {
            "query_id": f"Q{idx+1:03d}",
            "query": query,
            "mode": "lexical_bm25",
            "retrieved_results": search_results
        }
        
        all_results.append(query_record)

    # 3. Lưu file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 HOÀN TẤT!")
    print(f"📂 Kết quả tìm kiếm BM25 đã được lưu tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()