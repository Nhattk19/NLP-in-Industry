import os
import json
import re
from rank_bm25 import BM25Okapi
from src.config import DATA_PATH, QUERY_PATH, TOP_K, OUTPUT_PATH_BM25

<<<<<<< HEAD
# ================= HELPER FUNCTIONS =================
def tokenize(text):
    """Simple word tokenization for BM25 (lowercase, remove punctuation)"""
=======
# ================= HÀM HỖ TRỢ =================
def tokenize(text):
    """Tách từ đơn giản cho BM25 (chuyển chữ thường, bỏ dấu câu)"""
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
    if not text: return[]
    return re.findall(r'\w+', str(text).lower())

def load_queries(path):
<<<<<<< HEAD
    """Read queries.json file"""
    if not os.path.exists(path):
        print(f"X File not found: {path}")
=======
    """Đọc file queries.json"""
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy file query tại: {path}")
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
        return[]
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data[0], dict):
        return[item.get("query", item.get("original_query")) for item in data]
    return data

<<<<<<< HEAD
# ================= BM25 SEARCHER CLASS =================
=======
# ================= CLASS BM25 SEARCHER =================
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
class BM25Searcher:
    def __init__(self, data_path):
        self.corpus_metadata = []
        tokenized_corpus =[]
        
<<<<<<< HEAD
        print(f"[BM25] Loading data from {os.path.basename(data_path)} to build index...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
=======
        print(f"📦 [BM25] Đang load dữ liệu từ {os.path.basename(data_path)} để build Index...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy file: {data_path}")
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
<<<<<<< HEAD
                    # Combine Title and Abstract for search text
=======
                    # Gộp Title và Abstract để làm văn bản tìm kiếm
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
                    title = doc.get("title", "")
                    abstract = doc.get("abstract", "")
                    search_text = f"{title} {abstract}"
                    
<<<<<<< HEAD
                    # Tokenize and add to corpus
                    tokenized_corpus.append(tokenize(search_text))
                    
                    # Store only essential metadata to optimize memory
=======
                    # Tokenize và đưa vào kho (corpus)
                    tokenized_corpus.append(tokenize(search_text))
                    
                    # Chỉ lưu đúng 3 thông tin cần thiết vào RAM để tối ưu bộ nhớ
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
                    self.corpus_metadata.append({
                        "paper_id": str(doc.get("paper_id", "")),
                        "title": title,
                        "abstract": abstract
                    })
                except json.JSONDecodeError:
                    continue

<<<<<<< HEAD
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
=======
        print(f"⚙️  [BM25] Khởi tạo BM25 Engine cho {len(tokenized_corpus)} bài báo...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ [BM25] Index đã sẵn sàng!\n")

    def search(self, query: str, top_k: int = 20):
        """Thực hiện tìm kiếm lexical"""
        tokenized_query = tokenize(query)
        
        # Chấm điểm toàn bộ document
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Chỉ giữ lại bài có điểm > 0
        scored_docs =[(i, float(score)) for i, score in enumerate(doc_scores) if score > 0]
        
        # Sắp xếp giảm dần theo điểm BM25
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Cắt lấy Top K
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
        top_results = scored_docs[:top_k]
        
        output =[]
        for rank, (doc_idx, score) in enumerate(top_results, start=1):
            meta = self.corpus_metadata[doc_idx]
            
<<<<<<< HEAD
            # Format as 5-field output
=======
            # FORMAT CHUẨN 5 TRƯỜNG ĐÚNG NHƯ YÊU CẦU
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
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