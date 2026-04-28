import os
<<<<<<< HEAD
import sys
import json
import chromadb
from contextlib import redirect_stdout, redirect_stderr
from chromadb.utils import embedding_functions

# Disable ChromaDB telemetry to avoid telemetry errors
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

# ================= CẨU HÌNH ĐƯỜNG DẪN =================
# Try fulltext first, fall back to abstracts
CHROMA_PATH_FULLTEXT = "./data/chroma_store_fulltext"
CHROMA_PATH_ABSTRACTS = "./src/chromadb/chroma_store_abstracts"

# Use whichever exists
import os
if os.path.exists(CHROMA_PATH_FULLTEXT):
    CHROMA_PATH = CHROMA_PATH_FULLTEXT
    COLLECTION_NAME = "papers"
else:
    CHROMA_PATH = CHROMA_PATH_ABSTRACTS
    COLLECTION_NAME = "papers_abstracts"

=======
import json
import chromadb
from chromadb.utils import embedding_functions

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
CHROMA_PATH = "./src/chromadb/chroma_store_abstracts"
COLLECTION_NAME = "papers_abstracts"
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
QUERY_PATH = "./src/queries.json"
TOP_K = 20

OUTPUT_PATH_RETRIEVED = "./src/chromadb/retrieved_results.json"
<<<<<<< HEAD
# ================= KH_I T_O CHROMADB =================
print(f"[INIT] Connecting to ChromaDB at: {CHROMA_PATH}...")
=======
# ================= KHỞI TẠO CHROMADB =================
print(f"🗄️ Đang kết nối tới ChromaDB tại: {CHROMA_PATH} ...")
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Sử dụng Embedding mặc định của ChromaDB (all-MiniLM-L6-v2 bản ONNX - Không cần Torch)
emb_fn = embedding_functions.DefaultEmbeddingFunction()

try:
    # Nạp collection và gán hàm embedding vào
    collection = client.get_collection(
        name=COLLECTION_NAME, 
        embedding_function=emb_fn
    )
<<<<<<< HEAD
    print(f"[OK] Connected to Collection: '{COLLECTION_NAME}' (Total papers: {collection.count()})")
except Exception as e:
    print(f"[WARNING] Collection '{COLLECTION_NAME}' not found: {e}")
    print("[INFO] Creating empty collection (will return no results until indexed)")
    try:
        # Create empty collection with metadata
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=emb_fn
        )
        print(f"[OK] Created empty collection: '{COLLECTION_NAME}'")
    except Exception as e2:
        print(f"[ERROR] Failed to create collection: {e2}")
        print("[WARNING] Semantic search will not be available")
        collection = None

# ================= HELPER FUNCTIONS =================
def load_queries(path):
    """Read list of questions from JSON file"""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
=======
    print(f"✅ Đã kết nối Collection: '{COLLECTION_NAME}' (Tổng số bài: {collection.count()})")
except Exception as e:
    print(f"❌ Lỗi kết nối Collection: {e}")
    print("Vui lòng đảm bảo bạn đã chạy file Ingest để tạo database trước.")
    exit(1)

# ================= HÀM HỖ TRỢ =================
def load_queries(path):
    """Đọc danh sách câu hỏi từ file JSON"""
    if not os.path.exists(path):
        print(f"❌ Không tìm thấy file: {path}")
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
        return []
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
<<<<<<< HEAD
    # Support format [{"query": "..."}]
=======
    # Hỗ trợ format [{"query": "..."}]
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
    if data and isinstance(data[0], dict) and "query" in data[0]:
        return [item["query"] for item in data]
    # Hỗ trợ format list string bình thường ["query 1", "query 2"]
    return data

# ================= HÀM SEARCH CỐT LÕI =================
def search(query: str, top_k: int = 20):
    """
    Tìm kiếm Semantic Search trực tiếp bằng Raw Query và trích xuất TOÀN BỘ metadata
    """
<<<<<<< HEAD
    # Return empty if collection not available
    if collection is None:
        print("[WARN] ChromaDB collection not available, returning empty results")
        return []
    
    # CHROMA TỰ ĐỘNG CHUYỂN TEXT THÀNH VECTOR
    # Suppress ChromaDB internal logging (Add of existing embedding ID: ...)
    with redirect_stdout(open(os.devnull, 'w')), redirect_stderr(open(os.devnull, 'w')):
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["metadatas", "distances"] 
        )
=======
    # CHROMA TỰ ĐỘNG CHUYỂN TEXT THÀNH VECTOR
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["metadatas", "distances"] 
    )
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9

    if not results["ids"] or not results["ids"][0]:
        return []

    output = []
    ids = results["ids"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        meta = metas[i] if metas[i] else {}
        
        output.append({
            "rank": i + 1,
            "score": float(distances[i]), # Distance (khoảng cách), Càng thấp càng giống nhau
            
            # --- TRÍCH XUẤT TOÀN BỘ METADATA TỪ INGEST.PY ---
            "paper_id": meta.get("paper_id", ids[i]),
            "title": meta.get("title", "Unknown Title"),
            "abstract": meta.get("abstract", ""),
            "authors": meta.get("authors", ""),
            "venue": meta.get("venue", ""),
            "publication_date": meta.get("publication_date", ""), # Sửa 'year' thành 'publication_date'
            "is_survey": meta.get("is_survey", False),
            "citation_count": meta.get("citation_count", 0),
            "nlp_score": meta.get("nlp_score", 0),
            
            # External IDs & URLs
            "doi": meta.get("doi", ""),
            "arxiv": meta.get("arxiv", ""),
            "s2_url": meta.get("s2_url", ""),
            
            # Network (References & Citations)
            "reference_titles": meta.get("reference_titles", ""),
            "reference_ids": meta.get("reference_ids", ""),
            "citation_titles": meta.get("citation_titles", ""),
            "citation_ids": meta.get("citation_ids", ""),
            "reference_count": meta.get("reference_count", 0),
            "citation_network_count": meta.get("citation_network_count", 0),
            
            # --- Cờ đánh giá ---
            "is_relevant": None  # Chờ gán nhãn đánh giá (1: Đúng, 0: Sai)
        })

    return output

# ================= LUỒNG CHẠY CHÍNH =================
def main():
    queries = load_queries(QUERY_PATH)
    if not queries:
        return

    all_results = []
    print(f"\n🚀 Bắt đầu Semantic Search cho {len(queries)} queries (Top {TOP_K})...")

    for idx, query in enumerate(queries):
        print(f"[{idx+1:02d}/{len(queries)}] Đang tìm: '{query}'")
        
        try:
            # Lấy kết quả tìm kiếm
            search_results = search(query, top_k=TOP_K)
            
            # Gói kết quả vào format chuẩn để sau này chạy Evaluation
            query_record = {
                "query_id": f"Q{idx+1:03d}",
                "query": query,
                "results": search_results
            }
            
            all_results.append(query_record)
            
        except Exception as e:
<<<<<<< HEAD
            print(f"   ERROR: Error searching query '{query}': {e}")

    # Ensure storage directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH_RETRIEVED), exist_ok=True)

    # Save JSON results
=======
            print(f"   ❌ Lỗi khi tìm kiếm query '{query}': {e}")

    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(OUTPUT_PATH_RETRIEVED), exist_ok=True)

    # Lưu file JSON kết quả
>>>>>>> 0fbc897ed8e0703e70ccfb0b334045576a9a10b9
    with open(OUTPUT_PATH_RETRIEVED, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 HOÀN TẤT!")
    print(f"📂 Kết quả tìm kiếm đã được lưu tại: {OUTPUT_PATH_RETRIEVED}")

if __name__ == "__main__":
    main()
