import os
import json
from config import QUERY_PROCESSING_MODEL, DEVICE, TOP_K, OUTPUT_PATH_RETRIEVED
import chromadb
from chromadb.utils import embedding_functions

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
CHROMA_PATH = "./src/chromadb/chroma_store_abstracts"
COLLECTION_NAME = "papers_abstracts"

QUERY_PATH = "./src/queries.json"
# ================= KHỞI TẠO CHROMADB =================
print(f"🗄️ Đang kết nối tới ChromaDB tại: {CHROMA_PATH} ...")
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Sử dụng Embedding mặc định của ChromaDB (all-MiniLM-L6-v2 bản ONNX - Không cần Torch)
emb_fn = embedding_functions.DefaultEmbeddingFunction()

try:
    # Nạp collection và gán hàm embedding vào
    collection = client.get_collection(
        name=COLLECTION_NAME, 
        embedding_function=emb_fn
    )
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
        return[]
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Hỗ trợ format [{"query": "..."}]
    if data and isinstance(data[0], dict) and "query" in data[0]:
        return [item["query"] for item in data]
    # Hỗ trợ format list string bình thường ["query 1", "query 2"]
    return data

# ================= HÀM SEARCH CỐT LÕI =================
def search(query: str, top_k: int = 20):
    """
    Tìm kiếm Semantic Search trực tiếp bằng Raw Query
    """
    # CHROMA TỰ ĐỘNG CHUYỂN TEXT THÀNH VECTOR
    results = collection.query(
        query_texts=[query], # Truyền thẳng câu text gốc vào đây
        n_results=top_k,
        include=["metadatas", "distances"] 
    )

    if not results["ids"] or not results["ids"][0]:
        return[]

    output = []
    ids = results["ids"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        meta = metas[i] if metas[i] else {}
        
        output.append({
            "rank": i + 1,
            "paper_id": meta.get("paper_id", ids[i]),
            "title": meta.get("title", "Unknown Title"),
            "abstract": meta.get("abstract", ""),
            "score": float(distances[i]), # Distance (khoảng cách), Càng thấp càng giống nhau
            "venue": meta.get("venue", ""),
            "year": meta.get("year", ""),
            "s2_url": meta.get("s2_url", ""),
            "is_relevant": None  # Chờ gán nhãn đánh giá (1: Đúng, 0: Sai)
        })

    return output

# ================= LUỒNG CHẠY CHÍNH =================
def main():
    queries = load_queries(QUERY_PATH)
    if not queries:
        return

    all_results =[]
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
            print(f"   ❌ Lỗi khi tìm kiếm query '{query}': {e}")

    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(OUTPUT_PATH_RETRIEVED), exist_ok=True)

    # Lưu file JSON kết quả
    with open(OUTPUT_PATH_RETRIEVED, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 HOÀN TẤT!")
    print(f"📂 Kết quả tìm kiếm đã được lưu tại: {OUTPUT_PATH_RETRIEVED}")

if __name__ == "__main__":
    main()