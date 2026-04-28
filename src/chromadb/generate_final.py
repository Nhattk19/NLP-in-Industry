import os
import json

OUTPUT_PATH_RERANKED = "./src/chromadb/reranked_results.json"
OUTPUT_PATH_CHROMADB = "./src/chromadb/results.json"
def main():
    """Clean and normalize reranked search results into final ChromaDB format."""
    print("🧹 BẮT ĐẦU DỌN DẸP VÀ CHUẨN HÓA KẾT QUẢ TÌM KIẾM...")

    # 1. Kiểm tra file input
    if not os.path.exists(OUTPUT_PATH_RERANKED):
        print(f"❌ Không tìm thấy file kết quả Rerank tại: {OUTPUT_PATH_RERANKED}")
        print("   Vui lòng chạy file rerank.py trước!")
        return

    # 2. Đọc file dữ liệu Reranked
    with open(OUTPUT_PATH_RERANKED, "r", encoding="utf-8") as f:
        reranked_data = json.load(f)

    final_data =[]

    # 3. Chuẩn hóa dữ liệu 
    for item in reranked_data:
        query = item.get("query", "")
        old_results = item.get("results", [])
        
        clean_results =[]
        for res in old_results:
            clean_res = {
                "rank": res.get("rank"),
                "paper_id": res.get("paper_id"),
                "title": res.get("title"),
                "abstract": res.get("abstract"),
                "score": res.get("rerank_score", res.get("score", 0.0)) 
            }
            clean_results.append(clean_res)

        # Gói lại thành format chuẩn
        final_data.append({
            "query": query,
            "results": clean_results
        })

    # 4. Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(OUTPUT_PATH_CHROMADB), exist_ok=True)

    # 5. Ghi ra file JSON Final
    with open(OUTPUT_PATH_CHROMADB, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"✅ Đã dọn dẹp xong {len(final_data)} queries!")
    print("📂 File kết quả chuẩn cuối cùng (Final Format) được lưu tại:")
    print(f"   -> {OUTPUT_PATH_CHROMADB}")

if __name__ == "__main__":
    main()
