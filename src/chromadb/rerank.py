import os
import json
from flashrank import Ranker, RerankRequest

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH_RETRIEVED = "./src/chromadb/retrieved_results.json"
OUTPUT_PATH_RERANKED = "./src/chromadb/reranked_results.json"

class PaperReranker:
    def __init__(self, model_name="ms-marco-MiniLM-L-12-v2"):
        """
        Khởi tạo mô hình Reranker.
        ms-marco-MiniLM-L-6-v2 là model tiêu chuẩn, nhẹ (~30MB) và cực kỳ chính xác cho RAG.
        Chạy hoàn toàn bằng ONNX (CPU), KHÔNG CẦN TORCH.
        """
        print(f"🚀 Đang tải mô hình Reranker ({model_name})...")
        # Tự động tải model (chỉ lần đầu) và lưu vào cache
        self.ranker = Ranker(model_name=model_name, cache_dir=os.path.join(BASE_DIR, "src", "models_cache"))
        print("✅ Khởi tạo Reranker thành công!\n")

    def rerank_results(self, query: str, retrieved_results: list):
        """
        Thực hiện Rerank một danh sách các kết quả dựa trên câu query.
        """
        if not retrieved_results:
            return []

        # 1. Chuyển đổi format kết quả Search sang format mà FlashRank yêu cầu
        passages = []
        for res in retrieved_results:
            full_text = f"{res.get('title', '')}. {res.get('abstract', '')}"
            passages.append({
                "id": str(res.get("paper_id")),
                "text": full_text,
                "original_data": res # Giữ lại toàn bộ metadata gốc (bao gồm cả authors, doi...)
            })

        # 2. Đóng gói Request
        rerank_request = RerankRequest(query=query, passages=passages)

        # 3. Thực hiện Rerank
        reranked_passages = self.ranker.rerank(rerank_request)

        # 4. Format lại output (Giữ nguyên metadata, chỉ cập nhật rank và score)
        final_output = []
        for i, passage in enumerate(reranked_passages):
            original = passage["original_data"].copy() # Tạo bản sao để không thay đổi trực tiếp
            
            # Lưu lại thông tin cũ để so sánh
            original["old_rank"] = original["rank"]
            original["old_score"] = original["score"]
            
            # Cập nhật thông tin mới từ Reranker
            original["rank"] = int(i + 1)
            original["rerank_score"] = round(float(passage["score"]), 4) # Thay score cũ bằng rerank_score
            
            final_output.append(original)

        return final_output



# ================= LUỒNG CHẠY CHÍNH ĐỂ TEST FILE JSON =================
def main():
    # 1. Kiểm tra file input
    if not os.path.exists(OUTPUT_PATH_RETRIEVED):
        print(f"❌ Không tìm thấy file kết quả search tại: {OUTPUT_PATH_RETRIEVED}")
        print("   Vui lòng chạy file retrieve.py trước!")
        return

    # 2. Đọc file kết quả Search
    with open(OUTPUT_PATH_RETRIEVED, "r", encoding="utf-8") as f:
        search_data = json.load(f)

    # 3. Khởi tạo Reranker
    reranker = PaperReranker()
    all_reranked_data =[]

    # 4. Duyệt qua từng Query trong file JSON để Rerank
    for item in search_data:
        query = item["query"]
        old_results = item["results"]
        
        print(f"🔍 Đang Rerank cho Query: '{query}'")
        print(f"   - Số lượng bài báo trước Rerank: {len(old_results)}")

        # Chạy thuật toán Rerank
        new_results = reranker.rerank_results(query, old_results)

        # In ra màn hình để thấy sự thay đổi kỳ diệu
        print("   => TOP 3 BÀI BÁO SAU KHI RERANK:")
        for res in new_results[:3]:
            # Đánh dấu mũi tên lên/xuống nếu rank thay đổi
            rank_diff = res['old_rank'] - res['rank']
            trend = "🟢 TĂNG" if rank_diff > 0 else ("🔴 GIẢM" if rank_diff < 0 else "⚪ GIỮ NGUYÊN")
            
            print(f"      [{res['rank']}] (Cũ: {res['old_rank']} - {trend}) | Điểm mới: {res['rerank_score']} | {res['title']}")

        # Lưu lại vào danh sách tổng
        all_reranked_data.append({
            "query": query,
            "mode": item.get("mode", "hybrid"),
            "reranked": True,
            "results": new_results
        })

    # 5. Ghi ra file JSON mới
    with open(OUTPUT_PATH_RERANKED, "w", encoding="utf-8") as f:
        json.dump(all_reranked_data, f, ensure_ascii=False, indent=4)

    print(f"\n📂 Đã lưu kết quả Rerank chi tiết vào: {OUTPUT_PATH_RERANKED}")

if __name__ == "__main__":
    main()
