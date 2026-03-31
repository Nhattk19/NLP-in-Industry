import os
import json
import re
import chromadb
from config import TOP_K, OUTPUT_PATH_RETRIEVED
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = "data/data_processed/final_cleaned_data.jsonl"  
CHROMA_PATH = os.path.join(BASE_DIR, "src", "chromadb", "chroma_store_abstracts")
COLLECTION_NAME = "papers_abstracts"

# Đường dẫn file Queries và thư mục Output
QUERY_PATH = os.path.join(BASE_DIR, "src", "ground_truth_queries.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "src", "results")

# Tạo thư mục lưu kết quả nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MasterSearchEngine:
    def __init__(self):
        """Khởi tạo toàn bộ các Engine (ChromaDB, BM25, FlashRank) vào RAM 1 lần duy nhất"""
        print("🚀 ĐANG KHỞI TẠO MASTER SEARCH ENGINE...")
        
        # 1. KHỞI TẠO CHROMADB (SEMANTIC)
        print("   🗄️ 1/3 Load ChromaDB (Semantic)...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction() # Dùng ONNX CPU
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=self.emb_fn)
        
        # 2. KHỞI TẠO FLASHRANK (RERANKER)
        print("   🧠 2/3 Load FlashRank (Reranker)...")
        self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=os.path.join(BASE_DIR, "src", "models_cache"))
        
        # 3. KHỞI TẠO BM25 (LEXICAL)
        print("   📖 3/3 Load BM25 (Lexical)...")
        self._init_bm25()
        
        print("✅ KHỞI TẠO HOÀN TẤT!\n")

    def _init_bm25(self):
        """Đọc file JSONL và xây dựng chỉ mục BM25"""
        self.corpus_metadata = {}
        tokenized_corpus =[]
        self.bm25_ids =[]
        
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    p_id = str(doc.get("paper_id"))
                    title = doc.get("title", "")
                    abstract = doc.get("abstract", "")
                    
                    # Lưu metadata
                    self.corpus_metadata[p_id] = {
                        "paper_id": p_id, "title": title, "abstract": abstract,
                        "venue": doc.get("venue", ""), "year": str(doc.get("year", "")),
                        "s2_url": doc.get("externalsid", {}).get("s2_url", "")
                    }
                    
                    # Tokenize
                    text = f"{title} {abstract}".lower()
                    tokens = re.findall(r'\w+', text)
                    tokenized_corpus.append(tokens)
                    self.bm25_ids.append(p_id)
                except json.JSONDecodeError: continue
                    
        self.bm25 = BM25Okapi(tokenized_corpus)

    # ---------------- LÕI TÌM KIẾM ----------------

    def _run_lexical(self, query, k):
        """Chạy BM25 thuần"""
        tokens = re.findall(r'\w+', query.lower())
        scores = self.bm25.get_scores(tokens)
        
        # Lấy Top K
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] == 0.0: break
            p_id = self.bm25_ids[idx]
            meta = self.corpus_metadata[p_id]
            meta_copy = meta.copy()
            meta_copy.update({"rank": rank, "score": round(float(scores[idx]), 4), "search_type": "lexical"})
            results.append(meta_copy)
        return results

    def _run_semantic(self, query, k):
        """Chạy ChromaDB thuần"""
        res = self.collection.query(query_texts=[query], n_results=k, include=["metadatas", "distances"])
        if not res["ids"] or not res["ids"][0]: return []
        
        results =[]
        for i in range(len(res["ids"][0])):
            meta = res["metadatas"][0][i]
            meta_copy = meta.copy()
            meta_copy.update({"rank": i+1, "score": float(res["distances"][0][i]), "search_type": "semantic"})
            results.append(meta_copy)
        return results

    def _run_rerank(self, query, results_list):
        """Chạy FlashRank để Rerank danh sách kết quả"""
        if not results_list: return []
        
        passages =[]
        for res in results_list:
            passages.append({
                "id": str(res["paper_id"]),
                "text": f"{res['title']}. {res['abstract']}",
                "original_data": res
            })
            
        rerank_req = RerankRequest(query=query, passages=passages)
        reranked = self.reranker.rerank(rerank_req)
        
        final = []
        for i, passage in enumerate(reranked):
            orig = passage["original_data"]
            orig["old_rank"] = orig["rank"]
            orig["rank"] = i + 1
            orig["rerank_score"] = round(float(passage["score"]), 4)
            final.append(orig)
        return final

    def _combine_rrf(self, lex_res, sem_res):
        """Thuật toán RRF trộn điểm BM25 và Vector"""
        rrf_scores = {}
        
        for rank, res in enumerate(lex_res):
            p_id = res['paper_id']
            if p_id not in rrf_scores: rrf_scores[p_id] = {"doc": res, "score": 0.0}
            rrf_scores[p_id]["score"] += 1.0 / (60 + rank + 1)
            
        for rank, res in enumerate(sem_res):
            p_id = res['paper_id']
            if p_id not in rrf_scores: rrf_scores[p_id] = {"doc": res, "score": 0.0}
            rrf_scores[p_id]["score"] += 1.0 / (60 + rank + 1)
            
        sorted_res = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        
        final =[]
        for i, item in enumerate(sorted_res):
            doc = item["doc"].copy()
            doc["rank"] = i + 1
            doc["rrf_score"] = round(item["score"], 4)
            doc["search_type"] = "hybrid"
            final.append(doc)
        return final

    # ---------------- GIAO DIỆN CHÍNH ----------------

    def search(self, query: str, k: int = 10, mode: str = "hybrid"):
        """
        Hàm Search Chính.
        mode: "lexical", "semantic", hoặc "hybrid"
        Trả về: kết quả (list) và tên file sẽ được lưu.
        """
        mode = mode.lower()
        
        if mode == "lexical":
            # 1. Lexical -> BM25
            results = self._run_lexical(query, k)
            filename = "results_bm25.json"
            
        elif mode == "semantic":
            # 2. Semantic -> ChromaDB -> Rerank
            raw_semantic = self._run_semantic(query, k * 2) # Lấy dư ra x2 để Rerank gạn lọc
            results = self._run_rerank(query, raw_semantic)[:k] # Rerank xong cắt lấy Top K
            filename = "rerank_results.json"
            
        elif mode == "hybrid":
            # 3. Hybrid -> (BM25 + ChromaDB) -> RRF -> Rerank
            pool_size = k * 2
            lex_res = self._run_lexical(query, pool_size)
            sem_res = self._run_semantic(query, pool_size)
            
            combined_rrf = self._combine_rrf(lex_res, sem_res)
            results = self._run_rerank(query, combined_rrf)[:k]
            filename = "hybrid_results.json"
            
        else:
            raise ValueError("Mode chỉ được là: lexical, semantic, hybrid")
            
        return results, filename


# ================= LUỒNG CHẠY CHÍNH (MAIN) =================
def load_queries(path):
    if not os.path.exists(path): return[]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data[0], dict):
        return [item.get("query", item.get("original_query")) for item in data]
    return data

def main():
    queries = load_queries(QUERY_PATH)
    if not queries:
        print(f"❌ Không tìm thấy queries tại {QUERY_PATH}")
        return

    # Khởi tạo Engine (Load model 1 lần duy nhất)
    engine = MasterSearchEngine()
    
    # Cấu hình test: Bạn muốn test mode nào thì đổi ở đây
    # ("lexical", "semantic", "hybrid")
    MODE_TO_RUN = "hybrid" 

    all_results =[]
    print(f"\n🔍 Bắt đầu tìm kiếm {len(queries)} queries bằng mode:[{MODE_TO_RUN.upper()}] ...")

    for idx, query in enumerate(queries):
        print(f"[{idx+1:02d}/{len(queries)}] Query: '{query}'")
        
        # Gọi hàm Search
        results, output_filename = engine.search(query, k=TOP_K, mode=MODE_TO_RUN)
        
        all_results.append({
            "query_id": f"Q{idx+1:03d}",
            "query": query,
            "mode": MODE_TO_RUN,
            "results": results
        })

    # Lưu file JSON
    final_output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f"\n🎉 HOÀN TẤT! Đã lưu kết quả của mode {MODE_TO_RUN.upper()} vào file:")
    print(f"📂 {final_output_path}")

if __name__ == "__main__":
    main()