import os
import json
import re
import time
import requests
from tqdm import tqdm
import config  # Import config từ cùng thư mục src

# --- CẤU HÌNH ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
API_KEY_FILE = os.path.join(CURRENT_DIR, "api.txt")

# Input từ bước 1 (Loose Filter)
CANDIDATE_FILE = os.path.join(config.DATA_RAW_DIR, "1_candidates.jsonl")

# Output thay đổi theo yêu cầu: Lưu vào data_raw, tên 2_map.jsonl
OUTPUT_FILE = os.path.join(config.DATA_RAW_DIR, "2_map.jsonl")

# Semantic Scholar Graph API
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
BATCH_SIZE = 500  # API cho phép batch tối đa 500 papers
RATE_LIMIT_DELAY = 3  # Delay giữa các batch (seconds)

def get_api_key():
    """Đọc API key từ file api.txt"""
    try:
        with open(API_KEY_FILE, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                print(f"🔑 Đã load API Key: ...{key[-4:]}")
            return key
    except FileNotFoundError:
        print("⚠️ Không tìm thấy file api.txt")
        return ""

def enrich_papers_batch(paper_ids, api_key, retry=True):
    """
    Lấy abstract và citations cho 1 batch papers qua S2 Graph API.
    """
    if not paper_ids:
        return {}
    
    headers = {"x-api-key": api_key} if api_key else {}
    
    # Fields cần lấy từ API: Lấy cả references, citations VÀ externalIds của chúng
    fields = "paperId,abstract,externalIds,references.paperId,references.title,references.externalIds,citations.paperId,citations.title,citations.externalIds"
    
    url = f"{S2_API_BASE}/paper/batch"
    params = {"fields": fields}
    payload = {"ids": paper_ids}
    
    # DEBUG: In ra sample IDs
    print(f"   🔍 Sample IDs gửi tới API: {paper_ids[:3]}")
    
    try:
        response = requests.post(
            url, 
            params=params, 
            json=payload, 
            headers=headers,
            timeout=30
        )
        
        print(f"   📡 API Response Code: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"   📊 API trả về {len(results)} results")
            
            # Count None results
            none_count = sum(1 for r in results if r is None)
            success_count = len(results) - none_count
            print(f"   ✅ Thành công: {success_count}/{len(results)} papers")
            if none_count > 0:
                print(f"   ⚠️  Không tìm thấy: {none_count} papers")
            
            enriched = {}
            for i, paper in enumerate(results):
                if paper:  # paper có thể là None nếu API không tìm thấy
                    original_id = paper_ids[i]
                    has_abstract = bool(paper.get("abstract"))
                    has_refs = bool(paper.get("references"))
                    has_cits = bool(paper.get("citations"))
                    
                    # DEBUG: In thông tin paper đầu tiên tìm thấy
                    if len(enriched) == 0:
                        print(f"   📝 Sample: abstract={has_abstract}, refs={len(paper.get('references') or [])}, cits={len(paper.get('citations') or [])}")
                    
                    enriched[original_id] = {
                        "abstract": paper.get("abstract", ""),
                        "externalIds": paper.get("externalIds", {}),
                        "references": (paper.get("references") or [])[:20], # Các bài báo mà bài này trích dẫn
                        "citations": (paper.get("citations") or [])[:20]    # Các bài báo khác trích dẫn bài này
                    }
            return enriched
            
        elif response.status_code == 429:
            print(f"⚠️ Rate limit hit. Chờ 10 giây...")
            time.sleep(10)
            if retry:
                print("   🔄 Retry lại...")
                return enrich_papers_batch(paper_ids, api_key, retry=False)
            return {}
        else:
            print(f"⚠️ API Error {response.status_code}: {response.text[:200]}")
            return {}
            
    except Exception as e:
        print(f"❌ Exception khi gọi API: {e}")
        import traceback
        traceback.print_exc()
        return {}

def format_paper_id_for_api(paper_data):
    """
    Format paper ID cho S2 API từ format mới (clean format).
    Input đã có paper_id và externalsid.
    S2 API cần prefix: CorpusId:, ArXiv:, ACL:
    """
    # Đọc từ format mới
    paper_id = paper_data.get("paper_id", "")
    externalsid = paper_data.get("externalsid", {})
    
    # Priority 1: CorpusId (numeric, short) - PHẢI CÓ PREFIX!
    if paper_id and paper_id != "unknown":
        # Nếu paper_id là số thuần túy (CorpusId) → thêm prefix
        if paper_id.isdigit():
            return f"CorpusId:{paper_id}"
    
    # Priority 2: ArXiv ID
    if externalsid.get("arxiv"):
        return f"ArXiv:{externalsid['arxiv']}"
    
    # Priority 3: ACL ID
    if externalsid.get("acl"):
        return f"ACL:{externalsid['acl']}"
    
    # Fallback: Dùng paper_id với prefix CorpusId
    if paper_id and paper_id != "unknown":
        return f"CorpusId:{paper_id}"
    
    return None

def extract_consistent_id(paper_data):
    """
    Extract consistent ID format from paper data.
    Priority: CorpusId (numeric) > ArXiv > ACL > paperId
    Returns a short, consistent ID format.
    """
    external_ids = paper_data.get("externalIds", {})
    paper_id = paper_data.get("paperId")
    
    # Priority 1: CorpusId (numeric, short)
    if external_ids and "CorpusId" in external_ids:
        return str(external_ids["CorpusId"])
    
    # Priority 2: ArXiv ID
    if external_ids and "ArXiv" in external_ids:
        return external_ids["ArXiv"]
    
    # Priority 3: ACL ID
    if external_ids and "ACL" in external_ids:
        return external_ids["ACL"]
    
    # Fallback: Use paperId (long hash)
    return paper_id if paper_id else "unknown"

def detect_is_survey(title, abstract, publication_types):
    """Detect if paper is a survey/review paper"""
    text = f"{title} {abstract}".lower()
    
    # Check title and abstract for survey/review keywords
    survey_keywords = [
        r'\bsurvey\b', r'\breview\b', r'\boverview\b', 
        r'\btutorial\b', r'\bstate of the art\b', r'\bstate-of-the-art\b'
    ]
    
    for pattern in survey_keywords:
        if re.search(pattern, text):
            return True
    
    # Check publication types
    if isinstance(publication_types, list):
        for ptype in publication_types:
            if isinstance(ptype, str) and ptype.lower() in ['review', 'survey']:
                return True
    
    return False

def format_paper_output(paper):
    """Paper đã ở format sạch rồi, chỉ cần return luôn (không cần format lại)"""
    # Input từ bước 1 đã đúng format, chỉ cần đảm bảo is_survey được update
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    
    # Re-detect is_survey sau khi có abstract mới (nếu có)
    if abstract and len(abstract) > 50:
        publication_types = paper.get("publicationtypes")  # Có thể không có field này
        paper["is_survey"] = detect_is_survey(title, abstract, publication_types)
    
    # Return paper as-is (đã đúng format từ bước 1)
    return paper

def main():
    api_key = get_api_key()
    if not api_key:
        print("⚠️ CẢNH BÁO: Không có API key. Tốc độ sẽ bị giới hạn nhiều.")
        confirm = input("   -> Tiếp tục? (y/n): ")
        if confirm.lower() != 'y':
            return
    
    # 1. Load Candidates
    print("\n⏳ [1/3] Đang load danh sách Candidates từ bước 1...")
    if not os.path.exists(CANDIDATE_FILE):
        print(f"❌ Không tìm thấy file: {CANDIDATE_FILE}")
        return

    candidates = []
    with open(CANDIDATE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                candidates.append(json.loads(line))
            except:
                continue
    
    print(f"✅ Đã load {len(candidates)} papers.")
    
    # 2. Stream API để map data
    print(f"\n🌊 [2/3] Đang gọi S2 API để map Abstract & Citations...")
    enriched_count = 0
    total_batches = (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE
    
    # Thống kê chi tiết
    stats = {
        "total_papers": len(candidates),
        "api_found": 0,
        "api_not_found": 0,
        "has_abstract": 0,
        "has_references": 0,
        "has_citations": 0
    }
    
    for batch_idx in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        print(f"\n📦 Batch {batch_num}/{total_batches} ({len(batch)} papers)...")
        
        paper_ids = []
        id_to_idx = {} 
        
        for i, paper in enumerate(batch):
            api_id = format_paper_id_for_api(paper)
            if api_id:  # Chỉ thêm nếu có ID hợp lệ
                paper_ids.append(api_id)
                id_to_idx[api_id] = i
        
        # Lấy data từ API
        enriched_data = enrich_papers_batch(paper_ids, api_key)
        
        stats["api_found"] += len(enriched_data)
        stats["api_not_found"] += len(paper_ids) - len(enriched_data)
        
        # Map dữ liệu mới vào list candidates hiện tại
        for api_id, data in enriched_data.items():
            idx = id_to_idx.get(api_id)
            if idx is not None:
                paper = batch[idx]
                
                # 2.1 Map Abstract
                if data.get("abstract"):
                    paper["abstract"] = data["abstract"]
                    enriched_count += 1
                    stats["has_abstract"] += 1
                
                # 2.1b Cập nhật externalsid từ API (chính xác hơn)
                if data.get("externalIds"):
                    # Merge với externalsid cũ (format mới)
                    old_external = paper.get("externalsid", {})
                    api_external = data["externalIds"]
                    
                    # Convert IDs to full URLs
                    arxiv_id = api_external.get("ArXiv")
                    acl_id = api_external.get("ACL")
                    doi_id = api_external.get("DOI")
                    
                    paper["externalsid"] = {
                        "arxiv": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else old_external.get("arxiv"),
                        "acl": f"https://aclanthology.org/{acl_id}" if acl_id else old_external.get("acl"),
                        "doi": f"https://doi.org/{doi_id}" if doi_id else old_external.get("doi"),
                        "s2_url": old_external.get("s2_url")  # Giữ nguyên từ bước 1
                    }
                
                # 2.2 Map References vào network.references
                if data.get("references"):
                    network = paper.get("network", {})
                    network["references"] = [
                        {"title": ref.get("title"), "id": extract_consistent_id(ref)} 
                        for ref in (data.get("references") or []) if ref
                    ]
                    paper["network"] = network
                    stats["has_references"] += 1
                
                # 2.3 Map Citations vào network.citations
                if data.get("citations"):
                    network = paper.get("network", {})
                    network["citations"] = [
                        {"title": cit.get("title"), "id": extract_consistent_id(cit)} 
                        for cit in (data.get("citations") or []) if cit
                    ]
                    paper["network"] = network
                    paper["citation_count"] = len(data.get("citations") or [])
                    stats["has_citations"] += 1
        
        print(f"   ✅ Map thành công: {len(enriched_data)}/{len(batch)} papers")
        
        if batch_num < total_batches:
            time.sleep(RATE_LIMIT_DELAY)
    
    print(f"\n📊 Tổng kết API Mapping:")
    print(f"   - Papers gửi đi:        {stats['total_papers']}")
    print(f"   - API tìm thấy:       {stats['api_found']} (✅ {stats['api_found']/stats['total_papers']*100:.1f}%)")
    print(f"   - API không tìm thấy: {stats['api_not_found']} (⚠️  {stats['api_not_found']/stats['total_papers']*100:.1f}%)")
    print(f"   - Có Abstract:        {stats['has_abstract']}")
    print(f"   - Có References:      {stats['has_references']}")
    print(f"   - Có Citations:       {stats['has_citations']}")
    
    # 3. Lưu kết quả ra file 2_map.jsonl trong data_raw
    print(f"\n💾 [3/3] Đang lưu kết quả...")
    papers_with_abstract = 0
    papers_with_citations = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for paper in tqdm(candidates, desc="Lưu file"):
            # Format và lưu theo cấu trúc mới
            formatted_paper = format_paper_output(paper)
            f_out.write(json.dumps(formatted_paper, ensure_ascii=False) + "\n")
            
            # Chỉ đếm thống kê
            if formatted_paper.get("abstract") and len(formatted_paper.get("abstract", "")) > 50:
                papers_with_abstract += 1
            if formatted_paper.get("network", {}).get("citations") or formatted_paper.get("network", {}).get("references"):
                papers_with_citations += 1
    
    print(f"\n🎉 HOÀN TẤT MAP DATA!")
    print(f"📂 Output lưu tại: {OUTPUT_FILE}")
    print(f"\n📊 THỐNG KÊ CUỐI CÙNG:")
    print(f"   - Tổng papers:             {len(candidates)}")
    print(f"   - Có abstract (>50 chars): {papers_with_abstract} ({papers_with_abstract/len(candidates)*100:.1f}%)")
    print(f"   - Có citations/references:  {papers_with_citations} ({papers_with_citations/len(candidates)*100:.1f}%)")
    print(f"\n💡 Lưu ý: Nếu có papers không có abstract/citations, có thể:")
    print(f"   1. Paper đó không có trong Graph API (chỉ có trong Dataset)")
    print(f"   2. Paper thực sự không có abstract")
    print(f"   3. Paper chưa được trích dẫn bởi ai")

if __name__ == "__main__":
    os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
    main()