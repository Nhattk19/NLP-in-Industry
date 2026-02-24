import os
import json
import gzip
import re
import time
import requests
import config  # Import từ cùng thư mục src

# --- CẤU HÌNH ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Đọc API key từ file api.txt
API_KEY_FILE = os.path.join(CURRENT_DIR, "api.txt")
try:
    with open(API_KEY_FILE, "r", encoding="utf-8") as f:
        S2_API_KEY = f.read().strip()
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file api.txt")
    S2_API_KEY = ""

OUTPUT_FILE = os.path.join(config.DATA_RAW_DIR, "1_candidates.jsonl")
DATASETS_BASE_URL = "https://api.semanticscholar.org/datasets/v1"

def get_download_links(dataset_name="papers"):
    """Lấy danh sách URL download các phần (parts) của dataset."""
    if len(S2_API_KEY) > 4:
        print(f"🔑 Đang dùng API Key: ...{S2_API_KEY[-4:]}")
    else:
        print("⚠️ CẢNH BÁO: API Key có vẻ quá ngắn hoặc rỗng!")

    headers = {"x-api-key": S2_API_KEY}
    
    print(f"📡 Đang lấy danh sách Releases của '{dataset_name}'...")
    try:
        # 1. Lấy danh sách các bản release
        release_url = f"{DATASETS_BASE_URL}/release"
        resp = requests.get(release_url, headers=headers)
        
        if resp.status_code != 200:
            print(f"❌ Lỗi lấy release: {resp.status_code} - {resp.text}")
            return []
        
        releases = resp.json()
        if not releases:
            print("❌ Không tìm thấy release nào.")
            return []
            
        latest_release_id = releases[-1]
        print(f"✨ Release mới nhất: {latest_release_id}")

        print("⏳ Đang nghỉ 2 giây để tránh Rate Limit...")
        time.sleep(2) 

        # 2. Lấy link download cho release đó
        links_url = f"{DATASETS_BASE_URL}/release/{latest_release_id}/dataset/{dataset_name}"
        resp_links = requests.get(links_url, headers=headers)
        
        if resp_links.status_code != 200:
            print(f"❌ Lỗi lấy download links: {resp_links.status_code} - {resp_links.text}")
            return []
        
        data = resp_links.json()
        files = data.get("files", [])
        
        print(f"📦 Tìm thấy {len(files)} phần (parts) cần xử lý.")
        return files 
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []

def is_loose_candidate(doc):
    """
    Enhanced filter: check title and venue
    Returns True if paper is likely NLP-related
    """
    title = str(doc.get("title", "")).lower()
    venue = str(doc.get("venue", "")).lower()
    
    # Combine text for checking
    combined_text = f"{title} . {venue} "
    
    if not title: return False
    
    # 1. Kill Switch (Loại ngay ngành khác)
    for pattern in config.HARD_EXCLUDE:
        if re.search(pattern, combined_text):
            return False
    
    # 2. Auto-accept for Pure NLP Venues
    for pattern in config.PURE_NLP_VENUES:
        if re.search(pattern, venue):
            return True
            
    # 3. Match count for title 
    match_count = 0
    for pattern in config.LOOSE_KEYWORDS:
        if re.search(pattern, title):
            match_count += 1
            
    # Nếu là General AI Venue, chỉ cần 1 match là đủ cho qua vòng loose
    is_general_ai = any(re.search(p, venue) for p in config.GENERAL_AI_VENUES)
    if is_general_ai and match_count >= 1.0:
        return True
        
    return match_count >= 2.0


def detect_is_survey(title, publication_types):
    """Detect if paper is a survey/review paper"""
    text = f"{title}".lower()
    
    # Check title and for survey/review keywords
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

def format_to_clean_output(doc):
    """Format raw S2 dataset doc to clean output format"""
    # Extract paper ID (ưu tiên CorpusId)
    external_ids = doc.get("externalids") or doc.get("externalIds") or {}
    paper_id = str(doc.get("corpusid") or external_ids.get("CorpusId") or "unknown")
    
    # Extract authors 
    authors_data = doc.get("authors", [])
    authors = []
    if isinstance(authors_data, list):
        for author in authors_data:
            if isinstance(author, dict):
                name = author.get("name")
                author_id = author.get("authorId")
                if name:
                    authors.append({
                        "name": name,
                        "id": author_id
                    })
            elif isinstance(author, str):
                # Fallback nếu author là string thuần
                authors.append({
                    "name": author,
                    "id": None
                })
    
    # Extract external IDs - tạo link đầy đủ
    arxiv_id = external_ids.get("ArXiv")
    acl_id = external_ids.get("ACL")
    doi_id = external_ids.get("DOI")
    
    externalsid = {
        "arxiv": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "acl": f"https://aclanthology.org/{acl_id}" if acl_id else None,
        "doi": f"https://doi.org/{doi_id}" if doi_id else None,
        "s2_url": doc.get("url")
    }
    
    # Network sẽ rỗng ở bước 1, chỉ populate ở bước 2 (2_map.py)
    network = {
        "references": [],
        "citations": []
    }
    
    # Detect if survey
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")  # Thường rỗng ở Dataset API
    publication_types = doc.get("publicationtypes")
    is_survey = detect_is_survey(title,  publication_types)
    
    # Build clean output
    output = {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "venue": doc.get("venue", ""),
        "publication_date": doc.get("publicationdate", doc.get("year")),
        "is_survey": is_survey,
        "externalsid": externalsid,
        "network": network,
        "citation_count": doc.get("citationcount", 0)
    }
    
    return output

def stream_and_filter(download_links):
    print(f"🚀 [Bước 1] Bắt đầu Loose Filter... Lưu vào: {OUTPUT_FILE}")
    total_count = 0
    
    # Mở file output 1 lần duy nhất ở mode 'write'
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, url in enumerate(download_links):
            # API trả về list string URL trực tiếp
            if isinstance(url, dict):
                url = url.get("url", url)
            print(f"\n⬇️  Processing Part {i+1}/{len(download_links)}...")
            
            try:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    dctx = gzip.GzipFile(fileobj=r.raw)
                    
                    local_count = 0
                    for line in dctx:
                        try:
                            json_str = line.decode('utf-8')
                            doc = json.loads(json_str)
                            
                            if is_loose_candidate(doc):
                                # Format sang cấu trúc sạch trước khi lưu
                                clean_doc = format_to_clean_output(doc)
                                f_out.write(json.dumps(clean_doc, ensure_ascii=False) + "\n")
                                local_count += 1
                        except json.JSONDecodeError:
                            continue
                            
                    print(f"   ✅ Part {i+1} xong. Tìm thấy {local_count} papers.")
                    total_count += local_count

            except Exception as e:
                print(f"   ⚠️ Lỗi xử lý file {i}: {e}")
                time.sleep(2)

    print(f"\n✅ [DONE] Tìm thấy {total_count} papers tiềm năng.")

def main():
    if not S2_API_KEY:
        print("⛔ Vui lòng điền API KEY vào file api.txt!")
        return
        
    links = get_download_links("papers")
    
    if links:
        stream_and_filter(links[19:20])  # edit

if __name__ == "__main__":
    main()