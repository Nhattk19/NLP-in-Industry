import os
import json
import re
from collections import Counter, defaultdict

# --- CẤU HÌNH ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # data/
INPUT_FILE = os.path.join(BASE_DIR, "data_raw", "3_final_nlp_papers3040.jsonl") # Sửa lại tên file đúng của bạn
REPORT_FILE = os.path.join(BASE_DIR, "data_raw", "statistics_report.json")   # File lưu kết quả chi tiết

def normalize_text(text):
    """Chuẩn hóa text để so sánh trùng lặp"""
    if not text: return ""
    text = str(text).lower().strip()
    return re.sub(r'[^a-z0-9]', '', text)

def is_vietnamese_related(doc):
    """Kiểm tra bài báo liên quan đến Tiếng Việt"""
    keywords = [
        r'\bvietnamese\b', r'\bvietnam\b', r'\btieng viet\b', r'\btiếng việt\b',
        r'\bphobert\b', r'\bvi-bert\b', r'\bvndt\b', r'\bvlsp\b'
    ]
    text = (doc.get("title", "") + " " + doc.get("abstract", "")).lower()
    for pattern in keywords:
        if re.search(pattern, text):
            return True
    return False

def analyze_and_save(file_path):
    print(f"📂 Đang phân tích: {file_path} ...")
    if not os.path.exists(file_path):
        print("❌ File không tồn tại.")
        return

    # --- KHỞI TẠO CẤU TRÚC LƯU TRỮ ---
    # Dùng list để lưu ID
    list_surveys = []
    list_vietnamese = []
    
    # Dùng dict để map: {normalized_title: [id1, id2, ...]}
    title_map = defaultdict(list)
    
    # Thống kê số lượng
    stats_count = Counter()
    venue_counter = Counter()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line)
                p_id = doc.get("paper_id")
                title = doc.get("title", "")
                
                stats_count["total"] += 1
                
                # 1. Gom nhóm theo Title để tìm trùng lặp
                norm_title = normalize_text(title)
                if norm_title:
                    title_map[norm_title].append({
                        "id": p_id,
                        "title": title, # Lưu title gốc để dễ đọc
                        "year": doc.get("year") or doc.get("publication_date")
                    })

                # 2. Check Survey
                if doc.get("is_survey"):
                    list_surveys.append(p_id)
                    stats_count["surveys"] += 1

                # 3. Check Vietnamese
                if is_vietnamese_related(doc):
                    list_vietnamese.append(p_id)
                    stats_count["vietnamese"] += 1
                
                # 4. Venue Stats
                venue = doc.get("venue") or "Unknown"
                venue_counter[venue] += 1

            except: continue

    # --- XỬ LÝ TRÙNG LẶP ---
    # Chỉ giữ lại những title có > 1 bài báo
    duplicate_groups = {}
    for norm_title, papers in title_map.items():
        if len(papers) > 1:
            duplicate_groups[norm_title] = papers
            stats_count["duplicate_titles"] += 1

    # --- CẤU TRÚC JSON ĐẦU RA ---
    final_report = {
        "summary": {
            "total_papers": stats_count["total"],
            "total_surveys": stats_count["surveys"],
            "total_vietnamese": stats_count["vietnamese"],
            "total_duplicate_titles": stats_count["duplicate_titles"]
        },
        "top_venues": dict(venue_counter.most_common(20)),
        "details": {
            "survey_ids": list_surveys,         # List ID các bài survey
            "vietnamese_ids": list_vietnamese,  # List ID bài tiếng Việt
            "duplicates": duplicate_groups      # Danh sách các bài bị trùng
        }
    }

    # --- LƯU FILE ---
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)

    print(f"✅ Đã lưu báo cáo chi tiết vào: {REPORT_FILE}")
    print(f"   - Tìm thấy {stats_count['surveys']} bài survey")
    print(f"   - Tìm thấy {stats_count['vietnamese']} bài tiếng Việt")
    print(f"   - Tìm thấy {len(duplicate_groups)} nhóm tiêu đề bị trùng")

if __name__ == "__main__":
    analyze_and_save(INPUT_FILE)