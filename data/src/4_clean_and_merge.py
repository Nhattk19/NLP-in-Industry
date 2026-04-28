import os
import json
import re
import glob

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # data/
INPUT_DIR = os.path.join(BASE_DIR, "data_raw") # Thư mục chứa các file JSONL đã qua bước 3_strict_filter
CLEAN_DIR = os.path.join(BASE_DIR, "data_processed")
OUTPUT_FILE = os.path.join(CLEAN_DIR, "final_cleaned_data.jsonl")

# ==========================================
# CÁC HÀM HỖ TRỢ VÀ LUẬT GIẢI QUYẾT XUNG ĐỘT
# ==========================================

def normalize_text(text):
    if not text: return ""
    return re.sub(r'[^a-z0-9]', '', str(text).lower().strip())

def is_empty(val):
    if val is None: return True
    if isinstance(val, (str, list, dict)) and len(val) == 0: return True
    return False

def get_author_names_set(record):
    """Lấy tập hợp tên tác giả để so sánh (Case 3)"""
    return {normalize_text(a.get("name", "")) for a in record.get("authors",[]) if a.get("name")}

def have_common_authors(rec1, rec2):
    """CASE 3 (Anti-Merge): Kiểm tra xem 2 bài báo có ít nhất 1 tác giả chung không"""
    names1 = get_author_names_set(rec1)
    names2 = get_author_names_set(rec2)
    # Nếu 1 trong 2 bài bị khuyết data tác giả hoàn toàn, tạm coi là có thể trùng để an toàn (tránh tách nhầm)
    if not names1 or not names2: return True 
    return len(names1.intersection(names2)) > 0

def get_venue_score(venue):
    """Đánh giá chất lượng Venue (Case 1 - Ưu tiên 1)"""
    v = str(venue).lower()
    if not v or 'arxiv' in v or 'unknown' in v: return 0
    return 1 # Là hội nghị chính thức

def has_good_pdf_link(record):
    """Kiểm tra xem có link tải PDF tốt không (Case 1 - Ưu tiên 3)"""
    ext = record.get("externalsid", {})
    return bool(ext.get("arxiv") or ext.get("acl"))

def determine_winner(rec1, rec2):
    """
    CASE 1: Quyết định xem bản ghi nào làm bản chính (Winner), bản nào là phụ (Loser).
    Trả về: (Winner, Loser)
    """
    # 1. Ưu tiên Venue (Hội nghị chính thức thắng ArXiv/Unknown)
    v1_score = get_venue_score(rec1.get("venue"))
    v2_score = get_venue_score(rec2.get("venue"))
    if v1_score > v2_score: return rec1, rec2
    if v2_score > v1_score: return rec2, rec1

    # 2. Ưu tiên Citation Count (Nhiều trích dẫn hơn thắng)
    c1 = int(rec1.get("citation_count") or 0)
    c2 = int(rec2.get("citation_count") or 0)
    if c1 > c2: return rec1, rec2
    if c2 > c1: return rec2, rec1

    # 3. Ưu tiên Link PDF (Có ACL/ArXiv thắng chỉ có link S2)
    l1 = has_good_pdf_link(rec1)
    l2 = has_good_pdf_link(rec2)
    if l1 and not l2: return rec1, rec2
    if l2 and not l1: return rec2, rec1

    # 4. Ưu tiên Ngày xuất bản (Mới nhất thắng)
    d1 = str(rec1.get("publication_date") or rec1.get("year") or "0")
    d2 = str(rec2.get("publication_date") or rec2.get("year") or "0")
    if d1 > d2: return rec1, rec2
    if d2 > d1: return rec2, rec1

    # Mặc định lấy bài 1 làm Winner nếu giống hệt nhau
    return rec1, rec2

def is_numeric_id(aid):
    """Kiểm tra ID tác giả có phải là số chuẩn không"""
    return aid and str(aid).isdigit()

def merge_authors(winner_authors, loser_authors):
    """
    CASE 2: Xử lý gộp tác giả. 
    Lấy danh sách tác giả của Winner làm gốc, đắp ID xịn từ Loser sang nếu Winner bị thiếu.
    """
    loser_map = {normalize_text(a.get("name", "")): a for a in loser_authors if a.get("name")}
    
    merged_authors =[]
    for wa in winner_authors:
        w_name = normalize_text(wa.get("name", ""))
        w_id = wa.get("id")
        
        # Nếu Loser cũng có tác giả này
        if w_name in loser_map:
            la = loser_map[w_name]
            l_id = la.get("id")
            
            # Nếu Winner không có ID, hoặc Winner có mã Hash dài nhưng Loser có số ngắn chuẩn -> Lấy của Loser
            if not w_id or (not is_numeric_id(w_id) and is_numeric_id(l_id)):
                w_id = l_id
                
        merged_authors.append({
            "name": wa.get("name"), 
            "id": w_id
        })
        
    return merged_authors

# ==========================================
# LOGIC GỘP (MERGE) TỰ ĐỘNG 100%
# ==========================================

def deep_merge(winner, loser):
    """
    Gộp 2 record: 100% TỰ ĐỘNG.
    - Lấy Winner làm nền.
    - Loser đắp vào chỗ thiếu.
    - Xung đột (cả 2 đều có nhưng khác nhau) -> Ép lấy của Winner.
    """
    merged = {}
    all_keys = set(winner.keys()).union(set(loser.keys()))
    
    for k in all_keys:
        v_win = winner.get(k)
        v_los = loser.get(k)
        
        # Bỏ qua tác giả (vì đã có hàm merge_authors xử lý riêng cực chuẩn)
        if k == "authors": 
            continue
            
        # Ưu tiên lấy nlp_score của Winner
        if k == "nlp_score":
            merged[k] = v_win if v_win is not None else v_los
            continue
            
        # LOGIC GỘP TỰ ĐỘNG
        if v_win == v_los:
            merged[k] = v_win
        elif is_empty(v_win) and not is_empty(v_los):
            merged[k] = v_los # Winner bị khuyết -> Lấy của Loser đắp vào
        elif is_empty(v_los) and not is_empty(v_win):
            merged[k] = v_win # Loser khuyết -> Giữ nguyên của Winner
        elif isinstance(v_win, dict) and isinstance(v_los, dict):
            # Cả 2 đều là Dictionary (vd: network, externalsid) -> Đệ quy gộp bên trong
            merged[k] = deep_merge(v_win, v_los)
        else:
            # XUNG ĐỘT (Cả 2 đều có nhưng khác nhau) -> ÉP BUỘC LẤY CỦA WINNER (Bản chính thức)
            merged[k] = v_win
            
    return merged

# ==========================================
# HÀM CHẠY CHÍNH
# ==========================================

def clean_and_merge():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    files_to_process = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    
    if not files_to_process:
        print(f"❌ Không tìm thấy file nào trong {INPUT_DIR}")
        return

    print(f"🚀 Bắt đầu CLEAN & MERGE TỰ ĐỘNG 100% từ {len(files_to_process)} file...")
    
    papers_db = {}           # {paper_id: record_dict}
    title_to_id = {}         # {normalized_title: paper_id}
    
    stats = {"read": 0, "merged": 0, "auto_split": 0}

    for file_path in files_to_process:
        # BỎ QUA FILE OUTPUT NẾU NÓ NẰM CÙNG THƯ MỤC
        if os.path.abspath(file_path) == os.path.abspath(OUTPUT_FILE): 
            continue
            
        print(f"   ⏳ Đang xử lý file: {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line: continue
                    record = json.loads(line)
                    stats["read"] += 1
                    
                    p_id = str(record.get("paper_id"))
                    norm_title = normalize_text(record.get("title", ""))
                    
                    is_duplicate = False
                    existing_id = None
                    
                    # 1. Phát hiện trùng lặp
                    if p_id in papers_db:
                        is_duplicate = True
                        existing_id = p_id
                    elif norm_title in title_to_id:
                        existing_id = title_to_id[norm_title]
                        
                        # CASE 3: ANTI-MERGE RULE (Kiểm tra tác giả chung)
                        if have_common_authors(papers_db[existing_id], record):
                            is_duplicate = True
                        else:
                            # Khác tác giả hoàn toàn -> Tự động tách thành 2 bài độc lập
                            is_duplicate = False
                            stats["auto_split"] += 1
                            
                    # 2. Xử lý gộp hoặc thêm mới
                    if is_duplicate:
                        existing_record = papers_db[existing_id]
                        
                        # CASE 1: XÁC ĐỊNH BẢN CHÍNH (WINNER) VÀ BẢN PHỤ (LOSER)
                        winner, loser = determine_winner(existing_record, record)
                        
                        # GỘP TỰ ĐỘNG 100%
                        merged_record = deep_merge(winner, loser)
                        
                        # CASE 2: GỘP TÁC GIẢ THÔNG MINH
                        merged_record["authors"] = merge_authors(winner.get("authors",[]), loser.get("authors",[]))
                        
                        # Cập nhật DB theo ID của Winner (đề phòng bản Winner là bản đang đọc vào)
                        winner_id = str(winner.get("paper_id"))
                        if winner_id != existing_id and existing_id in papers_db:
                            del papers_db[existing_id] # Xóa khóa ID cũ của Loser
                            
                        papers_db[winner_id] = merged_record
                        title_to_id[norm_title] = winner_id
                        stats["merged"] += 1
                        
                    else:
                        # Bài mới hoàn toàn (Hoặc vừa bị Auto-Split)
                        new_id = p_id
                        # Chống đè dữ liệu nếu ID hệ thống giống hệt nhau nhưng bị tách ở Case 3
                        if new_id in papers_db:
                            new_id = f"{new_id}_split_{stats['read']}"
                            record["paper_id"] = new_id

                        papers_db[new_id] = record
                        if norm_title: 
                            title_to_id[norm_title] = new_id
                            
                except json.JSONDecodeError: 
                    continue

    # Ghi kết quả ra file
    print("\n" + "="*50)
    print("💾 Đang ghi dữ liệu đã làm sạch ra file...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for p_id, record in papers_db.items():
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("="*50)
    print("🎉 HOÀN TẤT CLEAN & MERGE TỰ ĐỘNG!")
    print(f"   - Tổng số dòng đã đọc:          {stats['read']:,}")
    print(f"   - Số lần gộp bản chính/phụ:     {stats['merged']:,}")
    print(f"   - Số lần tự động tách (Case 3): {stats['auto_split']:,}")
    print(f"   - TỔNG SỐ BÀI DUY NHẤT CUỐI CÙNG: {len(papers_db):,}")
    print(f"📂 File kết quả sạch lưu tại:      {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_and_merge()