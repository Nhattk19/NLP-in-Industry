import os
import json
import re
import glob
from collections import Counter, defaultdict

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Thư mục data/
INPUT_DIR = os.path.join(BASE_DIR, "data_processed")                               # Đọc TẤT CẢ file trong data/final/
REPORT_FILE = os.path.join(INPUT_DIR, "6_statistics_report.json")         # File lưu ID chi tiết

def normalize_text(text):
    if not text: return ""
    text = str(text).lower().strip()
    return re.sub(r'[^a-z0-9]', '', text)

def extract_year(date_str):
    if not date_str: return "Unknown"
    try:
        if "-" in str(date_str): return str(date_str).split("-")[0]
        return str(date_str)
    except: return "Unknown"

def is_vietnamese_related(doc):
    keywords =[
        r'\bvietnamese\b', r'\bvietnam\b', r'\btieng viet\b', r'\btiếng việt\b',
        r'\bphobert\b', r'\bvi-bert\b', r'\bvndt\b', r'\bvlsp\b'
    ]
    text = (doc.get("title", "") + " " + doc.get("abstract", "")).lower()
    for pattern in keywords:
        if re.search(pattern, text): return True
    return False

def analyze_directory(input_dir):
    print(f"🔍 Đang quét thư mục: {input_dir}")
    
    # Lấy tất cả các file .jsonl trong thư mục (bỏ qua file report nếu có đuôi jsonl)
    file_pattern = os.path.join(input_dir, "*.jsonl")
    files_to_process = glob.glob(file_pattern)
    
    if not files_to_process:
        print(f"❌ Không tìm thấy file .jsonl nào trong thư mục {input_dir}")
        return

    print(f"📁 Tìm thấy {len(files_to_process)} file dữ liệu. Bắt đầu tổng hợp...\n")

    # --- KHỞI TẠO BIẾN THỐNG KÊ TỔNG ---
    stats = {
        "total_papers": 0, "surveys": 0, "papers": 0,
        "duplicates_id": 0, "duplicates_title": 0,
        "vietnamese_related": 0,
        "missing": {
            "abstract": 0, "authors": 0, "venue": 0,
            "publication_date": 0, "citations_list": 0, "pdf_link": 0
        },
        "years": Counter(),
        "venues": Counter(),
        "nlp_scores":[],
        "authors_count": Counter(),
        "abstract_word_counts": [],
        "citations_lengths":[],
        "references_lengths":[],
        "sources": {"arxiv": 0, "acl": 0, "doi": 0}
    }

    # Các biến dùng để lưu ra file JSON Report
    seen_ids = set()
    title_map = defaultdict(list)
    list_vietnamese =[]

    # --- ĐỌC TỪNG FILE ---
    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        print(f"   ⏳ Đang đọc file: {file_name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line: continue
                    doc = json.loads(line)
                    
                    p_id = str(doc.get("paper_id"))
                    title = doc.get("title", "")
                    
                    # Nếu ID đã tồn tại -> Bỏ qua toàn bộ logic bên dưới để không đếm đúp (Tùy chọn)
                    # Ở đây ta vẫn đếm để biết trùng bao nhiêu, nhưng đưa vào list trùng
                    if p_id in seen_ids:
                        stats["duplicates_id"] += 1
                    else:
                        seen_ids.add(p_id)

                    stats["total_papers"] += 1

                    # Xử lý trùng Tiêu đề
                    norm_title = normalize_text(title)
                    if norm_title:
                        title_map[norm_title].append({
                            "id": p_id,
                            "title": title,
                            "file_source": file_name # Lưu lại file gốc chứa bài này
                        })

                    # Phân loại
                    if doc.get("is_survey"):
                        stats["surveys"] += 1
                    else:
                        stats["papers"] += 1

                    # Vietnamese
                    if is_vietnamese_related(doc):
                        stats["vietnamese_related"] += 1
                        list_vietnamese.append(p_id)

                    # Metadata
                    year = extract_year(doc.get("publication_date") or doc.get("year"))
                    stats["years"][year] += 1
                    
                    venue = doc.get("venue")
                    if venue: stats["venues"][venue] += 1
                    else: stats["venues"]["Unknown"] += 1

                    if doc.get("nlp_score"): stats["nlp_scores"].append(doc.get("nlp_score"))

                    # Missing Data & Deep Stats
                    abs_text = doc.get("abstract", "")
                    if not abs_text: stats["missing"]["abstract"] += 1
                    else: stats["abstract_word_counts"].append(len(abs_text.split()))

                    authors = doc.get("authors", [])
                    if not authors: stats["missing"]["authors"] += 1
                    else:
                        for author in authors:
                            if isinstance(author, dict) and author.get("name"):
                                stats["authors_count"][author["name"]] += 1

                    if not doc.get("venue"): stats["missing"]["venue"] += 1
                    if not doc.get("publication_date"): stats["missing"]["publication_date"] += 1

                    network = doc.get("network", {})
                    cits = network.get("citations",[])
                    refs = network.get("references", [])
                    
                    if not cits: stats["missing"]["citations_list"] += 1
                    stats["citations_lengths"].append(len(cits))
                    stats["references_lengths"].append(len(refs))

                    ext = doc.get("externalsid", {})
                    has_link = False
                    if ext.get("arxiv"): stats["sources"]["arxiv"] += 1; has_link = True
                    if ext.get("acl"): stats["sources"]["acl"] += 1; has_link = True
                    if ext.get("doi"): stats["sources"]["doi"] += 1; has_link = True
                    if not has_link and not ext.get("s2_url"):
                        stats["missing"]["pdf_link"] += 1

                except json.JSONDecodeError:
                    continue

    # --- XỬ LÝ TRÙNG LẶP TIÊU ĐỀ ĐỂ LƯU REPORT ---
    duplicate_groups = {}
    for norm_title, papers_list in title_map.items():
        if len(papers_list) > 1:
            duplicate_groups[norm_title] = papers_list
            stats["duplicates_title"] += (len(papers_list) - 1)

    # --- IN KẾT QUẢ TỔNG HỢP ---
    print("\n" + "="*60)
    print(f"📊 TỔNG QUAN DỮ LIỆU TỪ {len(files_to_process)} FILE")
    print("="*60)
    print(f"🔹 Tổng số bài báo:       {stats['total_papers']:,}")
    print(f"🔹 Bài nghiên cứu (Papers): {stats['papers']:,} ({stats['papers']/(stats['total_papers'] or 1)*100:.1f}%)")
    print(f"🔹 Bài khảo sát (Surveys): {stats['surveys']:,} ({stats['surveys']/(stats['total_papers'] or 1)*100:.1f}%)")
    print(f"🔹 Bài liên quan Việt Nam: {stats['vietnamese_related']:,}")
    
    print("\n⚠️  CHẤT LƯỢNG DỮ LIỆU")
    print("-" * 30)
    print(f"🔸 Trùng lặp ID tuyệt đối: {stats['duplicates_id']}")
    print(f"🔸 Trùng lặp Tiêu đề:      {stats['duplicates_title']} (Các bài có thể bị duplicate)")
    print(f"🔸 Thiếu Abstract:         {stats['missing']['abstract']} ({stats['missing']['abstract']/(stats['total_papers'] or 1)*100:.1f}%)")
    print(f"🔸 Thiếu Citation List:    {stats['missing']['citations_list']}")
    
    print("\n👥 THỐNG KÊ TÁC GIẢ (AUTHORSHIP)")
    print("-" * 30)
    print(f"   - Tổng số tác giả duy nhất: {len(stats['authors_count']):,}")
    print("   - Top 3 tác giả có nhiều bài nhất:")
    for author, count in stats['authors_count'].most_common(3):
        print(f"     * {author}: {count} bài")

    print("\n📝 THỐNG KÊ ABSTRACT & NETWORK")
    print("-" * 30)
    if stats["abstract_word_counts"]:
        words = stats["abstract_word_counts"]
        print(f"   - Abstract: Trung bình {sum(words)/len(words):.0f} từ/bài")
    if stats["citations_lengths"]:
        cits = stats["citations_lengths"]
        refs = stats["references_lengths"]
        print(f"   - Citations: Trung bình {sum(cits)/len(cits):.1f} links/bài (Max: {max(cits)})")
        print(f"   - References: Trung bình {sum(refs)/len(refs):.1f} links/bài (Max: {max(refs)})")

    print("\n🔗 NGUỒN LIÊN KẾT NGOÀI")
    print("-" * 30)
    print(f"   - Có link ArXiv: {stats['sources']['arxiv']:,} bài")
    print(f"   - Có link ACL:   {stats['sources']['acl']:,} bài")
    
    print("\n📅 PHÂN BỐ THEO NĂM (Top 5)")
    print("-" * 30)
    sorted_years = sorted(stats["years"].items(), key=lambda x: x[0], reverse=True)
    for year, count in sorted_years[:5]:
        if year != "Unknown": print(f"   - {year}: {count} bài")

    print("="*60)

    # --- LƯU FILE REPORT JSON ---
    final_report = {
        "summary": {
            "total_files_scanned": len(files_to_process),
            "total_papers": stats["total_papers"],
            "total_surveys": stats["surveys"],
            "total_vietnamese": stats["vietnamese_related"],
            "total_duplicate_titles": stats["duplicates_title"]
        },
        "details": {
            "vietnamese_ids": list_vietnamese,
            "duplicates_by_title": duplicate_groups
        }
    }

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)
    
    print(f"✅ Đã lưu danh sách ID chi tiết (Surveys, Tiếng Việt, Trùng lặp) vào: {REPORT_FILE}")

if __name__ == "__main__":
    analyze_directory(INPUT_DIR)