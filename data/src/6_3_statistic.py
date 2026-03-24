import os
import json
import re
import glob
from collections import Counter, defaultdict

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data_processed")
REPORT_FILE = os.path.join(INPUT_DIR, "6_3_statistics_report.json")


# ================= UTIL FUNCTIONS =================
def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower().strip()
    return re.sub(r'[^a-z0-9]', '', text)


def extract_year(date_str):
    if not date_str:
        return "Unknown"
    try:
        if "-" in str(date_str):
            return str(date_str).split("-")[0]
        return str(date_str)
    except:
        return "Unknown"


def is_vietnamese_related(doc):
    keywords = [
        r'\bvietnamese\b', r'\bvietnam\b',
        r'\btieng viet\b', r'\btiếng việt\b',
        r'\bphobert\b', r'\bvi-bert\b',
        r'\bvndt\b', r'\bvlsp\b'
    ]
    text = (doc.get("title", "") + " " + doc.get("abstract", "")).lower()
    return any(re.search(pattern, text) for pattern in keywords)


def convert_counter(counter_obj):
    return dict(counter_obj)


# ================= MAIN ANALYSIS =================
def analyze_directory(input_dir):
    file_pattern = os.path.join(input_dir, "*.jsonl")
    files = glob.glob(file_pattern)

    if not files:
        raise ValueError(f"No .jsonl files found in {input_dir}")

    # ===== INIT STATS =====
    stats = {
        "total_papers": 0,
        "surveys": 0,
        "papers": 0,
        "duplicates_id": 0,
        "duplicates_title": 0,
        "vietnamese_related": 0,
        "missing": {
            "abstract": 0,
            "authors": 0,
            "venue": 0,
            "publication_date": 0,
            "citations_list": 0,
            "references_list": 0,
            "pdf_link": 0
        },
        "years": Counter(),
        "venues": Counter(),
        "authors_count": Counter(),
        "abstract_word_counts": [],
        "citations_lengths": [],
        "references_lengths": [],
        "sources": {"arxiv": 0, "acl": 0, "doi": 0}
    }

    seen_ids = set()
    title_map = defaultdict(list)
    vietnamese_ids = []

    # ===== READ FILES =====
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.strip()
                    if not line:
                        continue

                    doc = json.loads(line)

                    p_id = str(doc.get("paper_id", ""))
                    title = doc.get("title", "")

                    # --- duplicate ID ---
                    if p_id in seen_ids:
                        stats["duplicates_id"] += 1
                    else:
                        seen_ids.add(p_id)

                    stats["total_papers"] += 1

                    # --- duplicate title ---
                    norm_title = normalize_text(title)
                    if norm_title:
                        title_map[norm_title].append({
                            "id": p_id,
                            "title": title
                        })

                    # --- survey / paper ---
                    if doc.get("is_survey"):
                        stats["surveys"] += 1
                    else:
                        stats["papers"] += 1

                    # --- vietnamese ---
                    if is_vietnamese_related(doc):
                        stats["vietnamese_related"] += 1
                        vietnamese_ids.append(p_id)

                    # --- metadata ---
                    year = extract_year(doc.get("publication_date") or doc.get("year"))
                    stats["years"][year] += 1

                    venue = doc.get("venue") or "Unknown"
                    stats["venues"][venue] += 1

                    # if doc.get("nlp_score") is not None:
                    #     stats["nlp_scores"].append(doc.get("nlp_score"))

                    # --- abstract ---
                    abstract = doc.get("abstract", "")
                    if not abstract:
                        stats["missing"]["abstract"] += 1
                    else:
                        stats["abstract_word_counts"].append(len(abstract.split()))

                    # --- authors ---
                    authors = doc.get("authors", [])
                    if not authors:
                        stats["missing"]["authors"] += 1
                    # else:
                    #     for a in authors:
                    #         if isinstance(a, dict) and a.get("name"):
                    #             stats["authors_count"][a["name"]] += 1

                    if not doc.get("venue"):
                        stats["missing"]["venue"] += 1

                    if not doc.get("publication_date"):
                        stats["missing"]["publication_date"] += 1

                    # --- network ---
                    network = doc.get("network", {})
                    cits = network.get("citations", [])
                    refs = network.get("references", [])

                    if not cits:
                        stats["missing"]["citations_list"] += 1
                    if not refs:
                        stats["missing"]["references_list"] +=1

                    stats["citations_lengths"].append(len(cits))
                    stats["references_lengths"].append(len(refs))

                    # --- sources ---
                    ext = doc.get("externalsid", {})
                    has_link = False

                    if ext.get("arxiv"):
                        stats["sources"]["arxiv"] += 1
                        has_link = True
                    if ext.get("acl"):
                        stats["sources"]["acl"] += 1
                        has_link = True
                    if ext.get("doi"):
                        stats["sources"]["doi"] += 1
                        has_link = True

                    if not has_link and not ext.get("s2_url"):
                        stats["missing"]["pdf_link"] += 1

                except json.JSONDecodeError:
                    continue

    # ===== DUPLICATE TITLE GROUP =====
    duplicate_groups = {}
    for title, papers in title_map.items():
        if len(papers) > 1:
            duplicate_groups[title] = papers
            stats["duplicates_title"] += len(papers) - 1

    # ===== BUILD REPORT =====
    report = {
        "summary": {
            "total_files": len(files),
            "total_papers": stats["total_papers"],
            "papers": stats["papers"],
            "surveys": stats["surveys"],
            "vietnamese_related": stats["vietnamese_related"],
            "duplicates_id": stats["duplicates_id"],
            "duplicates_title": stats["duplicates_title"]
        },
        "missing_data": stats["missing"],
        "distributions": {
            "years": convert_counter(stats["years"]),
            "venues": convert_counter(stats["venues"]),
            # "authors": convert_counter(stats["authors_count"])
        },
        "statistics": {
            "abstract_avg_len": (
                sum(stats["abstract_word_counts"]) / len(stats["abstract_word_counts"])
                if stats["abstract_word_counts"] else 0
            ),
            "citations_avg": (
                sum(stats["citations_lengths"]) / len(stats["citations_lengths"])
                if stats["citations_lengths"] else 0
            ),
            "references_avg": (
                sum(stats["references_lengths"]) / len(stats["references_lengths"])
                if stats["references_lengths"] else 0
            )
        },
        "sources": stats["sources"],
        #"nlp_scores": stats["nlp_scores"],
        "details": {
            "vietnamese_ids": vietnamese_ids,
            "duplicate_titles": duplicate_groups
        }
    }

    # ===== SAVE =====
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print(f"Saved report → {REPORT_FILE}")


# ================= ENTRY =================
if __name__ == "__main__":
    analyze_directory(INPUT_DIR)