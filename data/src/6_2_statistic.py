import os
import json
import re
import glob
from collections import Counter, defaultdict


# ==============================
# PATH CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data_processed")
REPORT_FILE = os.path.join(INPUT_DIR, "6_2_statistics_report.json")


# ==============================
# UTILITY FUNCTIONS
# ==============================

def normalize_text(text):
    if not text:
        return ""
    text = str(text).lower().strip()
    return re.sub(r'[^a-z0-9]', '', text)


def extract_year(date_str):
    if not date_str:
        return "Unknown"

    try:
        date_str = str(date_str)
        if "-" in date_str:
            return date_str.split("-")[0]
        return date_str
    except:
        return "Unknown"


def is_vietnamese_related(doc):
    keywords = [
        r'\bvietnamese\b',
        r'\bvietnam\b',
        r'\btieng viet\b',
        r'\btiếng việt\b',
        r'\bphobert\b',
        r'\bvi-bert\b',
        r'\bvndt\b',
        r'\bvlsp\b'
    ]

    text = (doc.get("title", "") + " " + doc.get("abstract", "")).lower()

    for pattern in keywords:
        if re.search(pattern, text):
            return True

    return False


# ==============================
# MAIN ANALYSIS
# ==============================

def analyze_directory(input_dir):

    print(f"🔍 Scanning directory: {input_dir}")

    file_pattern = os.path.join(input_dir, "*.jsonl")
    files_to_process = glob.glob(file_pattern)

    if not files_to_process:
        print("❌ No JSONL files found.")
        return

    print(f"📁 Found {len(files_to_process)} files\n")

    # ==============================
    # STATISTICS STORAGE
    # ==============================

    stats = {

        "total_papers": 0,
        "surveys": 0,
        "papers": 0,

        "duplicates_id": 0,
        "duplicates_title": 0,

        "vietnamese_related": 0,

        # citation / reference coverage
        "has_citations_list": 0,
        "has_references_list": 0,

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

        "sources": {
            "arxiv": 0,
            "acl": 0,
            "doi": 0
        }
    }

    # ==============================
    # AUX STRUCTURES
    # ==============================

    seen_ids = set()
    title_map = defaultdict(list)

    vietnamese_ids = []

    # ==============================
    # PROCESS FILES
    # ==============================

    for file_path in files_to_process:

        file_name = os.path.basename(file_path)
        print(f"   ⏳ Processing: {file_name}")

        with open(file_path, 'r', encoding='utf-8') as f:

            for line in f:

                try:

                    line = line.strip()
                    if not line:
                        continue

                    doc = json.loads(line)

                    stats["total_papers"] += 1

                    p_id = str(doc.get("paper_id"))
                    title = doc.get("title", "")

                    # ==============================
                    # DUPLICATE ID
                    # ==============================

                    if p_id in seen_ids:
                        stats["duplicates_id"] += 1
                    else:
                        seen_ids.add(p_id)

                    # ==============================
                    # DUPLICATE TITLE
                    # ==============================

                    norm_title = normalize_text(title)

                    if norm_title:
                        title_map[norm_title].append({
                            "id": p_id,
                            "title": title,
                            "file_source": file_name
                        })

                    # ==============================
                    # SURVEY / PAPER
                    # ==============================

                    if doc.get("is_survey"):
                        stats["surveys"] += 1
                    else:
                        stats["papers"] += 1

                    # ==============================
                    # VIETNAMESE RELATED
                    # ==============================

                    if is_vietnamese_related(doc):
                        stats["vietnamese_related"] += 1
                        vietnamese_ids.append(p_id)

                    # ==============================
                    # YEAR
                    # ==============================

                    year = extract_year(
                        doc.get("publication_date") or doc.get("year")
                    )

                    stats["years"][year] += 1

                    # ==============================
                    # VENUE
                    # ==============================

                    venue = doc.get("venue")

                    if venue:
                        stats["venues"][venue] += 1
                    else:
                        stats["venues"]["Unknown"] += 1
                        stats["missing"]["venue"] += 1

                    # ==============================
                    # ABSTRACT
                    # ==============================

                    abstract = doc.get("abstract")

                    if not abstract:
                        stats["missing"]["abstract"] += 1
                    else:
                        stats["abstract_word_counts"].append(
                            len(abstract.split())
                        )

                    # ==============================
                    # AUTHORS
                    # ==============================

                    authors = doc.get("authors", [])

                    if not authors:
                        stats["missing"]["authors"] += 1
                    else:
                        for author in authors:
                            if isinstance(author, dict) and author.get("name"):
                                stats["authors_count"][author["name"]] += 1

                    # ==============================
                    # PUBLICATION DATE
                    # ==============================

                    if not doc.get("publication_date"):
                        stats["missing"]["publication_date"] += 1

                    # ==============================
                    # NETWORK DATA
                    # ==============================

                    network = doc.get("network", {})

                    citations = network.get("citations", [])
                    references = network.get("references", [])

                    # citations
                    if citations:
                        stats["has_citations_list"] += 1
                    else:
                        stats["missing"]["citations_list"] += 1

                    # references
                    if references:
                        stats["has_references_list"] += 1
                    else:
                        stats["missing"]["references_list"] += 1

                    stats["citations_lengths"].append(len(citations))
                    stats["references_lengths"].append(len(references))

                    # ==============================
                    # EXTERNAL LINKS
                    # ==============================

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

    # ==============================
    # DUPLICATE TITLES
    # ==============================

    duplicate_groups = {}

    for norm_title, papers_list in title_map.items():

        if len(papers_list) > 1:

            duplicate_groups[norm_title] = papers_list

            stats["duplicates_title"] += (len(papers_list) - 1)

    # ==============================
    # PRINT REPORT
    # ==============================

    total = stats["total_papers"] or 1

    print("\n" + "=" * 60)
    print("📊 DATASET OVERVIEW")
    print("=" * 60)

    print(f"Total papers: {stats['total_papers']:,}")
    print(f"Papers: {stats['papers']:,}")
    print(f"Surveys: {stats['surveys']:,}")
    print(f"Vietnam-related: {stats['vietnamese_related']:,}")

    print("\n⚠️ DATA QUALITY")
    print("-" * 30)

    print(f"Duplicate IDs: {stats['duplicates_id']}")
    print(f"Duplicate titles: {stats['duplicates_title']}")

    print(f"Missing abstract: {stats['missing']['abstract']}")
    print(f"Missing citations: {stats['missing']['citations_list']}")
    print(f"Missing references: {stats['missing']['references_list']}")

    print("\n🧠 NETWORK COVERAGE")
    print("-" * 30)

    abstract_pct = (total - stats["missing"]["abstract"]) / total * 100
    citations_pct = stats["has_citations_list"] / total * 100
    references_pct = stats["has_references_list"] / total * 100

    print(f"Papers with citations list: {stats['has_citations_list']:,} ({citations_pct:.1f}%)")
    print(f"Papers with references list: {stats['has_references_list']:,} ({references_pct:.1f}%)")

    if stats["citations_lengths"]:

        avg_citations = sum(stats["citations_lengths"]) / len(stats["citations_lengths"])
        avg_refs = sum(stats["references_lengths"]) / len(stats["references_lengths"])

        print(f"Avg citations: {avg_citations:.1f}")
        print(f"Avg references: {avg_refs:.1f}")

    print("\n👥 AUTHORS")
    print("-" * 30)

    print(f"Unique authors: {len(stats['authors_count']):,}")

    for author, count in stats["authors_count"].most_common(3):
        print(f"{author}: {count}")

    print("\n📅 YEAR DISTRIBUTION")
    print("-" * 30)

    sorted_years = sorted(stats["years"].items(), key=lambda x: x[0], reverse=True)

    for year, count in sorted_years[:5]:
        if year != "Unknown":
            print(year, count)

    print("=" * 60)

    # ==============================
    # SAVE JSON REPORT
    # ==============================

    final_report = {
        "summary": {
            "total_files": len(files_to_process),
            "total_papers": stats["total_papers"],
            "total_surveys": stats["surveys"],
            "vietnamese_related": stats["vietnamese_related"],
            "duplicate_titles": stats["duplicates_title"],
            "abstract_coverage_percent": abstract_pct,
            "citations_coverage_percent": citations_pct,
            "references_coverage_percent": references_pct
        },
        "details": {
            "vietnamese_ids": vietnamese_ids,
            "duplicate_titles": duplicate_groups
        }
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=4)

    print(f"✅ Report saved to {REPORT_FILE}")


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    analyze_directory(INPUT_DIR)