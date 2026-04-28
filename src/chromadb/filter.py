# 0_filter.py
# Clean and preprocess raw dataset: remove invalid/short abstracts, normalize text, ensure data quality before indexing

import json

DATA_PROCESSED_DIR = "data/data_processed"
CHROMA_DIR = "src/chromadb"

FILE_PATH = f"{DATA_PROCESSED_DIR}/final_cleaned_data.jsonl"
OUTPUT_PATH = f"{CHROMA_DIR}/0_data.jsonl"

MIN_WORDS = 30

def is_valid_text(text):
    text = text.strip()
    if not text:
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    if len(set(text)) < 5:  # tránh ".", "aaaa", ...
        return False
    return True


def normalize(text):
    return " ".join(text.split())


def filter_data_with_abstract(input_path, output_path):
    total = 0
    kept = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            total += 1

            record = json.loads(line)
            title = normalize(record.get("title", ""))
            abstract = normalize(record.get("abstract", ""))

            if not is_valid_text(abstract):
                continue

            # update lại record sạch
            record["title"] = title
            record["abstract"] = abstract

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Total: {total}")
    print(f"Kept: {kept}")
    print(f"Dropped: {total - kept}")


if __name__ == "__main__":
    filter_data_with_abstract(FILE_PATH, OUTPUT_PATH)
    print(f"Filtered data saved to {OUTPUT_PATH}")