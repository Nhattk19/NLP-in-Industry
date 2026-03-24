import json

DATA_PROCESSED_DIR = "data/data_processed"
CHROMA_DIR = "src/chromadb"

FILE_PATH = f"{DATA_PROCESSED_DIR}/final_cleaned_data.jsonl"
OUTPUT_PATH = f"{CHROMA_DIR}/data_with_abstract.jsonl"

def filter_data_with_abstract(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            record = json.loads(line)
            abstract = record.get("abstract", "").strip()
            if abstract:  # chỉ giữ lại những record có abstract không rỗng
                outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
if __name__ == "__main__":
    filter_data_with_abstract(FILE_PATH, OUTPUT_PATH)
    print(f"Filtered data with abstracts saved to {OUTPUT_PATH}")