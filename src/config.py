# config.py
# Store all configs here, including Hugging Face API key
import torch

QUERY_PROCESSING_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # can change to smaller model if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 20

OUTPUT_PATH_RETRIEVED = "./src/chromadb/retrieved_results.json"
OUTPUT_PATH_RERANKED = "./src/chromadb/reranked_results.json"
OUTPUT_PATH_CHROMADB = "./src/chromadb/results.json"
OUTPUT_PATH_BM25 = "./src/bm25/results.json"

DATA_PATH = "./data/data_processed/final_cleaned_data.jsonl"  # File chứa dữ liệu đã làm sạch (JSONL)
QUERY_PATH = "./src/queries.json"  # File chứa các câu query (JSON)
CHROMA_PATH = "./src/chromadb/chroma_store_abstracts"
COLLECTION_NAME = "papers_abstracts"
