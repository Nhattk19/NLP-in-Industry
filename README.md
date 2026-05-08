# NLP-KG Search

Hệ thống tra cứu và hỏi đáp paper NLP/ML dựa trên kết hợp:

- `BM25` cho tìm kiếm từ khóa
- `ChromaDB` cho tìm kiếm ngữ nghĩa
- `RRF` để trộn kết quả lexical + semantic
- `LangGraph` để điều phối luồng RAG
- `Gemini` để phân loại intent, tạo câu trả lời và tự đánh giá chất lượng
- `arXiv API` để tìm paper bên ngoài khi kết quả nội bộ chưa đủ tốt
- `Streamlit` để cung cấp giao diện duyệt paper và chat RAG

## Tính năng chính

- Tìm kiếm paper theo 2 chế độ:
  - `Lexical` bằng BM25
  - `Semantic` bằng embedding + ChromaDB
- `Hybrid search` trộn hai nguồn bằng Reciprocal Rank Fusion
- Hiểu intent của người dùng:
  - `ood` - câu hỏi ngoài miền NLP/ML
  - `global` - câu hỏi chung về NLP/ML
  - `specific` - câu hỏi về một paper cụ thể
- Sinh câu trả lời có trích dẫn paper
- Tự đánh giá chất lượng câu trả lời
- Nếu câu trả lời còn yếu, agent có thể:
  - tìm paper mới từ arXiv
  - nạp vào ChromaDB
  - truy vấn lại và tạo câu trả lời lần nữa
- Giao diện web gồm:
  - trang home
  - trang kết quả tìm kiếm
  - trang chi tiết paper
  - trang chat RAG

## Kiến trúc tổng quan

```text
Query
  -> Intent Classifier
  -> Search Mode Selector
  -> Search Executor
  -> Context Extractor
  -> Answer Generator
  -> Result Evaluator
  -> External Search (nếu cần)
  -> Re-search / Re-evaluate
  -> Response Formatter
```

## Cấu trúc thư mục

```text
.
├── run_agent.py                 # CLI entry point cho agent
├── src/
│   ├── agent/                   # LangGraph agent và các node
│   ├── bm25/                    # BM25 searcher
│   ├── chromadb/                # Semantic search / rerank / ingest
│   ├── web/                     # Streamlit web app
│   ├── config.py                # Cấu hình chung cho search engine
│   └── search_engine.py         # Orchestrator cho lexical/semantic/hybrid search
├── data/
│   ├── data_raw/                # Dữ liệu thô và dữ liệu trung gian
│   ├── data_processed/          # Dữ liệu đã làm sạch
│   └── src/                     # Script tiền xử lý và đánh giá
├── configs/
├── md/
└── requirements.txt
```

## Dữ liệu và tài nguyên có sẵn

Repo hiện đã có sẵn một số asset để chạy demo ngay:

- `data/data_processed/final_cleaned_data.jsonl`: dữ liệu paper đã làm sạch
- `src/chromadb/chroma_store_abstracts/`: vector store cho search theo abstract
- `src/chroma_fulltext/chroma_store_fulltext/`: vector store cho chat RAG full-text
- `src/src/models_cache/`: cache model reranker

Nếu muốn build lại dữ liệu từ đầu, xem phần `Pipeline dữ liệu` bên dưới.

## Cài đặt

### 1. Tạo môi trường Python

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Cài dependency

```bash
pip install -r requirements.txt
```

Một số module UI và chat trong code còn dùng thêm các package như `streamlit`, `pandas`, `altair`, `tqdm`, `openai`, `torch`, `transformers`. Nếu môi trường của bạn chưa có sẵn, hãy cài bổ sung khi gặp lỗi import.

## Cấu hình

### 1. Gemini cho CLI agent

File `.env.example` đang trỏ tới:

```env
GOOGLE_API_KEY=./Google_api_key.txt
```

Bạn có thể:

- đặt API key vào file `Google_api_key.txt`
- hoặc tự set biến môi trường `GOOGLE_API_KEY`

Các biến chính cho agent:

| Biến | Mục đích | Mặc định |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Key cho Gemini | `./Google_api_key.txt` |
| `GEMINI_MODEL` | Model dùng cho classifier/generator/evaluator | `gemini-2.5-flash` |
| `MAX_CONTEXT_TOKENS` | Giới hạn context cho RAG | `2000` trong `.env.example` |
| `LLM_TEMPERATURE` | Nhiệt độ sinh câu trả lời | `0.2` |
| `LLM_MAX_TOKENS` | Giới hạn token output | `1000` |
| `EXTERNAL_SEARCH_NUM_RESULTS` | Số paper lấy từ arXiv | `5` |
| `PDF_PARSER_ENABLED` | Bật/tắt parse PDF | `true` |

### 2. OpenAI cho chat RAG trên web

Chat page của Streamlit đọc key từ:

```text
src/web/pages/api_agent.txt
```

Hãy tạo file này và đặt OpenAI API key vào đó nếu bạn muốn dùng trang chat.

### 3. Semantic Scholar API cho pipeline dữ liệu

Các script trong `data/src/` đọc key từ:

```text
data/src/api.txt
```

File này được dùng khi bạn muốn tải và xử lý dữ liệu gốc từ Semantic Scholar.

## Chạy ứng dụng

### 1. Chạy CLI agent

```bash
python run_agent.py --interactive
```

Các mode khác:

```bash
python run_agent.py --query "What is SOTA for Named Entity Recognition?"
python run_agent.py --examples
python run_agent.py --batch queries.json
python run_agent.py --output results.json
```

CLI agent sẽ:

- phân loại intent
- chọn search mode
- truy vấn BM25/ChromaDB
- build context
- tạo câu trả lời bằng Gemini
- chấm điểm câu trả lời
- gọi arXiv nếu cần

### 2. Chạy search engine pipeline

```bash
python src/search_engine.py
```

File này điều phối pipeline tìm kiếm theo `MODE` trong `src/search_engine.py`.

Các chế độ hiện có:

- `lexical`
- `semantic`
- `hybrid`

Mặc định repo đang dùng `hybrid`.

### 3. Chạy web app

```bash
streamlit run src/web/app.py
```

Web app gồm:

- Home page: tìm kiếm paper, bật/tắt semantic search, mở chat
- Results page: danh sách kết quả, filter theo năm và survey
- Detail page: xem thông tin paper, citations, references, related papers
- Chat page: RAG chat dùng full-text ChromaDB + rerank + OpenAI

## Pipeline dữ liệu

Các script trong `data/src/` tạo ra bộ dữ liệu sạch dùng cho indexing và search.

### Các bước chính

1. `data/src/1_loose_filter.py`
   - lấy candidate papers từ Semantic Scholar dataset
2. `data/src/2_map.py`
   - enrich paper bằng abstract, references, citations, external IDs
3. `data/src/3_strict_filter.py`
   - chấm điểm và lọc paper thật sự liên quan đến NLP
4. `data/src/4_clean_and_merge.py`
   - gộp và làm sạch dữ liệu, xuất `final_cleaned_data.jsonl`
5. `data/src/5_check_is_survey.py`
   - gán nhãn survey/review bằng model phân loại

### Tái tạo ChromaDB abstract

Nếu muốn build lại vector store cho search theo abstract:

```bash
python src/chromadb/ingest.py
```

### Tái tạo full-text store

Nếu muốn build lại kho full-text cho chat RAG:

```bash
python src/chroma_fulltext/ingest.py
```

## Luồng tìm kiếm

### BM25

`src/bm25/search_bm25.py` xây index từ `title + abstract` và trả về top-k paper theo điểm BM25.

### Semantic search

`src/chromadb/retrieve.py` query ChromaDB bằng embedding model `all-MiniLM-L6-v2`.

### Reranking

`src/chromadb/rerank.py` dùng FlashRank (`ms-marco-MiniLM-L-12-v2`) để sắp xếp lại kết quả semantic.

### Hybrid search

`src/search_engine.py` và `src/agent/nodes/search_executor.py` dùng Reciprocal Rank Fusion để trộn lexical và semantic.

## Agent RAG

Agent trong `src/agent/` chạy theo state machine của `LangGraph`:

- `intent_classifier.py`
- `search_executor.py`
- `context_extractor.py`
- `answer_generator.py`
- `result_evaluator.py`
- `external_searcher.py`
- `response_formatter.py`

Đặc điểm đáng chú ý:

- Query ngoài miền NLP được xử lý riêng, không đi qua search nội bộ
- Câu trả lời luôn cố gắng trích dẫn bằng `Paper: ...`
- Nếu điểm đánh giá thấp, agent có thể gọi arXiv và thử lại tối đa 2 vòng

## Lưu ý

- `data/data_raw/`, `data/data_processed/*.jsonl`, các file kết quả search và DB nhị phân đã được ignore trong `.gitignore`
- `Google_api_key.txt`, `data/src/api.txt`, `src/web/pages/api_agent.txt` nên được xem là secret local, không commit lên git
- Nếu web app báo thiếu package, hãy cài thêm các dependency UI/chat còn thiếu trong môi trường của bạn

## Gợi ý sử dụng nhanh

Nếu bạn chỉ muốn thử ngay:

```bash
pip install -r requirements.txt
python run_agent.py --query "What is BERT pretraining?"
streamlit run src/web/app.py
```

Nếu bạn muốn build lại dữ liệu từ đầu, hãy chạy pipeline `data/src/` trước rồi mới ingest ChromaDB.

