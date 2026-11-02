# NLP-KG Search

NLP-KG Search is a search and question-answering system for NLP/ML papers. The project combines lexical search, semantic search, and a RAG agent to help users retrieve papers, inspect paper details, and ask questions through a web interface.

## 1. Project Overview

The system includes the following main components:

- `BM25`: performs lexical keyword search over paper titles and abstracts.
- `ChromaDB`: stores vector embeddings for semantic search.
- `Sentence Transformers`: generates embeddings for paper abstracts and full texts.
- `FlashRank`: reranks semantic search results.
- `RRF`: combines lexical and semantic results for hybrid search.
- `LangGraph`: orchestrates the RAG agent workflow.
- `Gemini`: classifies intent, generates answers, and evaluates answer quality.
- `arXiv API`: retrieves external papers when the internal corpus is insufficient.
- `Streamlit`: provides the web interface for search, paper details, and RAG chat.

Important project structure:

```text
.
|-- run_agent.py                 # CLI entry point for the RAG agent
|-- requirements.txt             # Main dependencies
|-- .env.example                 # Environment configuration template
|-- data/
|   |-- data_processed/          # Processed paper data
|   `-- src/                     # Data processing scripts
|-- src/
|   |-- agent/                   # LangGraph agent and RAG nodes
|   |-- bm25/                    # BM25 search
|   |-- chromadb/                # Abstract vector store and semantic search
|   |-- chroma_fulltext/         # Full-text vector store for RAG chat
|   |-- web/                     # Streamlit web app
|   `-- search_engine_for_rag.py # Hybrid search / RRF
`-- md/                          # System flow documentation
```

## 2. Environment Setup Instructions

Environment requirements:

- Python 3.10 or later.
- Git.
- Internet access for the first-time download of embedding/reranker models and external API calls.
- A Google Gemini API key if you want to run the RAG agent or chat features.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Create the environment configuration file:

```bash
copy .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

Open `.env` and configure the important variables:

```env
GOOGLE_API_KEY=./Google_api_key.txt
GEMINI_MODEL=gemini-2.5-flash
MAX_CONTEXT_TOKENS=2000
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2000
STREAMLIT_PORT=8501
```

There are two ways to provide the Gemini key:

- Put the API key directly in the `GOOGLE_API_KEY` variable.
- Or create `Google_api_key.txt` in the project root and keep `GOOGLE_API_KEY=./Google_api_key.txt`.

If you run the data pipeline with Semantic Scholar, create this file:

```text
data/src/api.txt
```

Then place the Semantic Scholar API key inside that file.

## 3. Dependency Installation Steps

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## 4. How To Train The Model

This project does not train a supervised neural model from scratch. In this project, "training" means processing the paper data, generating embeddings, and rebuilding the search indexes.

### 4.1. Process Paper Data

Run the scripts in `data/src/` in order if you need to rebuild the dataset:

```bash
python data/src/1_loose_filter.py
python data/src/2_map.py
python data/src/3_strict_filter.py
python data/src/4_clean_and_merge.py
python data/src/5_check_is_survey.py
```

After the pipeline finishes, the cleaned data is stored in `data/data_processed/`.

### 4.2. Build The Abstract Vector Store

This vector store is used for semantic search over paper titles and abstracts:

```bash
python src/chromadb/ingest.py
```

Default output:

```text
src/chromadb/chroma_store_abstracts/
```

### 4.3. Build The Full-Text Vector Store

This vector store is used for RAG chat over full-text papers:

```bash
python src/chroma_fulltext/ingest.py
```

Default output:

```text
data/chroma_store_fulltext/
```

### 4.4. Run The Search Pipeline

```bash
python src/search_engine_for_rag.py
```

This pipeline combines BM25, ChromaDB, and RRF to produce hybrid search results.

## 5. How To Run Inference Or The Deployed System

### 5.1. Run CLI Inference

Run a single query:

```bash
python run_agent.py --query "What is BERT pretraining?"
```

Run interactive question answering:

```bash
python run_agent.py --interactive
```

Run built-in examples:

```bash
python run_agent.py --examples
```

Run batch queries from a JSON file:

```bash
python run_agent.py --batch queries.json --output results.json
```

### 5.2. Run The Streamlit Web App

```bash
streamlit run src/web/app.py
```

After the app starts, open:

```text
http://localhost:8501
```

The web app supports:

- Home page: enter a query and enable/disable semantic search.
- Results page: view, filter, and rank retrieved papers.
- Detail page: inspect detailed information for a selected paper.
- Chat page: ask questions using the RAG agent.

If the app reports that ChromaDB cannot be found, rebuild the indexes:

```bash
python src/chromadb/ingest.py
python src/chroma_fulltext/ingest.py
```

### 5.3. Run With Docker

Docker is the easiest way to give this project to someone else to run. The
image includes the Streamlit app and any local processed data / ChromaDB stores
that exist in this folder at build time.

Prepare the environment file:

```bash
cp .env.docker.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.docker.example .env
```

Then open `.env` and set:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

Build and run:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

To stop the app:

```bash
docker compose down
```

If you want to send a ready-built image to another machine:

```bash
docker compose build
docker save -o nlp-kg-search.tar nlp-kg-search:latest
```

On the other machine:

```bash
docker load -i nlp-kg-search.tar
docker run --env-file .env -p 8501:8501 nlp-kg-search:latest
```

Notes:

- Do not bake `.env`, `Google_api_key.txt`, or API tokens into the image.
- Build the image from a folder that already contains
  `data/data_processed/final_cleaned_data.jsonl`,
  `src/chromadb/chroma_store_abstracts/`, and
  `data/chroma_store_fulltext/` if you want the app to run without rebuilding
  indexes.
- The image can be large because the ChromaDB stores are a few GB.

## 6. Description Of Deployment Method

The deployment method is a Streamlit application running in a Python environment with the required data files and vector stores available locally.

Deployment steps:

1. Prepare a server or local machine with Python 3.10+.
2. Clone the project source code.
3. Create a virtual environment and install dependencies.
4. Configure `.env` and the required API keys.
5. Make sure the processed data and ChromaDB stores are available, or rebuild them with the ingest scripts.
6. Start the application with Streamlit.

Local deployment command:

```bash
streamlit run src/web/app.py 
```

Deployment architecture:

```text
User Browser
    |
    v
Streamlit Web App
    |
    |-- BM25 lexical search
    |-- ChromaDB semantic search
    |-- FlashRank reranker
    |-- LangGraph RAG agent
    |-- Gemini API
    `-- arXiv API fallback
```

The application does not require a separate model server. Embedding and reranker models are loaded directly inside the Python process, ChromaDB is stored locally on disk, and the LLM is accessed through the Gemini API.
