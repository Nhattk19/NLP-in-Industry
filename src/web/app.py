import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from core.config import CHROMA_PATH
from core.resources import init_bm25, init_chromadb, init_reranker, preload_search_resources
from core.rag import preload_agent
from core.theme import apply_page_config, inject_global_css
from pages.detail import render_paper_detail_page
from pages.home import render_home_page
from pages.results import render_results_page
from pages.chat_rag import render_chat_page

def _init_session_state() -> None:
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("search_query", "")
    st.session_state.setdefault("enable_semantic", True)


def _matches_detail_key(item: dict, detail_key: str) -> bool:
    for field in ("doi", "arxiv", "paper_id", "title"):
        value = str(item.get(field, "")).strip()
        if value and value == detail_key:
            return True
    return False


def _resolve_detail_paper(detail_key: str, bm25_metadata: list[dict]) -> dict | None:
    cached_results = st.session_state.get("_cached_results") or []
    for item in cached_results:
        if _matches_detail_key(item, detail_key):
            return item

    for item in bm25_metadata:
        if _matches_detail_key(item, detail_key):
            return item

    return None


apply_page_config()
inject_global_css()
_init_session_state()

query_param = str(st.query_params.get("q", "") or "").strip()
if query_param:
    st.session_state.search_query = query_param
    if st.session_state.page == "home" and not st.query_params.get("detail", "").strip():
        st.session_state.page = "results"

semantic_param = str(st.query_params.get("semantic", "") or "").strip().lower()
if semantic_param in {"0", "false", "lexical"}:
    st.session_state.enable_semantic = False
elif semantic_param in {"1", "true", "semantic"}:
    st.session_state.enable_semantic = True

page_param = str(st.query_params.get("page", "") or "").strip()
if page_param.isdigit():
    st.session_state.result_page = int(page_param)

preload_search_resources()
preload_agent()

detail_key = st.query_params.get("detail", "").strip()
collection = None
reranker = None
bm25_engine = None
bm25_metadata = []

if detail_key:
    collection = init_chromadb()
    reranker = init_reranker()
    bm25_engine, bm25_metadata = init_bm25()
    selected_paper = _resolve_detail_paper(detail_key, bm25_metadata)
    if selected_paper:
        st.session_state.selected_paper = selected_paper
        st.session_state.page = "detail"

if st.session_state.page in {"results", "detail"} and collection is None:
    collection = init_chromadb()
    reranker = init_reranker()
    bm25_engine, bm25_metadata = init_bm25()

    if not collection:
        st.error(f"ChromaDB was not found at `{CHROMA_PATH}`.")
    if not bm25_engine:
        st.warning(f"BM25 could not be initialized from `{CHROMA_PATH}`. Lexical search may be unavailable.")

if st.session_state.page == "home":
    render_home_page()
elif st.session_state.page == "results":
    render_results_page(collection, reranker, bm25_engine, bm25_metadata)
elif st.session_state.page == "detail":
    render_paper_detail_page(collection, reranker, bm25_engine, bm25_metadata)
elif st.session_state.page == "chat_rag":
    render_chat_page()
