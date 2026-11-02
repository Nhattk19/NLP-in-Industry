import os
import sys
from pathlib import Path

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.runtime_config import load_runtime_config

load_runtime_config()

import streamlit as st

from core.config import CHROMA_PATH
from core.paper_lookup import resolve_detail_paper
from core.resources import preload_search_resources
from core.theme import apply_page_config, inject_global_css
from pages.home import render_home_page
from pages.results import render_results_page

def _init_session_state() -> None:
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("search_query", "")
    st.session_state.setdefault("enable_semantic", True)


apply_page_config()
_init_session_state()
if not st.query_params.get("detail"):
    if st.query_params.get("view") == "chat" or st.query_params.get("chat"):
        st.session_state.page = "chat_rag"
    elif st.query_params.get("q"):
        st.session_state.page = "results"

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


inject_global_css()


@st.cache_resource(show_spinner="Initializing search and RAG resources. Please wait...")
def initialize_all_resources():
    """Initialize all shared resources once per server process, before rendering pages."""
    from core.rag import preload_agent, load_query_rewriter
    from core.detail_rag import load_detail_rag_components
    from src.chroma_fulltext.retrieve import load_embedder, load_reranker

    search_resources = preload_search_resources()
    agent = preload_agent()
    load_embedder()
    load_reranker()
    load_query_rewriter()
    load_detail_rag_components()
    if agent.search_executor.bm25_searcher is None or agent.search_executor.bm25_searcher.bm25 is None:
        raise RuntimeError("The full-text BM25 index could not be initialized.")
    if agent.search_executor.chroma_collection is None:
        raise RuntimeError("The full-text ChromaDB collection could not be initialized.")
    return search_resources


collection, reranker, bm25_engine, bm25_metadata = initialize_all_resources()

detail_key = st.query_params.get("detail", "").strip()
if detail_key:
    from src.chroma_fulltext.retrieve import collection as fulltext_collection

    selected_paper = resolve_detail_paper(
        detail_key, st.session_state.get("_cached_results") or [],
        bm25_metadata, fulltext_collection,
    )
    # A missing reference must show an explicit missing-paper state, never home
    # or a stale paper from an earlier selection.
    st.session_state.selected_paper = selected_paper
    st.session_state.page = "detail"

if not collection:
    st.error(f"ChromaDB was not found at `{CHROMA_PATH}`.")
if not bm25_engine:
    st.warning(f"BM25 could not be initialized from `{CHROMA_PATH}`.")

if st.session_state.page == "home":
    render_home_page()
elif st.session_state.page == "results":
    render_results_page(collection, reranker, bm25_engine, bm25_metadata)
elif st.session_state.page == "detail":
    from pages.detail import render_paper_detail_page

    render_paper_detail_page(collection, reranker, bm25_engine, bm25_metadata)
elif st.session_state.page == "chat_rag":
    from pages.chat_rag import render_chat_page

    render_chat_page()
