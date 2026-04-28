import streamlit as st

from core.config import CHROMA_PATH
from core.resources import init_bm25, init_chromadb, init_reranker
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

collection = init_chromadb()
reranker = init_reranker()
bm25_engine, bm25_metadata = init_bm25()

if not collection:
    st.error(f"ChromaDB was not found at `{CHROMA_PATH}`.")
if not bm25_engine:
    st.warning(f"BM25 could not be initialized from `{CHROMA_PATH}`. Lexical search may be unavailable.")

detail_key = st.query_params.get("detail", "").strip()
if detail_key:
    selected_paper = _resolve_detail_paper(detail_key, bm25_metadata)
    if selected_paper:
        st.session_state.selected_paper = selected_paper
        st.session_state.page = "detail"

if st.session_state.page == "home":
    render_home_page()
elif st.session_state.page == "results":
    render_results_page(collection, reranker, bm25_engine, bm25_metadata)
elif st.session_state.page == "detail":
    render_paper_detail_page(collection, reranker, bm25_engine, bm25_metadata)
elif st.session_state.page == "chat_rag":
    render_chat_page()
