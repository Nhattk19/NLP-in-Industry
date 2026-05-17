import re
from html import escape

import streamlit as st

from core.search import hybrid_search
from core.detail_rag import paper_has_fulltext_chunks, run_detail_rag
from pages.results import RESULT_CARD_STYLE, build_results_dataframe, render_result_card


DETAIL_PAGE_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #ffffff;
    color: #2f2923;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.stApp,
section[data-testid="stSidebar"] {
    background: #ffffff !important;
}
[data-testid="stSidebarNavItems"] { display: none; }

div.stButton > button {
    background: #ffffff;
    color: #1a73e8;
    border: 1px solid #d6e3fb;
    border-radius: 16px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.65rem 1rem;
    box-shadow: 0 8px 22px rgba(26, 115, 232, 0.08);
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: #eef5ff;
    border-color: #bfd5fb;
    color: #1557b0;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%);
    color: #ffffff;
    border-color: #1a73e8;
    box-shadow: 0 12px 28px rgba(26, 115, 232, 0.18);
}

.detail-title { font-size: 1.6rem; font-weight: 700; color: #2f2923; margin: 10px 0 12px; line-height: 1.3; }
.detail-authors { font-size: 1rem; color: #5f5145; margin-bottom: 10px; font-weight: 500; }
.detail-meta { font-size: 0.9rem; color: #6b7280; margin-bottom: 16px; }
.detail-meta .venue { color: #1a73e8; font-weight: 600; }
.section-title { font-size: 1.1rem; font-weight: 600; color: #2f2923; margin: 24px 0 12px; border-bottom: 1px solid #dbe7fb; padding-bottom: 6px; }
.detail-abstract {
    font-size: 0.95rem;
    color: #463c34;
    line-height: 1.6;
    text-align: justify;
    background: #ffffff;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #dbe7fb;
    box-shadow: 0 10px 24px rgba(26, 115, 232, 0.04);
}
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; align-items: center; }
.chip {
    background: #f5f8fd;
    color: #4f5f7a;
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid #e2e9f5;
    white-space: nowrap;
}
.chip-survey { background:#eef4ff; color:#1e63e8; border-color:#d0defb; font-weight:700; }
.link-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background-color: #ffffff;
    color: #1a73e8 !important;
    border: 1px solid #cfe0ff;
    border-radius: 999px;
    padding: 7px 16px;
    font-size: 0.85rem;
    font-weight: 600;
    text-decoration: none !important;
    margin-right: 8px;
    margin-bottom: 8px;
}
.list-item { font-size: 0.9rem; margin-bottom: 8px; color: #4f4033; line-height: 1.5; }
.chat-placeholder {
    background: #ffffff;
    border: 1px solid #dbe7fb;
    border-radius: 20px;
    padding: 18px;
    min-height: 420px;
    position: sticky;
    top: 20px;
    box-shadow: 0 12px 28px rgba(26, 115, 232, 0.05);
}
.chat-placeholder-title { font-size: 1.05rem; font-weight: 700; color: #3f342b; margin-bottom: 10px; }
.chat-placeholder-text { font-size: 0.9rem; color: #746556; line-height: 1.6; }
.paper-rag-panel {
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.paper-rag-header {
    background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
    border: 1px solid #dbe7fb;
    border-radius: 16px;
    padding: 14px 14px 12px;
}
.paper-rag-kicker {
    display: inline-flex;
    align-items: center;
    background: #eef5ff;
    color: #1a73e8;
    border: 1px solid #d0defb;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 10px;
    margin-bottom: 10px;
}
.paper-rag-title {
    font-size: 1rem;
    font-weight: 800;
    color: #2f2923;
    margin-bottom: 6px;
}
.paper-rag-subtitle {
    font-size: 0.85rem;
    color: #6f5f52;
    line-height: 1.55;
}
.paper-rag-history {
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-height: 420px;
    overflow-y: auto;
    padding-right: 4px;
}
.paper-rag-empty {
    background: #fbfdff;
    border: 1px dashed #dbe7fb;
    border-radius: 14px;
    padding: 14px;
    color: #7a6d60;
    font-size: 0.88rem;
    line-height: 1.6;
}
.paper-rag-bubble {
    border-radius: 16px;
    padding: 12px 14px;
    line-height: 1.65;
    font-size: 0.9rem;
    border: 1px solid #e2e9f5;
    white-space: pre-wrap;
    word-break: break-word;
}
.paper-rag-bubble-user {
    background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%);
    color: #ffffff;
    margin-left: 24px;
}
.paper-rag-bubble-assistant {
    background: #ffffff;
    color: #2f2923;
    margin-right: 10px;
    box-shadow: 0 8px 20px rgba(26, 115, 232, 0.05);
}
.paper-rag-sources {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: flex-start;
    margin-top: 10px;
}
.paper-rag-source-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    background: #f6f8fc;
    border: 1px solid #dbe5f2;
    border-radius: 999px;
    box-shadow: 0 1px 0 rgba(26, 115, 232, 0.02);
    transition: transform 0.18s ease, border-color 0.18s ease, background-color 0.18s ease;
    max-width: 100%;
}
.paper-rag-source-pill:hover {
    background: #eef4ff;
    border-color: #c9d8f2;
    transform: translateY(-1px);
}
.paper-rag-source-title {
    color: #2f2923;
    font-size: 0.78rem;
    font-weight: 600;
    line-height: 1.25;
    max-width: 280px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.paper-rag-source-chip {
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    background: #edf3ff;
    color: #1a73e8;
    border: 1px solid #d3dff5;
    border-radius: 999px;
    font-size: 0.66rem;
    font-weight: 700;
    padding: 3px 8px;
    white-space: nowrap;
}
.chunk-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    vertical-align: baseline;
    padding: 0.08rem 0.55rem;
    margin: 0 2px;
    border-radius: 999px;
    background: #f4f7ff;
    border: 1px solid #d8e2fb;
    color: #5f6b86;
    font-size: 0.78em;
    font-weight: 600;
    line-height: 1.3;
    white-space: nowrap;
}
.chunk-tag-label {
    color: #7d8aa5;
    font-size: 0.9em;
    letter-spacing: 0.01em;
}
.chunk-tag-sep {
    color: #a4afc3;
}
.chunk-tag-id {
    color: #1a73e8;
    font-variant-numeric: tabular-nums;
    font-weight: 700;
}
.paper-rag-form {
    background: #fbfdff;
    border: 1px solid #dbe7fb;
    border-radius: 16px;
    padding: 12px;
}
.paper-tab-empty { font-size: 0.92rem; color: #7e7063; padding: 8px 0; }
.detail-tab-offset { margin-top: 34px; }
</style>
"""


def _normalize_text(value: object) -> str:
    return " ".join(str(value).split())


def _parse_list_string(text: str) -> list[str]:
    if not text or str(text).lower() == "nan":
        return []
    return [item.strip() for item in str(text).split("|") if item.strip()]


def _build_related_query(paper: dict) -> str:
    title = str(paper.get("title", "")).strip()
    abstract = str(paper.get("abstract", "")).strip()
    return f"{title}. {abstract}" if title and abstract else title or abstract


def _detail_chat_key(paper_id: str) -> str:
    return f"detail_chat_history::{paper_id}"


_CHUNK_ANNOTATION_RE = re.compile(r"\[Chunk:\s*([^\]]+)\]", re.IGNORECASE)


def _render_annotated_text(content: object) -> str:
    safe = escape(str(content or ""))

    def _replace(match: re.Match[str]) -> str:
        chunk_id = escape(match.group(1).strip())
        return (
            '<span class="chunk-tag">'
            '<span class="chunk-tag-label">Chunk</span>'
            '<span class="chunk-tag-sep">:</span>'
            f'<span class="chunk-tag-id">{chunk_id}</span>'
            '</span>'
        )

    return _CHUNK_ANNOTATION_RE.sub(_replace, safe).replace("\n", "<br>")


def _sources_html(sources: list[dict]) -> str:
    items = ""
    for source in sources or []:
        title = escape(_normalize_text(source.get("title") or "Untitled"))
        chunk_id = escape(_normalize_text(source.get("chunk_id") or ""))
        metric_value = source.get("confidence")
        metric_label = "conf"
        if metric_value in (None, "", "nan"):
            metric_value = source.get("score")
            metric_label = "score"
        if metric_value in (None, "", "nan"):
            metric_value = source.get("source_score")
            metric_label = "score"

        row = f'<span class="paper-rag-source-title" title="{title}">{title}</span>'
        if chunk_id:
            row += f'<span class="paper-rag-source-chip">{chunk_id}</span>'
        if metric_value not in (None, "", "nan"):
            try:
                metric_text = f"{float(metric_value):.3f}"
            except (TypeError, ValueError):
                metric_text = str(metric_value)
            row += f'<span class="paper-rag-source-chip">{metric_label} {escape(metric_text)}</span>'

        items += f'<div class="paper-rag-source-pill">{row}</div>'

    return f'<div class="paper-rag-sources">{items}</div>' if items else ""


def _init_detail_chat_session(paper_id: str) -> list[dict]:
    st.session_state.setdefault("detail_chat_history", {})
    st.session_state.detail_chat_history.setdefault(paper_id, [])
    return st.session_state.detail_chat_history[paper_id]


def _render_detail_chat_history(history: list[dict]) -> None:
    if not history:
        st.markdown(
            """
            <div class="paper-rag-empty">
                Ask a question about this paper and the assistant will retrieve
                relevant chunks from the current paper only, then answer with
                chunk-level grounding.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in history:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        sources = message.get("sources", [])

        bubble_class = "paper-rag-bubble-user" if role == "user" else "paper-rag-bubble-assistant"
        with st.container():
            st.markdown(
                f'<div class="paper-rag-bubble {bubble_class}">{_render_annotated_text(content)}</div>',
                unsafe_allow_html=True,
            )
            if role == "assistant" and sources:
                st.markdown(_sources_html(sources), unsafe_allow_html=True)


def _render_detail_chat_unavailable() -> None:
    st.markdown(
        """
        <div class="paper-rag-panel">
            <div class="paper-rag-header">
                <div class="paper-rag-kicker">Paper RAG</div>
                <div class="paper-rag-title">Ask about this paper</div>
                <div class="paper-rag-subtitle">
                    This paper does not yet support Q&A.
                </div>
            </div>
            <div class="paper-rag-empty">
                This paper does not yet support Q&A.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_detail_chat_panel(paper: dict) -> None:
    paper_id = str(paper.get("paper_id", "")).strip()
    history = _init_detail_chat_session(paper_id)

    st.markdown(
        """
        <div class="paper-rag-panel">
            <div class="paper-rag-header">
                <div class="paper-rag-kicker">Paper RAG</div>
                <div class="paper-rag-title">Ask about this paper</div>
                <div class="paper-rag-subtitle">
                    Retrieval is limited to the current paper's chunks, so answers stay
                    grounded in the selected paper only.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_detail_chat_history(history)


    with st.form(key=f"detail_rag_form::{paper_id}", clear_on_submit=True):
        question = st.text_area(
            "Ask this paper",
            placeholder="For example: What is the main method? How do they evaluate it? What are the limitations?",
            height=100,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask paper")

    if not submitted:
        return

    question = (question or "").strip()
    if not question:
        st.warning("Please type a question first.")
        return

    history.append({"role": "user", "content": question})
    with st.spinner("Retrieving from this paper..."):
        result = run_detail_rag(paper, question)

    answer = result.get("answer", "")
    history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "execution_path": result.get("execution_path", []),
        }
    )
    st.rerun()


def _render_detail_chat_section(paper: dict) -> None:
    paper_id = str(paper.get("paper_id", "")).strip()
    if paper_has_fulltext_chunks(paper_id):
        _render_detail_chat_panel(paper)
        return

    _render_detail_chat_unavailable()


def _back_to_results() -> None:
    st.session_state.page = "results"

    query = str(st.session_state.get("search_query", "") or "").strip()
    if query and not str(st.query_params.get("q", "") or "").strip():
        st.query_params["q"] = query
    if "semantic" not in st.query_params:
        st.query_params["semantic"] = "1" if bool(st.session_state.get("enable_semantic", True)) else "0"
    if "page" not in st.query_params:
        st.query_params["page"] = str(st.session_state.get("result_page", 0) or 0)

    if "detail" in st.query_params:
        del st.query_params["detail"]
    st.rerun()


def _get_related_papers(paper: dict, collection, reranker, bm25_engine, bm25_metadata) -> list[dict]:
    st.session_state.setdefault("detail_related_cache", {})

    paper_id = str(paper.get("paper_id", "")).strip()
    cache_key = paper_id or str(paper.get("title", "")).strip()
    if cache_key in st.session_state.detail_related_cache:
        return st.session_state.detail_related_cache[cache_key]

    query = _build_related_query(paper)
    if not query:
        return []

    related = hybrid_search(
        collection=collection,
        reranker=reranker,
        bm25_engine=bm25_engine,
        bm25_metadata=bm25_metadata,
        query=query,
        top_k=12,
        semantic_top_k=10,
        bm25_top_k=10,
        use_rerank=False,
    )

    filtered = []
    for item in related:
        item_id = str(item.get("paper_id", "")).strip()
        if paper_id and item_id == paper_id:
            continue
        filtered.append(item)
        if len(filtered) == 10:
            break

    st.session_state.detail_related_cache[cache_key] = filtered
    return filtered


def render_paper_detail_page(collection, reranker, bm25_engine, bm25_metadata) -> None:
    st.markdown(DETAIL_PAGE_STYLE, unsafe_allow_html=True)
    st.markdown(RESULT_CARD_STYLE, unsafe_allow_html=True)
    st.session_state.setdefault("detail_active_tab", "citations")

    paper = st.session_state.get("selected_paper")
    if not paper:
        st.warning("Paper information was not found.")
        if st.button("Back to search results"):
            _back_to_results()
        return

    title = escape(_normalize_text(paper.get("title", "Unknown Title")))
    raw_authors = paper.get("authors", "")
    if isinstance(raw_authors, list):
        authors_str = ", ".join(author.get("name", "") if isinstance(author, dict) else str(author) for author in raw_authors)
    else:
        authors_str = str(raw_authors) if raw_authors else "Unknown Authors"
    authors_str = escape(_normalize_text(authors_str))

    venue = escape(_normalize_text(paper.get("venue", "")))
    pub_date = escape(_normalize_text(paper.get("publication_date", "") or paper.get("year", "")))
    abstract = escape(_normalize_text(paper.get("abstract", "No abstract available.")))
    is_survey = bool(paper.get("is_survey", False))
    cite_count = paper.get("citation_count") or paper.get("cite_count")
    paper_id = str(paper.get("paper_id", "")).strip()

    doi = str(paper.get("doi", "")).strip()
    arxiv = str(paper.get("arxiv", "")).strip()
    s2_url = str(paper.get("s2_url", "")).strip()

    ref_titles = _parse_list_string(paper.get("reference_titles", ""))
    cit_titles = _parse_list_string(paper.get("citation_titles", ""))
    ref_count = paper.get("reference_count", len(ref_titles))
    cit_net_count = paper.get("citation_network_count", len(cit_titles))

    back_col, _ = st.columns([1, 6])
    with back_col:
        if st.button("Back to Results", width="stretch"):
            _back_to_results()

    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        st.markdown(f'<div class="detail-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="detail-authors">{authors_str}</div>', unsafe_allow_html=True)

        venue_html = f'<span class="venue">@{venue}</span>' if venue else ""
        date_html = f" · {pub_date}" if pub_date else ""
        if venue_html or date_html:
            st.markdown(f'<div class="detail-meta">{venue_html}{date_html}</div>', unsafe_allow_html=True)

        chips_html = ['<div class="chips">']
        if cite_count not in ("", None, "nan"):
            chips_html.append(f'<span class="chip">Citations: {escape(str(cite_count))}</span>')
        if ref_count not in ("", None, "nan"):
            chips_html.append(f'<span class="chip">References: {escape(str(ref_count))}</span>')
        chips_html.append('<span class="chip chip-survey">Survey / Review</span>' if is_survey else '<span class="chip">Paper</span>')
        if paper_id:
            chips_html.append(f'<span class="chip">ID: {escape(paper_id)}</span>')
        chips_html.append("</div>")
        st.markdown("".join(chips_html), unsafe_allow_html=True)

        links_html = ["<div>"]
        if arxiv and arxiv.lower() != "nan":
            links_html.append(f'<a href="https://arxiv.org/abs/{escape(arxiv)}" target="_blank" class="link-btn">arXiv</a>')
        if doi and doi.lower() != "nan":
            links_html.append(f'<a href="https://doi.org/{escape(doi)}" target="_blank" class="link-btn">DOI</a>')
        if s2_url and s2_url.lower() != "nan":
            links_html.append(f'<a href="{escape(s2_url)}" target="_blank" class="link-btn">Semantic Scholar</a>')
        links_html.append("</div>")
        if len(links_html) > 2:
            st.markdown("".join(links_html), unsafe_allow_html=True)

        st.markdown('<div class="section-title">Abstract</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="detail-abstract">{abstract}</div>', unsafe_allow_html=True)

        st.markdown('<div class="detail-tab-offset"></div>', unsafe_allow_html=True)
        tab_cols = st.columns(3, gap="medium")
        with tab_cols[0]:
            if st.button(
                f"{cit_net_count} Citations",
                key="detail_tab_citations",
                width="stretch",
                type="primary" if st.session_state.detail_active_tab == "citations" else "secondary",
            ):
                st.session_state.detail_active_tab = "citations"
                st.rerun()
        with tab_cols[1]:
            if st.button(
                f"{ref_count} References",
                key="detail_tab_references",
                width="stretch",
                type="primary" if st.session_state.detail_active_tab == "references" else "secondary",
            ):
                st.session_state.detail_active_tab = "references"
                st.rerun()
        with tab_cols[2]:
            if st.button(
                "Related Papers",
                key="detail_tab_related",
                width="stretch",
                type="primary" if st.session_state.detail_active_tab == "related" else "secondary",
            ):
                st.session_state.detail_active_tab = "related"
                st.rerun()

        active_tab = st.session_state.detail_active_tab
        if active_tab == "citations":
            st.markdown(f'<div class="section-title">Citations ({cit_net_count})</div>', unsafe_allow_html=True)
            if cit_titles:
                st.markdown(
                    "".join(f'<div class="list-item">• {escape(_normalize_text(item))}</div>' for item in cit_titles),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="paper-tab-empty">No citation data available.</div>', unsafe_allow_html=True)
        elif active_tab == "references":
            st.markdown(f'<div class="section-title">References ({ref_count})</div>', unsafe_allow_html=True)
            if ref_titles:
                st.markdown(
                    "".join(f'<div class="list-item">• {escape(_normalize_text(item))}</div>' for item in ref_titles),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="paper-tab-empty">No reference data available.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-title">Related Papers</div>', unsafe_allow_html=True)
            related_results = _get_related_papers(paper, collection, reranker, bm25_engine, bm25_metadata)
            if related_results:
                related_df = build_results_dataframe(related_results, is_semantic=False)
                st.markdown(
                    f'<div class="paper-tab-empty">Top {len(related_df)} related papers from hybrid search.</div>',
                    unsafe_allow_html=True,
                )
                for index, (_, row) in enumerate(related_df.iterrows()):
                    render_result_card(row, card_key=f"related_{paper_id}_{index}")
            else:
                st.markdown('<div class="paper-tab-empty">No suitable related papers found.</div>', unsafe_allow_html=True)

    with side_col:
        _render_detail_chat_section(paper)
