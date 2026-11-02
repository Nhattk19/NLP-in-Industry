import re
from html import escape

import streamlit as st

from core.search import hybrid_search
from core.navigation import detail_back_label, return_from_detail
from core.detail_rag import paper_has_fulltext_chunks, run_detail_rag
from core.references import normalize_references
from pages.results import RESULT_CARD_STYLE, build_results_dataframe, render_result_card


DETAIL_PAGE_STYLE = ""  # Visual styles live in core/styles.css.


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


def _detail_chat_pending_key(paper_id: str) -> str:
    return f"detail_chat_pending::{paper_id}"


_CHUNK_ANNOTATION_RE = re.compile(r"\[Chunk:\s*([^\]]+)\]", re.IGNORECASE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _render_inline_markdown(text: str) -> str:
    safe = escape(text.strip())
    safe = _BOLD_RE.sub(r"<strong>\1</strong>", safe)
    return safe.replace("**", "").replace("*", "")


def _render_annotated_text(content: object) -> str:
    text = _CHUNK_ANNOTATION_RE.sub("", str(content or ""))
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    lines = text.strip().splitlines()
    html_parts = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        bullet_match = re.match(r"^[*\-]\s+(.*)$", stripped)
        if bullet_match:
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{_render_inline_markdown(bullet_match.group(1))}</li>")
            continue

        if in_list:
            html_parts.append("</ul>")
            in_list = False
        html_parts.append(f"<p>{_render_inline_markdown(stripped)}</p>")

    if in_list:
        html_parts.append("</ul>")

    return "".join(html_parts)


def _init_detail_chat_session(paper_id: str) -> list[dict]:
    st.session_state.setdefault("detail_chat_history", {})
    st.session_state.detail_chat_history.setdefault(paper_id, [])
    return st.session_state.detail_chat_history[paper_id]


def _paper_chat_message_html(message: dict) -> str:
    role = message.get("role", "assistant")
    if role == "assistant":
        message = dict(message)
        message["content"], message["sources"] = normalize_references(
            message.get("content", ""), message.get("sources", []))
    role_class = "paper-chat-row-user" if role == "user" else "paper-chat-row-assistant"
    content = _render_annotated_text(message.get("content", ""))
    return (
        f'<div class="paper-chat-row {role_class}">'
        f'<div class="paper-chat-bubble">{content}</div>'
        '</div>'
    )


def _paper_chat_typing_html() -> str:
    return (
        '<div class="paper-chat-row paper-chat-row-assistant">'
        '<div class="paper-chat-bubble paper-chat-typing" aria-label="Assistant is typing">'
        '<span class="paper-chat-dot"></span>'
        '<span class="paper-chat-dot"></span>'
        '<span class="paper-chat-dot"></span>'
        '</div>'
        '</div>'
    )


def _render_detail_chat_frame(history: list[dict], is_generating: bool = False, unavailable: bool = False) -> None:
    if unavailable:
        body_html = (
            '<div class="paper-chat-empty">'
            'This paper does not yet support Q&A.'
            '</div>'
        )
    elif history or is_generating:
        messages = []
        if is_generating:
            messages.append(_paper_chat_typing_html())
        messages.extend(_paper_chat_message_html(message) for message in reversed(history))
        body_html = "".join(messages)
    else:
        body_html = (
            '<div class="paper-chat-empty">'
            'Ask a question and the assistant will answer using chunks from this selected paper.'
            '</div>'
        )

    st.html(
        f"""
        <div class="paper-chat-frame">
            <div class="paper-chat-header">
                <div class="paper-chat-title">Paper Chat</div>
                <div class="paper-chat-description">Ask questions about this selected paper</div>
            </div>
            <div class="paper-chat-messages" aria-live="polite">
                <div class="paper-chat-messages-inner">
                    {body_html}
                </div>
            </div>
        </div>
        """,
    )


def _render_detail_chat_unavailable() -> None:
    with st.container(key="paper_chat_panel"):
        _render_detail_chat_frame([], unavailable=True)


def _render_detail_chat_panel(paper: dict) -> None:
    paper_id = str(paper.get("paper_id", "")).strip()
    history = _init_detail_chat_session(paper_id)
    pending_key = _detail_chat_pending_key(paper_id)
    pending_question = str(st.session_state.get(pending_key, "") or "").strip()

    with st.container(key="paper_chat_panel"):
        _render_detail_chat_frame(history, is_generating=bool(pending_question))

        with st.form(
            key=f"detail_rag_form::{paper_id}",
            clear_on_submit=True,
            enter_to_submit=False,
            border=False,
        ):
            question = st.text_input(
                "Ask this paper",
                placeholder="Ask about methods, results, limitations...",
                disabled=bool(pending_question),
            )
            submitted = st.form_submit_button(
                "Send", type="primary", width="stretch",
                disabled=bool(pending_question),
            )

    if pending_question:
        result = run_detail_rag(paper, pending_question)
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
        del st.session_state[pending_key]
        st.rerun()

    if not submitted:
        return

    question = (question or "").strip()
    if not question:
        st.warning("Please type a question first.")
        return

    history.append({"role": "user", "content": question})
    st.session_state[pending_key] = question
    st.rerun()


def _render_detail_chat_section(paper: dict) -> None:
    paper_id = str(paper.get("paper_id", "")).strip()
    if paper_has_fulltext_chunks(paper_id):
        _render_detail_chat_panel(paper)
        return

    _render_detail_chat_unavailable()


def _back_to_results() -> None:
    return_from_detail()


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
    st.session_state.setdefault("detail_active_tab", "citations")

    paper = st.session_state.get("selected_paper")
    if not paper:
        st.warning("Paper information was not found.")
        if st.button(detail_back_label(), key="missing_detail_back"):
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
    doi_url = doi if doi.startswith(("https://", "http://")) else f"https://doi.org/{doi}"
    arxiv = str(paper.get("arxiv", "")).strip()
    arxiv_url = arxiv if arxiv.startswith(("https://", "http://")) else f"https://arxiv.org/abs/{arxiv}"
    s2_url = str(paper.get("s2_url", "")).strip()

    ref_titles = _parse_list_string(paper.get("reference_titles", ""))
    cit_titles = _parse_list_string(paper.get("citation_titles", ""))
    ref_count = paper.get("reference_count", len(ref_titles))
    cit_net_count = paper.get("citation_network_count", len(cit_titles))

    with st.container(key="detail_navigation"):
        if st.button(detail_back_label(), key="detail_back", width="content"):
            return_from_detail()

    main_col, side_col = st.columns([2, 1], gap="large")

    with main_col:
        st.html(f'<div class="detail-title">{title}</div>',)
        st.html(f'<div class="detail-authors">{authors_str}</div>',)

        venue_html = f'<span class="venue">@{venue}</span>' if venue else ""
        date_html = f" · {pub_date}" if pub_date else ""
        if venue_html or date_html:
            st.html(f'<div class="detail-meta">{venue_html}{date_html}</div>',)

        chips_html = ['<div class="chips">']
        if cite_count not in ("", None, "nan"):
            chips_html.append(f'<span class="chip">Citations: {escape(str(cite_count))}</span>')
        if ref_count not in ("", None, "nan"):
            chips_html.append(f'<span class="chip">References: {escape(str(ref_count))}</span>')
        chips_html.append('<span class="chip chip-survey">Survey / Review</span>' if is_survey else '<span class="chip">Paper</span>')
        if paper_id:
            chips_html.append(f'<span class="chip">ID: {escape(paper_id)}</span>')
        chips_html.append("</div>")
        st.html("".join(chips_html),)

        links_html = ["<div>"]
        if arxiv and arxiv.lower() != "nan":
            links_html.append(f'<a href="{escape(arxiv_url, quote=True)}" target="_blank" class="link-btn">arXiv</a>')
        if doi and doi.lower() != "nan":
            links_html.append(f'<a href="{escape(doi_url, quote=True)}" target="_blank" class="link-btn">DOI</a>')
        if s2_url and s2_url.lower() != "nan":
            links_html.append(f'<a href="{escape(s2_url)}" target="_blank" class="link-btn">Semantic Scholar</a>')
        links_html.append("</div>")
        if len(links_html) > 2:
            st.html("".join(links_html),)

        st.html('<div class="section-title">Abstract</div>',)
        st.html(f'<div class="detail-abstract">{abstract}</div>',)

        st.html('<div class="detail-tab-offset"></div>',)
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
            st.html(f'<div class="section-title">Citations ({cit_net_count})</div>',)
            if cit_titles:
                st.html(
                    "".join(f'<div class="list-item">• {escape(_normalize_text(item))}</div>' for item in cit_titles),
                )
            else:
                st.html('<div class="paper-tab-empty">No citation data available.</div>',)
        elif active_tab == "references":
            st.html(f'<div class="section-title">References ({ref_count})</div>',)
            if ref_titles:
                st.html(
                    "".join(f'<div class="list-item">• {escape(_normalize_text(item))}</div>' for item in ref_titles),
                )
            else:
                st.html('<div class="paper-tab-empty">No reference data available.</div>',)
        else:
            st.html('<div class="section-title">Related Papers</div>',)
            related_results = _get_related_papers(paper, collection, reranker, bm25_engine, bm25_metadata)
            if related_results:
                related_df = build_results_dataframe(related_results, is_semantic=False)
                st.html(
                    f'<div class="paper-tab-empty">Top {len(related_df)} related papers from hybrid search.</div>',
                )
                for index, (_, row) in enumerate(related_df.iterrows()):
                    render_result_card(row, card_key=f"related_{paper_id}_{index}")
            else:
                st.html('<div class="paper-tab-empty">No suitable related papers found.</div>',)

    with side_col:
        _render_detail_chat_section(paper)
