import math
from html import escape
from urllib.parse import quote

import altair as alt
import pandas as pd
import streamlit as st

from core.search import bm25_search, semantic_search


RESULTS_PAGE_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff;
    color: #2d2d2d;
}
[data-testid="stSidebarNavItems"] { display: none; }

div.stButton > button {
    background: #ffffff;
    color: #1a73e8;
    border: 1px solid #dadce0;
    border-radius: 6px;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.35rem 0.9rem;
}
div.stButton > button:hover {
    background: #f1f3f4;
    border-color: #c0c5cb;
}

.paper-card-wrap { margin-bottom: 18px; }
.paper-card-link {
    display: block;
    color: inherit;
    text-decoration: none !important;
}
.paper-card-link:visited,
.paper-card-link:hover,
.paper-card-link:focus,
.paper-card-link:active {
    color: inherit;
    text-decoration: none !important;
}
.paper-card-link * { text-decoration: none !important; }
.paper-card-link:hover .paper-card {
    background: #fbfdff;
    border-color: #cfe0ff;
    box-shadow: 0 14px 34px rgba(35, 88, 183, 0.10);
    transform: translateY(-1px);
}
.paper-card {
    width: 100%;
    background: #ffffff;
    border: 1px solid #e8eefb;
    border-radius: 18px;
    padding: 18px 18px 16px 18px;
    box-shadow: 0 10px 28px rgba(35, 88, 183, 0.06);
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}
.paper-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #1e63e8;
    margin: 0 0 12px 0;
    line-height: 1.45;
}
.paper-authors {
    font-size: 0.88rem;
    color: #2f5fb3;
    margin-bottom: 12px;
    font-weight: 500;
    line-height: 1.6;
}
.paper-authors .author-chip {
    display: inline-block;
    background: #f7faff;
    border: 1px solid #d7e4fb;
    border-radius: 8px;
    padding: 4px 12px;
    margin: 0 6px 6px 0;
    color: #2d5fbe;
}
.paper-meta {
    font-size: 0.83rem;
    color: #727d90;
    margin-bottom: 12px;
}
.paper-meta .venue { color: #1e63e8; font-weight: 600; }
.paper-meta .date { color: #7a8291; }
.paper-tldr {
    font-size: 0.92rem;
    color: #2d3441;
    line-height: 1.8;
    margin: 6px 0 14px 0;
}
.tldr-label { font-weight: 700; color: #1e63e8; margin-right: 4px; }
.chips { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; align-items: center; }
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
.chip-cite { background:#e8f7ec; color:#2e9b58; border-color:#bce2c7; font-weight:700; }
.chip-survey { background:#eef4ff; color:#1e63e8; border-color:#d0defb; font-weight:700; }
.chip-score { background:#fff3f1; color:#e16159; border-color:#f4c9c4; font-family:monospace; }
.section-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9aa0a6;
    margin-bottom: 10px;
}
.result-header { font-size: 0.88rem; color: #70757a; margin-bottom: 4px; }
.pg-host { width: 100%; }
.pg-wrap > div[data-testid="stHorizontalBlock"] {
    gap: 1px !important;
    justify-content: center !important;
}
.pg-wrap > div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
    flex: 0 0 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
}
.pg-wrap button {
    border-radius: 8px !important;
    width: 26px !important;
    height: 26px !important;
    padding: 0 !important;
    font-size: 0.72rem !important;
    min-width: unset !important;
    border: 1px solid #d9e4fb !important;
    color: #1e63e8 !important;
    background: #ffffff !important;
    box-shadow: none !important;
}
.pg-active {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    color: #ffffff;
    background: linear-gradient(180deg, #2b75ee 0%, #1e63e8 100%);
    border: 1px solid #1756ca;
    box-shadow:
        0 0 0 1px rgba(255, 255, 255, 0.7) inset,
        0 8px 18px rgba(30, 99, 232, 0.28);
}
.pg-ellipsis {
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    color: #6f7785;
}
</style>
"""

PAGE_SIZE = 10
RESULT_CARD_STYLE = RESULTS_PAGE_STYLE


def _normalize_text(value: object) -> str:
    return " ".join(str(value).split())


def _parse_year(item: dict) -> str:
    year = str(item.get("publication_date", "")).strip()
    if len(year) >= 4 and year[:4].isdigit():
        return year[:4]
    return ""


def _format_date(item: dict) -> str:
    publication_date = str(item.get("publication_date", "")).strip()
    return publication_date if publication_date else _parse_year(item)


def build_results_dataframe(results: list[dict], is_semantic: bool) -> pd.DataFrame:
    rows = []
    for item in results:
        row = item.copy()
        if "_score_lbl_override" in item:
            score_label = item.get("_score_lbl_override", "")
        elif is_semantic:
            score_label = f"rerank={item.get('rerank_score', 0)} dist={item.get('retrieve_score', 0):.3f}"
        else:
            score_label = f"{item.get('bm25_score', 0):.2f}"

        row["title"] = item.get("title", "Unknown Title")
        row["authors"] = item.get("authors", "")
        row["venue"] = item.get("venue", "")
        row["date"] = _format_date(item)
        row["year"] = _parse_year(item)
        row["abstract"] = item.get("abstract", "")
        row["is_survey"] = bool(item.get("is_survey", False))
        row["cite_count"] = item.get("citation_count", "")
        row["score_lbl"] = score_label
        row["doi"] = item.get("doi", "")
        row["arxiv"] = item.get("arxiv", "")
        row["nlp_score"] = item.get("nlp_score", "")
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    dataframe["year_num"] = pd.to_numeric(dataframe["year"], errors="coerce")
    return dataframe


def _build_detail_key(row: pd.Series, card_key: str) -> str:
    for field in ("doi", "arxiv", "paper_id", "title"):
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return card_key


def _results_context_params() -> str:
    query = quote(str(st.session_state.get("search_query", "") or ""), safe="")
    semantic = "1" if bool(st.session_state.get("enable_semantic", True)) else "0"
    page = quote(str(st.session_state.get("result_page", 0) or 0), safe="")
    return f"&q={query}&semantic={semantic}&page={page}" if query else f"&semantic={semantic}&page={page}"


def _build_author_chips(authors: object) -> str:
    if isinstance(authors, list):
        names = [author.get("name", "") if isinstance(author, dict) else str(author) for author in authors]
    else:
        names = [name.strip() for name in str(authors).split(",") if name.strip()] if authors else []
    return "".join(f'<span class="author-chip">{escape(name)}</span>' for name in names if name)


def render_result_card(row: pd.Series, card_key: str) -> None:
    title = escape(_normalize_text(row["title"]))
    venue = escape(_normalize_text(row["venue"])) if row["venue"] else ""
    date = escape(_normalize_text(row["date"])) if row["date"] else ""
    abstract = _normalize_text(row["abstract"]) if row["abstract"] else ""
    tldr = abstract[:220].rstrip() + ("..." if len(abstract) > 220 else "") if abstract else ""

    venue_html = f'<span class="venue">@{venue}</span>' if venue else ""
    date_html = f'<span class="date">{date}</span>' if date else ""
    meta_separator = " · " if venue_html and date_html else ""
    meta_line = (
        f'<div class="paper-meta">{venue_html}{meta_separator}{date_html}</div>'
        if (venue_html or date_html)
        else ""
    )

    chips = ['<div class="chips">']
    if row["cite_count"] not in ("", None):
        chips.append(f'<span class="chip chip-cite">{escape(str(row["cite_count"]))}</span>')
    if row["is_survey"]:
        chips.append('<span class="chip chip-survey">Survey</span>')
    if row["score_lbl"] not in ("", None):
        chips.append(f'<span class="chip chip-score">score: {escape(str(row["score_lbl"]))}</span>')
    chips.append("</div>")

    tldr_block = (
        f"<div class='paper-tldr'><span class='tldr-label'>TLDR:</span>{escape(tldr)}</div>"
        if tldr
        else ""
    )
    detail_key = quote(_build_detail_key(row, card_key), safe="")
    author_chips = _build_author_chips(row["authors"])

    card_html = (
        '<div class="paper-card-wrap">'
        f'<a class="paper-card-link" href="?detail={detail_key}{_results_context_params()}" target="_self">'
        '<div class="paper-card">'
        f'<div class="paper-title">{title}</div>'
        f'<div class="paper-authors">{author_chips}</div>'
        f"{meta_line}{tldr_block}{''.join(chips)}"
        "</div></a></div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


def _render_filters(dataframe: pd.DataFrame) -> tuple[tuple[int, int] | None, bool]:
    df_with_year = dataframe.dropna(subset=["year_num"]).copy()
    if not df_with_year.empty:
        df_with_year["year_num"] = df_with_year["year_num"].astype(int)

    st.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)
    if not df_with_year.empty:
        year_min = int(df_with_year["year_num"].min())
        year_max = int(df_with_year["year_num"].max())
        year_range = st.slider("Publication year", year_min, year_max, (year_min, year_max), step=1)
    else:
        year_range = None
        st.caption("No year data available.")

    only_survey = st.checkbox("Survey")

    st.write("")
    st.markdown('<div class="section-label">Year Distribution</div>', unsafe_allow_html=True)
    if not df_with_year.empty:
        year_counts = df_with_year.groupby("year_num").size().reset_index(name="count")
        y_axis_max = max(5, math.ceil(int(year_counts["count"].max()) / 5) * 5)
        chart = (
            alt.Chart(year_counts)
            .mark_bar(color="#1a73e8", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("year_num:O", axis=alt.Axis(title=None, labelAngle=-45, labelFontSize=9)),
                y=alt.Y(
                    "count:Q",
                    axis=alt.Axis(title=None, values=list(range(0, y_axis_max + 1, 5)), labelFontSize=9),
                    scale=alt.Scale(domain=[0, y_axis_max]),
                ),
                tooltip=[
                    alt.Tooltip("year_num:O", title="Year"),
                    alt.Tooltip("count:Q", title="Papers"),
                ],
            )
            .properties(height=180)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.caption("No year data available.")

    return year_range, only_survey


def _apply_filters(
    dataframe: pd.DataFrame,
    year_range: tuple[int, int] | None,
    only_survey: bool,
) -> pd.DataFrame:
    filtered = dataframe.copy()
    if year_range:
        filtered = filtered[
            filtered["year_num"].isna()
            | ((filtered["year_num"] >= year_range[0]) & (filtered["year_num"] <= year_range[1]))
        ]
    if only_survey:
        filtered = filtered[filtered["is_survey"] == True]
    return filtered.reset_index(drop=True)


def _paginate(total_items: int, current_page: int) -> tuple[int, int, int, int]:
    max_page = max(0, (total_items - 1) // PAGE_SIZE)
    safe_page = min(current_page, max_page)
    start = safe_page * PAGE_SIZE
    end = min(start + PAGE_SIZE, total_items)
    return max_page, safe_page, start, end


def _render_pagination(current_page: int, max_page: int) -> None:
    if max_page <= 0:
        return

    pages_to_show = sorted(
        set([0] + list(range(max(0, current_page - 1), min(max_page + 1, current_page + 2))) + [max_page])
    )
    slots: list[str | int] = ["prev"]
    previous_page = None
    for page in pages_to_show:
        if previous_page is not None and page - previous_page > 1:
            slots.append("ellipsis")
        slots.append(page)
        previous_page = page
    slots.append("next")

    _, pagination_col, _ = st.columns([1, max(2, len(slots)), 1])
    with pagination_col:
        st.markdown('<div class="pg-host"><div class="pg-wrap">', unsafe_allow_html=True)
        columns = st.columns(len(slots), gap="small")
        for column, slot in zip(columns, slots):
            with column:
                if slot == "prev":
                    if st.button("<", key="pg_prev", disabled=(current_page == 0)):
                        st.session_state.result_page = current_page - 1
                        st.query_params["page"] = str(st.session_state.result_page)
                        st.rerun()
                elif slot == "next":
                    if st.button(">", key="pg_next", disabled=(current_page >= max_page)):
                        st.session_state.result_page = current_page + 1
                        st.query_params["page"] = str(st.session_state.result_page)
                        st.rerun()
                elif slot == "ellipsis":
                    st.markdown('<div class="pg-ellipsis">...</div>', unsafe_allow_html=True)
                elif slot == current_page:
                    st.markdown(f'<div class="pg-active">{slot + 1}</div>', unsafe_allow_html=True)
                elif st.button(str(slot + 1), key=f"pg_{slot}"):
                    st.session_state.result_page = slot
                    st.query_params["page"] = str(slot)
                    st.rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)


def _init_results_session() -> None:
    st.session_state.setdefault("search_query", "")
    st.session_state.setdefault("enable_semantic", True)
    st.session_state.setdefault("result_page", 0)
    st.session_state.setdefault("_cached_results", None)
    st.session_state.setdefault("_cached_key", "")

    query_param = str(st.query_params.get("q", "") or "").strip()
    if query_param:
        st.session_state.search_query = query_param

    semantic_param = str(st.query_params.get("semantic", "") or "").strip().lower()
    if semantic_param in {"0", "false", "lexical"}:
        st.session_state.enable_semantic = False
    elif semantic_param in {"1", "true", "semantic"}:
        st.session_state.enable_semantic = True

    page_param = str(st.query_params.get("page", "") or "").strip()
    if page_param.isdigit():
        st.session_state.result_page = int(page_param)


def _render_search_controls() -> None:
    col_home, col_query, col_mode, col_search = st.columns([1, 5, 2, 1])
    with col_home:
        if st.button("Home", use_container_width=True):
            st.query_params.clear()
            st.session_state.page = "home"
            st.rerun()
    with col_query:
        query_input = st.text_input(
            "q",
            value=st.session_state.search_query,
            label_visibility="collapsed",
            placeholder="Search publications...",
        )
    with col_mode:
        label = "Semantic" if st.session_state.enable_semantic else "Lexical"
        is_semantic = st.toggle(label, value=st.session_state.enable_semantic)
        if is_semantic != st.session_state.enable_semantic:
            st.session_state.enable_semantic = is_semantic
            st.session_state.result_page = 0
            st.session_state._cached_key = ""
            st.query_params["q"] = query_input
            st.query_params["semantic"] = "1" if is_semantic else "0"
            st.query_params["page"] = "0"
            st.rerun()
    with col_search:
        if st.button("Search", use_container_width=True):
            st.session_state.search_query = query_input
            st.session_state.result_page = 0
            st.session_state._cached_key = ""
            st.query_params["q"] = query_input
            st.query_params["semantic"] = "1" if st.session_state.enable_semantic else "0"
            st.query_params["page"] = "0"
            st.rerun()


def _load_results(collection, reranker, bm25_engine, bm25_metadata, query: str, is_semantic: bool):
    cache_key = f"{query}|{is_semantic}"
    if st.session_state._cached_key == cache_key:
        return st.session_state._cached_results

    if is_semantic:
        with st.spinner("Loading and reranking results..."):
            results = semantic_search(collection, reranker, query)
    else:
        if not bm25_engine:
            st.error("BM25 engine is not available.")
            return None
        with st.spinner("Searching with BM25..."):
            results = bm25_search(bm25_engine, bm25_metadata, query)

    st.session_state._cached_results = results
    st.session_state._cached_key = cache_key
    return results


def render_results_page(collection, reranker, bm25_engine, bm25_metadata) -> None:
    st.markdown(RESULTS_PAGE_STYLE, unsafe_allow_html=True)
    _init_results_session()
    _render_search_controls()
    st.divider()

    query = st.session_state.search_query
    if not query:
        st.warning("Please enter a keyword in the search box above.")
        return

    is_semantic = st.session_state.enable_semantic
    results = _load_results(collection, reranker, bm25_engine, bm25_metadata, query, is_semantic)
    if results is None:
        return
    if not results:
        st.warning("No matching results found.")
        return

    dataframe = build_results_dataframe(results, is_semantic)
    left_col, right_col = st.columns([1, 3], gap="large")

    with left_col:
        year_range, only_survey = _render_filters(dataframe)

    with right_col:
        filtered = _apply_filters(dataframe, year_range, only_survey)
        total = len(filtered)
        if total == 0:
            st.warning("No papers match the current filters.")
            return

        max_page, current_page, start, end = _paginate(total, st.session_state.result_page)
        if current_page != st.session_state.result_page:
            st.session_state.result_page = current_page
        mode = "Semantic Search" if is_semantic else "Lexical (BM25)"
        st.markdown(
            f'<div class="result-header">Publications found for <b>"{escape(query)}"</b> · {total} results · {mode}</div>'
            f'<div style="font-size:0.78rem;color:#9aa0a6;margin-bottom:10px;">Showing {start + 1}-{end} of {total}</div>',
            unsafe_allow_html=True,
        )

        for index, (_, row) in enumerate(filtered.iloc[start:end].iterrows()):
            render_result_card(row, card_key=f"{current_page}_{index}")

        _render_pagination(current_page, max_page)
