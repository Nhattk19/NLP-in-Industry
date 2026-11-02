import math
from html import escape
from urllib.parse import quote

import altair as alt
import pandas as pd
import streamlit as st

from core.search import bm25_search, semantic_search
from core.navigation import detail_context_params
from core.config import CHROMA_PATH


RESULTS_PAGE_STYLE = ""  # Visual styles live in core/styles.css.

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
    return detail_context_params()


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
        chips.append(f'<span class="chip chip-cite">{escape(str(row["cite_count"]))} citations</span>')
    if row["is_survey"]:
        chips.append('<span class="chip chip-survey">Survey</span>')
    if row["score_lbl"] not in ("", None):
        chips.append(f'<span class="chip chip-score">score: {escape(str(row["score_lbl"]))}</span>')
    chips.append("</div>")

    tldr_block = (
        f"<div class='paper-tldr'><span class='tldr-label'>Abstract:</span>{escape(tldr)}</div>"
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
    st.html(card_html,)


def _render_filters(dataframe: pd.DataFrame) -> tuple[tuple[int, int] | None, bool]:
    df_with_year = dataframe.dropna(subset=["year_num"]).copy()
    if not df_with_year.empty:
        df_with_year["year_num"] = df_with_year["year_num"].astype(int)

    st.html('<div class="section-label">Filters</div>',)
    if not df_with_year.empty:
        year_min = int(df_with_year["year_num"].min())
        year_max = int(df_with_year["year_num"].max())
        if year_min == year_max:
            st.caption(f"Publication year: {year_min}")
            year_range = (year_min, year_max)
        else:
            saved = st.session_state.get("search_year_range") or (year_min, year_max)
            lower = max(year_min, min(int(st.query_params.get("year_from", saved[0])), year_max))
            upper = max(lower, min(int(st.query_params.get("year_to", saved[1])), year_max))
            year_range = st.slider("Publication year", year_min, year_max, (lower, upper), step=1)
            st.query_params["year_from"] = str(year_range[0])
            st.query_params["year_to"] = str(year_range[1])
    else:
        year_range = None
        st.caption("No year data available.")

    only_survey = st.checkbox("Survey", value=st.query_params.get("survey", "0") == "1")
    st.session_state.search_only_survey = only_survey
    st.session_state.search_year_range = year_range
    st.query_params["survey"] = "1" if only_survey else "0"

    st.write("")
    st.html('<div class="section-label">Year Distribution</div>',)
    if not df_with_year.empty:
        year_counts = df_with_year.groupby("year_num").size().reset_index(name="count")
        y_axis_max = max(5, math.ceil(int(year_counts["count"].max()) / 5) * 5)
        chart = (
            alt.Chart(year_counts)
            .mark_bar(color="#3276d2", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
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
            .properties(height=180, background="#ffffff")
            .configure_view(strokeWidth=0)
            .configure_axis(grid=False, labelColor="#526176", titleColor="#202d40", domainColor="#b8c7db", tickColor="#b8c7db")
        )
        st.altair_chart(chart, width="stretch", theme=None)
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
    safe_page = max(0, min(current_page, max_page))
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
    with pagination_col, st.container(key="pagination"):

        columns = st.columns(len(slots), gap="small")
        for column, slot in zip(columns, slots):
            with column:
                if slot == "prev":
                    if st.button("<", key="pg_prev", help="Previous page", disabled=(current_page == 0)):
                        st.session_state.result_page = current_page - 1
                        st.query_params["page"] = str(st.session_state.result_page)
                        st.rerun()
                elif slot == "next":
                    if st.button(">", key="pg_next", help="Next page", disabled=(current_page >= max_page)):
                        st.session_state.result_page = current_page + 1
                        st.query_params["page"] = str(st.session_state.result_page)
                        st.rerun()
                elif slot == "ellipsis":
                    st.html('<div class="pg-ellipsis">...</div>',)
                elif slot == current_page:
                    st.html(f'<div class="pg-active">{slot + 1}</div>',)
                elif st.button(str(slot + 1), key=f"pg_{slot}"):
                    st.session_state.result_page = slot
                    st.query_params["page"] = str(slot)
                    st.rerun()



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
    col_home, col_query, col_search, col_mode = st.columns([1, 6, 1.2, 1.6], gap="small", vertical_alignment="bottom")
    with col_home:
        if st.button("Home", width="stretch"):
            st.query_params.clear()
            st.session_state.page = "home"
            st.rerun()
    with col_query:
        query_input = st.text_input(
            "Search publications",
            value=st.session_state.search_query,
            label_visibility="visible",
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
        if st.button("Search", type="primary", width="stretch"):
            st.session_state.search_query = query_input
            st.session_state.result_page = 0
            st.session_state._cached_key = ""
            st.query_params["q"] = query_input
            st.query_params["semantic"] = "1" if st.session_state.enable_semantic else "0"
            st.query_params["page"] = "0"
            st.rerun()


@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def _cached_semantic_results(query: str, store_revision: tuple, _collection, _reranker):
    return semantic_search(_collection, _reranker, query)


def _load_results(collection, reranker, bm25_engine, bm25_metadata, query: str, is_semantic: bool):
    cache_key = f"{query}|{is_semantic}"
    if st.session_state._cached_key == cache_key:
        return st.session_state._cached_results

    if is_semantic:
        with st.spinner("Loading and reranking results..."):
            results = _cached_semantic_results(query, (CHROMA_PATH, collection.count()), collection, reranker)
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
        st.html(
            f'<div class="result-header">Publications found for <b>"{escape(query)}"</b> · {total} results · {mode}</div>'
            f'<div style="font-size:0.78rem;color:#9aa0a6;margin-bottom:10px;">Showing {start + 1}-{end} of {total}</div>',
        )

        for index, (_, row) in enumerate(filtered.iloc[start:end].iterrows()):
            render_result_card(row, card_key=f"{current_page}_{index}")

        _render_pagination(current_page, max_page)
