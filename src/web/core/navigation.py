"""Explicit return destinations survive reference links and browser reloads."""
from urllib.parse import urlencode
import streamlit as st


def detail_context_params() -> str:
    origin = st.query_params.get("return_to", "results") if st.session_state.get("page") == "detail" else "results"
    if origin == "chat":
        return "&" + urlencode({"return_to": "chat", "chat": st.query_params.get("chat", "") or st.session_state.get("chat_active_id", "")})
    params = {"return_to": "results", "q": st.session_state.get("search_query", ""),
              "semantic": "1" if st.session_state.get("enable_semantic", True) else "0",
              "page": st.session_state.get("result_page", 0),
              "survey": "1" if st.session_state.get("search_only_survey", False) else "0"}
    years = st.session_state.get("search_year_range")
    if years:
        params.update(year_from=years[0], year_to=years[1])
    return "&" + urlencode(params)


def detail_back_label() -> str:
    return "Back to Chat" if st.query_params.get("return_to") == "chat" else "Back to Results"


def return_from_detail() -> None:
    origin = st.query_params.get("return_to", "results")
    params = dict(st.query_params)
    for key in ("detail", "return_to"):
        params.pop(key, None)
    if origin == "chat":
        chat_id = params.get("chat") or st.session_state.get("chat_active_id", "")
        params = {"chat": chat_id, "view": "chat"} if chat_id else {"view": "chat"}
        st.session_state.page = "chat_rag"
    else:
        params.pop("view", None)
        params.pop("chat", None)
        params.setdefault("q", st.session_state.get("search_query", ""))
        params.setdefault("semantic", "1" if st.session_state.get("enable_semantic", True) else "0")
        params.setdefault("page", str(st.session_state.get("result_page", 0)))
        st.session_state.page = "results"
    st.query_params.from_dict(params)
    st.rerun()
