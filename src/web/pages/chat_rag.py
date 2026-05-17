from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime, timezone
from html import escape

import streamlit as st

from core.chat_store import (
    build_chat_title,
    create_chat_session,
    get_chat_session,
    load_chat_store,
    update_chat_store,
    upsert_chat_session,
)
from core.rag import AGENT_MODEL, run_agent_rag


CHAT_STYLE = """
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
.stApp {
    background: #ffffff !important;
}
section[data-testid="stSidebar"] {
    display: block !important;
    background: #ffffff !important;
    border-right: 1px solid #ececec;
    box-shadow: none !important;
    min-width: 264px !important;
    max-width: 264px !important;
}
section[data-testid="stSidebar"] > div:first-child {
    display: block !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    display: block !important;
    padding: 16px 12px 10px;
    background: #ffffff !important;
}
section[data-testid="stSidebar"] div.stButton > button {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    min-height: 36px;
    border-radius: 9px !important;
    margin-bottom: 2px;
    box-shadow: none !important;
    background: transparent !important;
    border: 0 !important;
    color: #111111 !important;
    font-size: 0.88rem !important;
    font-weight: 400 !important;
    padding: 0.36rem 0.55rem !important;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: #f0f0f0 !important;
    color: #111111 !important;
    border: 0 !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background: #f5f5f5 !important;
    color: #111111 !important;
    border: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stTextInput"] {
    margin: 2px 0 12px;
}
section[data-testid="stSidebar"] input {
    min-height: 36px !important;
    border-radius: 9px !important;
    border: 0 !important;
    background: transparent !important;
    color: #111111 !important;
    font-size: 0.88rem !important;
    padding-left: 0.55rem !important;
}
section[data-testid="stSidebar"] input:focus {
    background: #f5f5f5 !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] [data-baseweb="input"] {
    border: 0 !important;
    background: transparent !important;
}
#MainMenu, footer, header { visibility: hidden; }

.chat-topbar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 0 16px;
    border-bottom: 1px solid #dbe7fb;
    margin-bottom: 28px;
}
.chat-topbar-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #2f2923;
    margin: 0;
    line-height: 1;
}
.chat-topbar-badge {
    background: #eef5ff;
    color: #1a73e8;
    border: 1px solid #d0defb;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 12px;
    letter-spacing: 0.02em;
}

.chat-sidebar-brandrow {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin: 4px 4px 20px;
}
.chat-sidebar-brand {
    color: #050505;
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1;
}
.chat-sidebar-toggle {
    width: 22px;
    height: 22px;
    border: 1px solid #9a9a9a;
    border-radius: 6px;
    position: relative;
    opacity: 0.9;
}
.chat-sidebar-toggle::after {
    content: "";
    position: absolute;
    top: 4px;
    bottom: 4px;
    left: 8px;
    border-left: 1px solid #9a9a9a;
}
.chat-sidebar-divider {
    height: 1px;
    background: #eeeeee;
    margin: 10px 6px 11px;
}
.chat-sidebar-heading {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0;
    text-transform: none;
    color: #111111;
    margin: 0 6px 8px;
}
.chat-sidebar-preview {
    font-size: 0.84rem;
    color: #6b6b6b;
    line-height: 1.45;
    margin: 8px 6px;
}

div.stButton > button {
    background: #ffffff !important;
    color: #1a73e8 !important;
    border: 1px solid #d6e3fb !important;
    border-radius: 16px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.1rem !important;
    box-shadow: 0 4px 12px rgba(26,115,232,0.08) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    background: #eef5ff !important;
    border-color: #bfd5fb !important;
    color: #1557b0 !important;
}

.chat-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 64px 24px;
    text-align: center;
}
.chat-empty-icon {
    font-size: 2.4rem;
    margin-bottom: 16px;
    opacity: 0.18;
}
.chat-empty-title {
    font-size: 1rem;
    font-weight: 600;
    color: #3f342b;
    margin-bottom: 8px;
}
.chat-empty-sub {
    font-size: 0.85rem;
    color: #9b8e83;
    line-height: 1.7;
    max-width: 340px;
}

div[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 4px 0 !important;
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%);
    color: #ffffff !important;
    border-radius: 18px 4px 18px 18px;
    padding: 12px 16px !important;
    box-shadow: 0 6px 18px rgba(26,115,232,0.18);
    max-width: 72%;
    margin-left: auto;
}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
    color: #ffffff !important;
    margin: 0;
}

div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: #ffffff;
    border: 1px solid #dbe7fb;
    border-radius: 4px 18px 18px 18px;
    padding: 14px 18px !important;
    box-shadow: 0 8px 22px rgba(26,115,232,0.05);
    max-width: 80%;
    line-height: 1.65;
}
.answer-text {
    white-space: pre-wrap;
    word-break: break-word;
}

div[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #1a73e8, #1557b0) !important;
    border: none !important;
}
div[data-testid="chatAvatarIcon-assistant"] {
    background: #eef5ff !important;
    border: 1px solid #d0defb !important;
    color: #1a73e8 !important;
}

.sources-block {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #dbe7fb;
}
.sources-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9b9eb8;
    margin-bottom: 8px;
}
.sources-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: flex-start;
}
.source-pill {
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
.source-pill:hover {
    background: #eef4ff;
    border-color: #c9d8f2;
    transform: translateY(-1px);
}
.source-title {
    color: #2f2923;
    font-size: 0.78rem;
    font-weight: 600;
    line-height: 1.25;
    max-width: 280px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.source-chip-sim {
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

div[data-testid="stChatInput"] [data-baseweb="textarea"] {
    border: 1px solid #d0d7e2;
    border-radius: 20px;
    padding: 0 8px;
    background: #f3f5f9;
    transition: border-color 0.18s ease, background-color 0.18s ease;
    display: flex;
    align-items: center;
}

div[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.4 !important;
    border: none !important;
    background: transparent !important;
    color: #2f2923 !important;
    box-shadow: none !important;
    margin: 0 !important;
    padding: 0.45rem !important;
    width: 100% !important;
}

div[data-testid="stChatInput"] {
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}

div[data-testid="stChatInput"] > div {
    padding: 0 !important;
    margin: 0 !important;
}

div[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
}
</style>
"""


def _clear_chat_query_param() -> None:
    if "chat" in st.query_params:
        del st.query_params["chat"]


def _start_draft_chat() -> None:
    st.session_state.chat_active_id = ""
    st.session_state.chat_history = []
    st.session_state.chat_title = "New chat"
    st.session_state.chat_draft_active = True
    _clear_chat_query_param()


def _init_chat_session() -> dict | None:
    store = load_chat_store()
    requested_chat_id = str(st.query_params.get("chat", "") or "").strip()
    session_chat_id = str(st.session_state.get("chat_active_id", "") or "").strip()
    draft_active = bool(st.session_state.get("chat_draft_active", False))

    active_session = None
    if requested_chat_id:
        active_session = get_chat_session(store, requested_chat_id)
    if active_session is None and session_chat_id and not draft_active:
        active_session = get_chat_session(store, session_chat_id)
    if active_session is None and not requested_chat_id and not draft_active and store.get("sessions"):
        active_session = store["sessions"][0]
    if active_session is None:
        _start_draft_chat()
        return None

    st.session_state.chat_active_id = active_session["chat_id"]
    st.session_state.chat_history = deepcopy(active_session.get("messages", []))
    st.session_state.chat_title = active_session.get("title", "New chat")
    st.session_state.chat_draft_active = False

    if str(st.query_params.get("chat", "") or "").strip() != active_session["chat_id"]:
        st.query_params["chat"] = active_session["chat_id"]

    return active_session


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
            "</span>"
        )

    return _CHUNK_ANNOTATION_RE.sub(_replace, safe).replace("\n", "<br>")


def _sources_html(sources) -> str:
    items = ""

    for source in sources or []:
        title = escape(" ".join(str(source.get("title") or "Untitled").split()))
        chunk_id = escape(" ".join(str(source.get("chunk_id") or "").split()))
        metric_value = source.get("confidence")
        metric_label = "conf"
        if metric_value in (None, "", "nan"):
            metric_value = source.get("score")
            metric_label = "score"
        if metric_value in (None, "", "nan"):
            metric_value = source.get("source_score")
            metric_label = "score"

        row = f'<span class="source-title" title="{title}">{title}</span>'
        if chunk_id:
            row += f'<span class="source-chip-sim">{chunk_id}</span>'
        if metric_value not in (None, "", "nan"):
            try:
                metric_text = f"{float(metric_value):.3f}"
            except (TypeError, ValueError):
                metric_text = str(metric_value)
            row += f'<span class="source-chip-sim">{metric_label} {escape(metric_text)}</span>'

        items += f'<div class="source-pill">{row}</div>'

    return (
        '<div class="sources-block">'
        '<div class="sources-label">Sources retrieved</div>'
        f'<div class="sources-list">{items}</div>'
        "</div>"
    )


def _render_topbar() -> None:
    col_back, col_info = st.columns([1, 9])

    with col_back:
        st.write("")
        if st.button("< Home"):
            st.session_state.page = "home"
            st.rerun()

    with col_info:
        st.markdown(
            '<div class="chat-topbar">'
            '<span class="chat-topbar-title">Research Chat</span>'
            f'<span class="chat-topbar-badge">RAG / {AGENT_MODEL}</span>'
            "</div>",
            unsafe_allow_html=True,
        )


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="chat-empty">
            <div class="chat-empty-icon">RAG</div>
            <div class="chat-empty-title">Ask anything about the papers</div>
            <div class="chat-empty-sub">
                Answers are grounded strictly in your indexed knowledge base.<br>
                Sources are cited automatically with every reply.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        sources = message.get("sources", [])

        with st.chat_message(role):
            if role == "assistant":
                st.markdown(
                    f'<div class="answer-text">{_render_annotated_text(content)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(content)
            if role == "assistant" and sources:
                st.markdown(_sources_html(sources), unsafe_allow_html=True)


def _is_chat_pending(session: dict | None) -> bool:
    return bool(session and session.get("pending", False))


def _render_pending_response() -> None:
    with st.chat_message("assistant"):
        with st.spinner("Running agent RAG..."):
            st.caption("Answer is still being generated. You can switch chats and come back.")


def _build_chat_preview(session: dict) -> str:
    for message in reversed(session.get("messages", [])):
        text = " ".join(str(message.get("content", "")).split()).strip()
        if not text:
            continue
        return text if len(text) <= 72 else text[:69].rstrip() + "..."
    return "No messages yet"


def _format_session_meta(session: dict) -> str:
    messages = session.get("messages", []) or []
    updated_at = str(session.get("updated_at", "") or "").strip()
    if updated_at:
        updated_at = updated_at.replace("T", " ").replace("+00:00", " UTC")
    else:
        updated_at = "unknown time"

    message_count = len(messages)
    label = "message" if message_count == 1 else "messages"
    return f"{message_count} {label} · updated {updated_at}"


def _build_sidebar_sessions(sessions: list[dict], search_text: str) -> list[dict]:
    if not search_text:
        return sessions

    needle = search_text.casefold()
    filtered: list[dict] = []
    for session in sessions:
        haystack = " ".join(
            [
                str(session.get("title", "")),
                _build_chat_preview(session),
                " ".join(str(message.get("content", "")) for message in session.get("messages", [])),
            ]
        ).casefold()
        if needle in haystack:
            filtered.append(session)
    return filtered


def _set_active_chat(chat_id: str) -> None:
    chat_id = str(chat_id or "").strip()
    if not chat_id:
        return
    st.session_state.chat_active_id = chat_id
    st.query_params["chat"] = chat_id
    st.rerun()


def _create_new_chat() -> None:
    _start_draft_chat()
    st.rerun()


def _render_sidebar(active_session: dict | None) -> None:
    store = load_chat_store()
    st.sidebar.markdown(
        """
        <div class="chat-sidebar-brandrow">
            <div class="chat-sidebar-brand">ChatGPT</div>
            <div class="chat-sidebar-toggle" aria-hidden="true"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("New chat", use_container_width=True):
        _create_new_chat()

    search_text = st.sidebar.text_input(
        "Search chats",
        key="chat_sidebar_search",
        placeholder="Search chats",
        label_visibility="collapsed",
    ).strip()

    st.sidebar.markdown('<div class="chat-sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="chat-sidebar-heading">Recents</div>', unsafe_allow_html=True)

    sessions = _build_sidebar_sessions(store.get("sessions", []), search_text)
    if not sessions:
        st.sidebar.markdown(
            '<div class="chat-sidebar-preview">No matching chats.</div>',
            unsafe_allow_html=True,
        )
        return

    for index, session in enumerate(sessions, start=1):
        chat_id = str(session.get("chat_id", "")).strip()
        if not chat_id:
            continue

        is_active = bool(active_session) and chat_id == str(active_session.get("chat_id", "")).strip()
        button_label = str(session.get("title", "New chat")).strip() or "New chat"
        if button_label == "New chat":
            button_label = f"{button_label} {index}"
        if len(button_label) > 54:
            button_label = button_label[:51].rstrip() + "..."

        if st.sidebar.button(
            button_label,
            key=f"chat_open_{chat_id}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            _set_active_chat(chat_id)


def _assistant_message(answer: str, result: dict) -> dict:
    return {
        "role": "assistant",
        "content": answer,
        "sources": result.get("sources", []),
        "confidence": result.get("confidence", 0.0),
        "search_mode_used": result.get("search_mode_used", "hybrid"),
        "execution_path": result.get("execution_path", []),
        "success": result.get("success", True),
        "error": result.get("error", ""),
        "query": result.get("query", ""),
        "intent": result.get("intent", "unclear"),
        "execution_time_ms": result.get("execution_time_ms", 0),
        "external_search_triggered": result.get("external_search_triggered", False),
        "used_external_papers": result.get("used_external_papers", False),
    }


def _submit_user_message(question: str) -> tuple[str, list[dict]]:
    active_chat_id = str(st.session_state.get("chat_active_id", "") or "").strip()
    if st.session_state.get("chat_draft_active", False):
        active_chat_id = ""

    submitted: dict = {}
    print(
        "[CHAT_UI] submit_user_message start "
        f"active_chat_id={active_chat_id or '<draft>'} question={question[:80]!r}",
        flush=True,
    )

    def _mutate(store: dict) -> dict:
        session = get_chat_session(store, active_chat_id) if active_chat_id else None
        if session is None:
            session = create_chat_session()

        messages = list(session.get("messages", []))
        messages.append({"role": "user", "content": question})

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        session["messages"] = messages
        session["title"] = build_chat_title(messages)
        session["updated_at"] = now
        session["pending"] = True
        session["pending_question"] = question
        session["pending_started_at"] = now
        if not session.get("created_at"):
            session["created_at"] = now

        submitted["chat_id"] = session["chat_id"]
        submitted["title"] = session["title"]
        submitted["messages"] = deepcopy(messages)
        return upsert_chat_session(store, session)

    update_chat_store(_mutate)

    chat_id = str(submitted.get("chat_id", "")).strip()
    messages = submitted.get("messages", [])
    st.session_state.chat_active_id = chat_id
    st.session_state.chat_history = deepcopy(messages)
    st.session_state.chat_title = submitted.get("title", "New chat")
    st.session_state.chat_draft_active = False

    print(
        "[CHAT_UI] submit_user_message saved "
        f"submitted_chat_id={chat_id} messages={len(messages)} pending=True",
        flush=True,
    )
    return chat_id, deepcopy(messages)


def _append_assistant_response(chat_id: str, answer: str, result: dict) -> list[dict]:
    chat_id = str(chat_id or "").strip()
    if not chat_id:
        print("[CHAT_UI] append_assistant skipped: empty chat_id", flush=True)
        return []

    saved: dict = {}
    print(
        "[CHAT_UI] append_assistant start "
        f"chat_id={chat_id} answer_chars={len(answer or '')} success={result.get('success', True)}",
        flush=True,
    )

    def _mutate(store: dict) -> dict:
        previous_active_chat_id = store.get("active_chat_id")
        session = get_chat_session(store, chat_id)
        if session is None:
            return store

        messages = list(session.get("messages", []))
        message = _assistant_message(answer, result)
        if not message["query"]:
            message["query"] = next(
                (
                    str(existing.get("content", ""))
                    for existing in reversed(messages)
                    if str(existing.get("role", "")).strip() == "user"
                ),
                "",
            )
        messages.append(message)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        session["messages"] = messages
        session["title"] = build_chat_title(messages)
        session["updated_at"] = now
        session["pending"] = False
        session["pending_question"] = ""
        session["pending_started_at"] = ""
        saved["title"] = session["title"]
        saved["messages"] = deepcopy(messages)
        store = upsert_chat_session(store, session)
        store["active_chat_id"] = previous_active_chat_id
        return store

    update_chat_store(_mutate)

    messages = saved.get("messages", [])
    if str(st.session_state.get("chat_active_id", "") or "").strip() == chat_id:
        st.session_state.chat_history = deepcopy(messages)
        st.session_state.chat_title = saved.get("title", st.session_state.get("chat_title", "New chat"))

    print(
        "[CHAT_UI] append_assistant done "
        f"chat_id={chat_id} messages={len(messages)} active_chat_id={st.session_state.get('chat_active_id', '')}",
        flush=True,
    )
    return deepcopy(messages)


def render_chat_page() -> None:
    st.markdown(CHAT_STYLE, unsafe_allow_html=True)
    active_session = _init_chat_session()
    _render_sidebar(active_session)
    _render_topbar()

    if not st.session_state.chat_history:
        _render_empty_state()

    _render_chat_history()
    active_chat_pending = _is_chat_pending(active_session)
    if active_chat_pending:
        _render_pending_response()

    user_input = st.chat_input(
        "Ask a question about the research paper ...",
        disabled=active_chat_pending,
    )
    if not user_input or not user_input.strip():
        return

    question = user_input.strip()
    submitted_chat_id, _ = _submit_user_message(question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        print(
            "[CHAT_UI] run_agent_rag start "
            f"chat_id={submitted_chat_id} question={question[:80]!r}",
            flush=True,
        )
        with st.spinner("Running agent RAG..."):
            result = run_agent_rag(question)
            answer = result.get("answer", "")
            _append_assistant_response(submitted_chat_id, answer, result)
            print(
                "[CHAT_UI] run_agent_rag done_and_saved "
                f"chat_id={submitted_chat_id} success={result.get('success', True)} "
                f"elapsed_ms={result.get('execution_time_ms', 0)}",
                flush=True,
            )

        st.markdown(
            f'<div class="answer-text">{_render_annotated_text(answer)}</div>',
            unsafe_allow_html=True,
        )
        sources = result.get("sources", [])
        if sources:
            st.markdown(_sources_html(sources), unsafe_allow_html=True)
        if result.get("error") and not result.get("success", True):
            st.caption(f"Agent error: {result['error']}")
