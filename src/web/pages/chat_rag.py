from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
from urllib.parse import quote

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

.chat-message-row {
    display: flex;
    width: 100%;
    margin: 8px 0;
}
.chat-message-row-user {
    justify-content: flex-end;
}
.chat-message-row-assistant {
    justify-content: flex-start;
}
.chat-message-stack {
    max-width: 80%;
}
.chat-message-row-user .chat-message-stack {
    max-width: 72%;
}
.chat-message-bubble {
    border-radius: 17px;
    padding: 10px 13px;
    line-height: 1.55;
    font-size: 0.92rem;
    overflow-wrap: anywhere;
}
.chat-message-bubble p {
    margin: 0 0 8px;
}
.chat-message-bubble p:last-child {
    margin-bottom: 0;
}
.chat-message-bubble ul {
    margin: 8px 0 8px 1.1rem;
    padding: 0;
}
.chat-message-bubble li {
    margin: 5px 0;
    padding-left: 2px;
}
.chat-message-bubble strong {
    font-weight: 700;
}
.chat-message-row-user .chat-message-bubble {
    color: #ffffff;
    background: #1a73e8;
    border-bottom-right-radius: 6px;
    box-shadow: 0 8px 18px rgba(26, 115, 232, 0.18);
}
.chat-message-row-assistant .chat-message-bubble {
    color: #2f2923;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-bottom-left-radius: 6px;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
}
.answer-text {
    white-space: pre-wrap;
    word-break: break-word;
}

.chat-typing {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    min-width: 48px;
}
.chat-typing-dot {
    width: 6px;
    height: 6px;
    border-radius: 999px;
    background: #9ca3af;
    animation: chat-typing-pulse 1s infinite ease-in-out;
}
.chat-typing-dot:nth-child(2) { animation-delay: 0.15s; }
.chat-typing-dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes chat-typing-pulse {
    0%, 80%, 100% { opacity: 0.35; transform: translateY(0); }
    40% { opacity: 1; transform: translateY(-2px); }
}

.sources-block {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #dbe7fb;
}
.sources-label {
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #2f2923;
    margin-bottom: 8px;
}
.sources-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    align-items: stretch;
}
.source-row {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 5px;
    align-items: baseline;
    min-height: 20px;
    padding: 0;
    background: transparent;
    border: 0;
    border-radius: 0;
    box-shadow: none;
    transition: color 0.18s ease;
    max-width: 100%;
    color: #2f2923 !important;
    text-decoration: none !important;
}
.source-row:hover {
    color: #1a73e8 !important;
}
.source-row:hover .source-title,
.source-row:hover .source-index {
    color: #1a73e8;
}
.source-index {
    color: #5f6b86;
    font-size: 0.76rem;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}
.source-title {
    display: block;
    color: #2f2923;
    font-size: 0.76rem;
    font-weight: 500;
    line-height: 1.35;
    min-width: 0;
    max-width: 100%;
    overflow-wrap: anywhere;
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

div[data-testid="stChatInput"] {
    background: #ffffff !important;
    border-top: 1px solid #eef2f7 !important;
    padding: 12px 0 16px !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
}

div[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

div[data-testid="stChatInput"] [data-baseweb="textarea"] {
    min-height: 44px !important;
    border: 1px solid #d7dde8 !important;
    border-radius: 14px !important;
    padding: 0 44px 0 12px !important;
    background: #f8fafc !important;
    box-shadow: none !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease;
    display: flex !important;
    align-items: center !important;
}

div[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
    border-color: #1a73e8 !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.12) !important;
}

div[data-testid="stChatInput"] textarea {
    min-height: 42px !important;
    height: 42px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.4 !important;
    border: 0 !important;
    outline: 0 !important;
    background: transparent !important;
    color: #2f2923 !important;
    box-shadow: none !important;
    margin: 0 !important;
    padding: 10px 0 !important;
    width: 100% !important;
    resize: none !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

div[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"] {
    width: 34px !important;
    height: 34px !important;
    min-width: 34px !important;
    min-height: 34px !important;
    border: 0 !important;
    border-radius: 10px !important;
    background: #e9eef6 !important;
    color: #526071 !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin-right: 5px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:not(:disabled) {
    background: #1a73e8 !important;
    color: #ffffff !important;
}

div[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:not(:disabled):hover {
    background: #dbeafe !important;
    color: #1a73e8 !important;
}

div[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:focus,
div[data-testid="stChatInput"] [data-testid="stChatInputSubmitButton"]:focus-visible {
    outline: 0 !important;
    box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.16) !important;
}

div[data-testid="stChatInput"] svg {
    width: 18px !important;
    height: 18px !important;
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
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _render_inline_markdown(text: str) -> str:
    safe = escape(text.strip())
    safe = _BOLD_RE.sub(r"<strong>\1</strong>", safe)
    return safe.replace("**", "").replace("*", "")


def _render_annotated_text(content: object, sources: list | None = None) -> str:
    chunk_ref_map = {}
    for index, source in enumerate(sources or [], 1):
        chunk_id = str(source.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        ref_number = source.get("ref_number") or index
        chunk_ref_map[chunk_id] = f"[{ref_number}]"

    def _replace_chunk_annotation(match: re.Match) -> str:
        chunk_id = match.group(1).strip()
        return chunk_ref_map.get(chunk_id, "")

    text = _CHUNK_ANNOTATION_RE.sub(_replace_chunk_annotation, str(content or ""))
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


def _sources_html(sources) -> str:
    items = ""

    for index, source in enumerate(sources or [], 1):
        title = escape(" ".join(str(source.get("title") or "Untitled").split()))
        detail_key = str(source.get("paper_id") or source.get("title") or "").strip()
        ref_number = escape(str(source.get("ref_number") or index))
        row = (
            f'<span class="source-index">[{ref_number}]</span>'
            f'<span class="source-title" title="{title}">{title}</span>'
        )
        if detail_key:
            href = f"?detail={quote(detail_key, safe='')}"
            items += f'<a class="source-row" href="{href}" target="_self">{row}</a>'
        else:
            items += f'<div class="source-row">{row}</div>'

    if not items:
        return ""

    return (
        '<div class="sources-block">'
        '<div class="sources-label">REFERENCES</div>'
        f'<div class="sources-list">{items}</div>'
        "</div>"
    )


def _chat_message_html(role: str, content: object, sources: list | None = None) -> str:
    role = "user" if str(role).strip() == "user" else "assistant"
    row_class = "chat-message-row-user" if role == "user" else "chat-message-row-assistant"
    sources_html = _sources_html(sources) if role == "assistant" and sources else ""
    body = (
        escape(str(content or "")).replace("\n", "<br>")
        if role == "user"
        else _render_annotated_text(content, sources)
    )
    return (
        f'<div class="chat-message-row {row_class}">'
        '<div class="chat-message-stack">'
        f'<div class="chat-message-bubble">{body}{sources_html}</div>'
        "</div>"
        "</div>"
    )


def _chat_typing_html() -> str:
    return (
        '<div class="chat-message-row chat-message-row-assistant">'
        '<div class="chat-message-stack">'
        '<div class="chat-message-bubble chat-typing" aria-label="Assistant is typing">'
        '<span class="chat-typing-dot"></span>'
        '<span class="chat-typing-dot"></span>'
        '<span class="chat-typing-dot"></span>'
        "</div>"
        "</div>"
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
        st.markdown(
            _chat_message_html(
                str(message.get("role", "assistant")),
                message.get("content", ""),
                message.get("sources", []),
            ),
            unsafe_allow_html=True,
        )


def _is_chat_pending(session: dict | None) -> bool:
    return bool(session and session.get("pending", False))


def _render_pending_response() -> None:
    st.markdown(_chat_typing_html(), unsafe_allow_html=True)


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

    if st.sidebar.button("New chat", use_container_width=True):
        _create_new_chat()

    st.sidebar.markdown('<div class="chat-sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="chat-sidebar-heading">Recents</div>', unsafe_allow_html=True)

    sessions = _build_sidebar_sessions(store.get("sessions", []), "")
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
        "standalone_question": result.get("standalone_question", ""),
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
    submitted_store = load_chat_store()
    submitted_session = get_chat_session(submitted_store, submitted_chat_id)
    submitted_messages = list(submitted_session.get("messages", [])) if submitted_session else []
    previous_messages = list(submitted_messages)
    if previous_messages:
        last_message = previous_messages[-1]
        if (
            str(last_message.get("role", "")).strip() == "user"
            and str(last_message.get("content", "")).strip() == question
        ):
            previous_messages = previous_messages[:-1]

    st.markdown(_chat_message_html("user", question), unsafe_allow_html=True)

    assistant_placeholder = st.empty()
    assistant_placeholder.markdown(_chat_typing_html(), unsafe_allow_html=True)
    print(
        "[CHAT_UI] run_agent_rag start "
        f"chat_id={submitted_chat_id} question={question[:80]!r}",
        flush=True,
    )
    result = run_agent_rag(
        question,
        chat_history=previous_messages,
        chat_id=submitted_chat_id,
    )
    answer = result.get("answer", "")
    _append_assistant_response(submitted_chat_id, answer, result)
    print(
        "[CHAT_UI] run_agent_rag done_and_saved "
        f"chat_id={submitted_chat_id} success={result.get('success', True)} "
        f"elapsed_ms={result.get('execution_time_ms', 0)}",
        flush=True,
    )

    assistant_placeholder.markdown(
        _chat_message_html("assistant", answer, result.get("sources", [])),
        unsafe_allow_html=True,
    )
    if result.get("error") and not result.get("success", True):
        st.caption(f"Agent error: {result['error']}")
