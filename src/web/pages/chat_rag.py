from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
from urllib.parse import quote, urlencode

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
from core.references import normalize_references


CHAT_STYLE = ""  # Visual styles live in core/styles.css.


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
            href = "?" + urlencode({"detail": detail_key, "return_to": "chat",
                                     "chat": st.session_state.get("chat_active_id", "")})
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
    if role == "assistant":
        content, sources = normalize_references(content, sources)
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
    col_back, col_info = st.columns([1.5, 8.5], vertical_alignment="center")

    with col_back:
        st.write("")
        if st.button("< Home"):
            st.query_params.clear()
            st.session_state.page = "home"
            st.rerun()

    with col_info:
        st.html(
            '<div class="chat-topbar">'
            '<span class="chat-topbar-title">Research Chat</span>'
            f'<span class="chat-topbar-badge">{AGENT_MODEL}</span>'
            "</div>",
        )


def _render_empty_state() -> None:
    st.html(
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
    )


def _render_chat_history() -> None:
    for message in st.session_state.chat_history:
        st.html(
            _chat_message_html(
                str(message.get("role", "assistant")),
                message.get("content", ""),
                message.get("sources", []),
            ),
        )


def _is_chat_pending(session: dict | None) -> bool:
    return bool(session and session.get("pending", False))


def _render_pending_response() -> None:
    st.html(_chat_typing_html(),)


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

    if st.sidebar.button("New chat", width="stretch"):
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
            help=str(session.get("title", "New chat")),
            width="stretch",
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

    st.html(_chat_message_html("user", question),)

    assistant_placeholder = st.empty()
    assistant_placeholder.html(_chat_typing_html())
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

    assistant_placeholder.html(
        _chat_message_html("assistant", answer, result.get("sources", []))
    )
    if result.get("error") and not result.get("success", True):
        st.caption(f"Agent error: {result['error']}")
