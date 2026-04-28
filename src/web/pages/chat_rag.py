import streamlit as st

from core.rag import (
    OPENAI_MODEL,
    generate_answer,
    init_rag_resources,
    rerank_chunks,
    retrieve_chunks,
)


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
.sources-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.source-chip {
    background: #f5f8fd;
    color: #4f5f7a;
    border: 1px solid #e2e9f5;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 12px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 240px;
}
.source-chip-sim {
    background: #eef5ff;
    color: #1a73e8;
    border-color: #d0defb;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 4px 10px;
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


def _init_chat_session() -> None:
    st.session_state.setdefault("chat_history", [])


def _sources_html(metas, dists, rerank_scores=None) -> str:
    chips = ""
    rerank_scores = rerank_scores or []

    for index, (meta, dist) in enumerate(zip(metas, dists), start=1):
        title = (meta.get("title") or "Untitled")[:38]
        similarity = round(max(0.0, 1.0 - float(dist)), 3)
        chips += (
            f'<span class="source-chip" title="{meta.get("source_url", "")}">'
            f"#{index} {title}...</span>"
            f'<span class="source-chip-sim">sim {similarity}</span>'
        )
        if index - 1 < len(rerank_scores):
            chips += (
                '<span class="source-chip-sim">'
                f"rerank {rerank_scores[index - 1]:.3f}"
                "</span>"
            )

    return (
        '<div class="sources-block">'
        '<div class="sources-label">Sources retrieved</div>'
        f'<div class="sources-chips">{chips}</div>'
        "</div>"
    )


def _render_topbar() -> None:
    col_back, col_info, col_clear = st.columns([1, 8, 1])

    with col_back:
        st.write("")
        if st.button("< Home"):
            st.session_state.page = "home"
            st.rerun()

    with col_info:
        st.markdown(
            '<div class="chat-topbar">'
            '<span class="chat-topbar-title">Research Chat</span>'
            f'<span class="chat-topbar-badge">RAG / {OPENAI_MODEL}</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    with col_clear:
        st.write("")
        if st.session_state.chat_history and st.button("Clear"):
            st.session_state.chat_history = []
            st.rerun()


def _render_empty_state() -> None:
    st.markdown(
        """
        <div class="chat-empty">
            <div class="chat-empty-icon">?</div>
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
        metas = message.get("sources", [])
        dists = message.get("dists", [])
        rerank_scores = message.get("rerank_scores", [])

        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and metas:
                st.markdown(
                    _sources_html(metas, dists, rerank_scores),
                    unsafe_allow_html=True,
                )


def render_chat_page() -> None:
    st.markdown(CHAT_STYLE, unsafe_allow_html=True)
    _init_chat_session()

    api_key, collection, chroma_err, embedder, reranker = init_rag_resources()
    _render_topbar()

    if chroma_err:
        st.error(chroma_err)
        return

    if not api_key:
        st.warning("Could not find `api_agent.txt`. Add your API key file and try again.")
        return

    if not st.session_state.chat_history:
        _render_empty_state()

    _render_chat_history()

    user_input = st.chat_input("Ask a question about the research paper ...")
    if not user_input or not user_input.strip():
        return

    question = user_input.strip()
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            docs, metas, dists = retrieve_chunks(collection, embedder, question)

        with st.spinner("Reranking retrieved chunks..."):
            docs, metas, dists, rerank_scores = rerank_chunks(
                question,
                docs,
                metas,
                dists,
                reranker,
            )

        with st.spinner("Generating answer..."):
            answer, metas, dists, rerank_scores = generate_answer(
                api_key,
                question,
                docs,
                metas,
                dists,
                rerank_scores,
            )

        st.markdown(answer)
        if metas:
            st.markdown(
                _sources_html(metas, dists, rerank_scores),
                unsafe_allow_html=True,
            )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": metas,
            "dists": dists,
            "rerank_scores": rerank_scores,
        }
    )
