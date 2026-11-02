import streamlit as st


def render_home_page() -> None:
    st.html('<div class="home-intro"><h1 class="title-text">NLP Scholar</h1>'
            '<p class="subtitle-text">Explore Scholarly Entities in Natural Language Processing</p></div>')
    with st.container(key="home_search"):
        search_col, action_col, chat_col = st.columns([6, 1.2, 1.2], vertical_alignment="bottom")
        with search_col:
            query_input = st.text_input("Search", placeholder="Search publications...")
        with action_col:
            search_btn = st.button("Search", type="primary", width="stretch")
        with chat_col:
            chat_btn = st.button("Chat", width="stretch")
    with st.container(key="home_mode"):
        toggle_label = "Semantic Search" if st.session_state.enable_semantic else "Lexical Search"
        is_semantic = st.toggle(toggle_label, value=st.session_state.enable_semantic)
        st.caption("Search by meaning with semantic search, or match keywords with lexical search.")
        if is_semantic != st.session_state.enable_semantic:
            st.session_state.enable_semantic = is_semantic
            st.rerun()
    if search_btn and query_input.strip():
        st.session_state.search_query = query_input.strip()
        st.session_state.page = "results"
        st.rerun()
    if search_btn and not query_input.strip():
        st.info("Enter a keyword or research question to find publications.")
    if chat_btn:
        st.session_state.page = "chat_rag"
        st.rerun()
