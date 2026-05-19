import streamlit as st


def render_home_page() -> None:
    st.write("<br><br>", unsafe_allow_html=True)
    st.markdown('<p class="title-text">NLP Scholar</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle-text">Explore Scholarly Entities in Natural Language Processing</p>',
        unsafe_allow_html=True,
    )

    _, search_col, action_col, chat_col, _ = st.columns([1.5, 4, 1, 1, 1.5])
    with search_col:
        query_input = st.text_input(
            "Search",
            label_visibility="collapsed",
            placeholder="Search publications...",
        )
    with action_col:
        search_btn = st.button("Search", use_container_width=True)
    with chat_col:
        chat_btn = st.button("Chat", use_container_width=True)

    _, toggle_col, _ = st.columns([3, 4, 3])
    with toggle_col:
        toggle_label = "Semantic Search" if st.session_state.enable_semantic else "Lexical Search"
        is_semantic = st.toggle(toggle_label, value=st.session_state.enable_semantic)
        if is_semantic != st.session_state.enable_semantic:
            st.session_state.enable_semantic = is_semantic
            st.rerun()

    if search_btn and query_input:
        st.session_state.search_query = query_input
        st.session_state.page = "results"
        st.rerun()

    if chat_btn:
        st.session_state.page = "chat_rag"   
        st.rerun()
