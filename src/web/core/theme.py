import streamlit as st


def apply_page_config() -> None:
    st.set_page_config(
        page_title="NLP-KG Search",
        page_icon="Search",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"],
        section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"],
        section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"],
        section[data-testid="stSidebar"] nav[aria-label="Sidebar navigation"],
        section[data-testid="stSidebar"] [aria-label="Sidebar navigation"],
        section[data-testid="stSidebar"] nav,
        section[data-testid="stSidebar"] ul,
        section[data-testid="stSidebar"] hr {
            display: none !important;
        }
        section[data-testid="stSidebar"] > div:first-child {
            display: none !important;
        }
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        .title-text {
            text-align: center;
            color: #0d47a1;
            font-size: 4.5rem;
            font-weight: 700;
            margin-bottom: 0;
            margin-top: 5vh;
        }
        .subtitle-text {
            text-align: center;
            color: #757575;
            font-size: 1.2rem;
            margin-bottom: 2rem;
            font-weight: 500;
        }
        div.stButton > button:first-child {
            background-color: #0d47a1;
            color: white;
            border-radius: 8px;
            font-weight: bold;
            transition: 0.2s;
        }
        div.stButton > button:first-child:hover {
            background-color: #0b3c8a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
