"""One shared visual theme for all Streamlit pages."""
from pathlib import Path
import streamlit as st


def apply_page_config() -> None:
    st.set_page_config(page_title="NLP Scholar", page_icon=":material/menu_book:", layout="wide", initial_sidebar_state="auto")


def inject_global_css() -> None:
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8-sig")
    # Hide the unused sidebar only on search/detail pages. Preserve chat's native
    # collapse controls so history remains accessible at narrow viewport widths.
    if st.session_state.get("page", "home") != "chat_rag":
        css += '\n[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none;}'
    st.html(f"<style>{css}</style>")
