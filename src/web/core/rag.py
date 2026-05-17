from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import streamlit as st

# Ensure the repository root is importable so `src.agent` resolves when
# Streamlit runs the web app from `src/web/app.py`.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import get_agent


AGENT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMPTY_ANSWER = "I don't know based on the provided documents."


@st.cache_resource(show_spinner=False)
def load_agent():
    """Load the shared LangGraph agent once per Streamlit session."""
    return get_agent()


@st.cache_resource(show_spinner="Preloading agent RAG...")
def preload_agent():
    """Warm the shared agent before the first chat question."""
    return load_agent()


def _normalize_confidence(value) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def run_agent_rag(question: str) -> dict:
    """Run the web chat through the shared agent pipeline."""
    question = (question or "").strip()
    if not question:
        return {
            "success": False,
            "query": "",
            "intent": "unclear",
            "answer": EMPTY_ANSWER,
            "sources": [],
            "confidence": 0.0,
            "search_mode_used": "hybrid",
            "execution_time_ms": 0,
            "external_search_triggered": False,
            "used_external_papers": False,
            "feedback_info": None,
            "execution_path": [],
            "error": "Empty question",
        }

    try:
        result = load_agent().run(question)
    except Exception as exc:
        return {
            "success": False,
            "query": question,
            "intent": "unclear",
            "answer": f"API error: {exc}",
            "sources": [],
            "confidence": 0.0,
            "search_mode_used": "hybrid",
            "execution_time_ms": 0,
            "external_search_triggered": False,
            "used_external_papers": False,
            "feedback_info": None,
            "execution_path": [],
            "error": str(exc),
        }

    if not isinstance(result, dict):
        return {
            "success": False,
            "query": question,
            "intent": "unclear",
            "answer": EMPTY_ANSWER,
            "sources": [],
            "confidence": 0.0,
            "search_mode_used": "hybrid",
            "execution_time_ms": 0,
            "external_search_triggered": False,
            "used_external_papers": False,
            "feedback_info": None,
            "execution_path": [],
            "error": "Unexpected agent output",
        }

    answer = str(result.get("answer") or "").strip() or EMPTY_ANSWER
    sources = result.get("sources") or []

    return {
        "success": bool(result.get("success", True)),
        "query": result.get("query", question),
        "intent": result.get("intent", "unclear"),
        "answer": answer,
        "sources": sources,
        "confidence": _normalize_confidence(result.get("confidence", 0.0)),
        "search_mode_used": result.get("search_mode_used", "hybrid"),
        "execution_time_ms": int(result.get("execution_time_ms", 0) or 0),
        "external_search_triggered": bool(result.get("external_search_triggered", False)),
        "used_external_papers": bool(result.get("used_external_papers", False)),
        "feedback_info": result.get("feedback_info"),
        "execution_path": result.get("execution_path", []),
        "raw": result,
        "error": result.get("error", ""),
    }
