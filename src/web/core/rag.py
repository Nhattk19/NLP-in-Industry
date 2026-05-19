from __future__ import annotations

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# Ensure the repository root is importable so `src.agent` resolves when
# Streamlit runs the web app from `src/web/app.py`.
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import get_agent


AGENT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMPTY_ANSWER = "I don't know based on the provided documents."
CHAT_HISTORY_MESSAGE_LIMIT = int(os.getenv("CHAT_HISTORY_MESSAGE_LIMIT", "5"))
CHAT_HISTORY_CHAR_LIMIT = int(os.getenv("CHAT_HISTORY_CHAR_LIMIT", "4500"))


@st.cache_resource(show_spinner=False)
def load_agent():
    """Load the shared LangGraph agent once per Streamlit session."""
    return get_agent()


@st.cache_resource(show_spinner="Preloading agent RAG...")
def preload_agent():
    """Warm the shared agent before the first chat question."""
    return load_agent()


@st.cache_resource(show_spinner=False)
def load_query_rewriter():
    """Load the lightweight LLM used to rewrite follow-up questions."""
    return ChatGoogleGenerativeAI(
        model=AGENT_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )


def _normalize_confidence(value) -> float:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0


def _compact_chat_history(chat_history: list[dict] | None) -> list[dict]:
    """Keep only recent user/assistant messages with bounded content."""
    if not isinstance(chat_history, list):
        return []

    compacted: list[dict] = []
    remaining_chars = CHAT_HISTORY_CHAR_LIMIT

    for message in reversed(chat_history):
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue

        content = " ".join(str(message.get("content", "")).split()).strip()
        if not content:
            continue

        if remaining_chars <= 0:
            break

        if len(content) > remaining_chars:
            content = content[:remaining_chars].rstrip()

        compacted.append({"role": role, "content": content})
        remaining_chars -= len(content)

        if len(compacted) >= CHAT_HISTORY_MESSAGE_LIMIT:
            break

    return list(reversed(compacted))


def _format_history_for_rewrite(chat_history: list[dict]) -> str:
    rows = []
    for message in chat_history:
        role = "User" if message.get("role") == "user" else "Assistant"
        rows.append(f"{role}: {message.get('content', '')}")
    return "\n".join(rows)


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _rewrite_follow_up_question(question: str, chat_history: list[dict]) -> str:
    """Rewrite a conversational follow-up into a standalone retrieval query."""
    question = (question or "").strip()
    if not question or not chat_history:
        return question

    prompt = f"""You rewrite follow-up questions for an academic NLP/ML/AI paper RAG system.

Given the recent chat history and the latest user question, produce one standalone question for retrieval.

Rules:
- Do not answer the question.
- Resolve pronouns and references like it, this, that, they, the method, the paper, its limitations.
- Preserve the user's intent and technical scope.
- If the latest question is already standalone, return it unchanged.
- Keep it concise and search-friendly.

Recent chat history:
{_format_history_for_rewrite(chat_history)}

Latest user question:
{question}

Return ONLY valid JSON:
{{
  "standalone_question": "string"
}}"""

    try:
        response = load_query_rewriter().invoke(prompt)
        payload = _extract_json_object(getattr(response, "content", ""))
        rewritten = str(payload.get("standalone_question", "")).strip()
        if rewritten:
            return rewritten
    except Exception as exc:
        print(f"! [CHAT_RAG] Question rewrite failed: {exc}", flush=True)

    return question


def run_agent_rag(
    question: str,
    chat_history: list[dict] | None = None,
    chat_id: str | None = None,
) -> dict:
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

    compact_history = _compact_chat_history(chat_history)
    standalone_question = _rewrite_follow_up_question(question, compact_history)
    if standalone_question != question:
        print(
            "[CHAT_RAG] rewritten follow-up "
            f"chat_id={chat_id or ''} original={question[:80]!r} standalone={standalone_question[:120]!r}",
            flush=True,
        )

    try:
        result = load_agent().run(
            standalone_question,
            session_id=chat_id or "",
            chat_history=compact_history,
            original_question=question,
        )
    except Exception as exc:
        return {
            "success": False,
            "query": question,
            "standalone_question": standalone_question,
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
            "standalone_question": standalone_question,
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
        "query": question,
        "standalone_question": result.get("standalone_question", standalone_question),
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
