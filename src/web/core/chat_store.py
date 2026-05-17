from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parents[3]
CHAT_HISTORY_DIR = ROOT_DIR / "data" / "chat_history"
CHAT_HISTORY_FILE = CHAT_HISTORY_DIR / "chat_sessions.json"
_CHAT_STORE_LOCK = RLock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_store() -> dict[str, Any]:
    return {"active_chat_id": None, "sessions": []}


def _normalize_messages(messages: object) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []

    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        item = dict(message)
        item["role"] = str(item.get("role", "assistant")).strip() or "assistant"
        item["content"] = str(item.get("content", ""))
        if item["role"] == "assistant":
            if not isinstance(item.get("sources"), list):
                item["sources"] = []
            if not isinstance(item.get("execution_path"), list):
                item["execution_path"] = []
        normalized.append(item)

    return normalized


def _normalize_session(session: object) -> dict[str, Any] | None:
    if not isinstance(session, dict):
        return None

    chat_id = str(session.get("chat_id") or session.get("id") or "").strip()
    if not chat_id:
        chat_id = uuid4().hex

    created_at = str(session.get("created_at") or session.get("updated_at") or _now_iso())
    updated_at = str(session.get("updated_at") or created_at)
    title = str(session.get("title") or "New chat").strip() or "New chat"

    normalized = {
        "chat_id": chat_id,
        "title": title,
        "created_at": created_at,
        "updated_at": updated_at,
        "messages": _normalize_messages(session.get("messages", [])),
    }
    if bool(session.get("pending", False)):
        normalized["pending"] = True
        normalized["pending_question"] = str(session.get("pending_question") or "")
        normalized["pending_started_at"] = str(session.get("pending_started_at") or updated_at)
    return normalized


def _has_messages(session: dict[str, Any]) -> bool:
    return bool(session.get("messages"))


def _sort_key(session: dict[str, Any]) -> str:
    return str(session.get("updated_at") or session.get("created_at") or "")


def load_chat_store() -> dict[str, Any]:
    with _CHAT_STORE_LOCK:
        if not CHAT_HISTORY_FILE.exists():
            return _default_store()

        try:
            with CHAT_HISTORY_FILE.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            return _default_store()

        sessions = []
        removed_empty_sessions = False
        for raw_session in data.get("sessions", []) if isinstance(data, dict) else []:
            session = _normalize_session(raw_session)
            if session and _has_messages(session):
                sessions.append(session)
            elif session:
                removed_empty_sessions = True

        sessions.sort(key=_sort_key, reverse=True)
        active_chat_id = None
        if isinstance(data, dict):
            active_value = str(data.get("active_chat_id") or "").strip()
            if active_value and any(session["chat_id"] == active_value for session in sessions):
                active_chat_id = active_value

        store = {"active_chat_id": active_chat_id, "sessions": sessions}
        if removed_empty_sessions:
            save_chat_store(store)

        return store


def save_chat_store(store: dict[str, Any]) -> None:
    with _CHAT_STORE_LOCK:
        CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        payload = {"active_chat_id": None, "sessions": []}
        requested_active_id = str(store.get("active_chat_id") or "").strip()

        for raw_session in store.get("sessions", []):
            session = _normalize_session(raw_session)
            if session and _has_messages(session):
                payload["sessions"].append(session)

        payload["sessions"].sort(key=_sort_key, reverse=True)
        if requested_active_id and any(session["chat_id"] == requested_active_id for session in payload["sessions"]):
            payload["active_chat_id"] = requested_active_id

        tmp_path = CHAT_HISTORY_FILE.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(CHAT_HISTORY_FILE)


def update_chat_store(mutator: Callable[[dict[str, Any]], dict[str, Any] | None]) -> dict[str, Any]:
    with _CHAT_STORE_LOCK:
        store = load_chat_store()
        updated_store = mutator(deepcopy(store))
        if updated_store is None:
            updated_store = store
        save_chat_store(updated_store)
        return load_chat_store()


def create_chat_session(title: str = "New chat") -> dict[str, Any]:
    now = _now_iso()
    return {
        "chat_id": uuid4().hex,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def upsert_chat_session(store: dict[str, Any], session: dict[str, Any]) -> dict[str, Any]:
    sessions = list(store.get("sessions", []))
    session = _normalize_session(session) or create_chat_session()
    if not _has_messages(session):
        chat_id = str(session.get("chat_id", "")).strip()
        store["sessions"] = [
            existing
            for existing in sessions
            if str(existing.get("chat_id", "")).strip() != chat_id
        ]
        if str(store.get("active_chat_id") or "").strip() == chat_id:
            store["active_chat_id"] = None
        return store

    replaced = False
    for index, existing in enumerate(sessions):
        if str(existing.get("chat_id", "")).strip() == str(session.get("chat_id", "")).strip():
            sessions[index] = session
            replaced = True
            break

    if not replaced:
        sessions.append(session)

    sessions.sort(key=_sort_key, reverse=True)
    store["sessions"] = sessions
    store["active_chat_id"] = session.get("chat_id")
    return store


def get_chat_session(store: dict[str, Any], chat_id: str) -> dict[str, Any] | None:
    chat_id = str(chat_id or "").strip()
    if not chat_id:
        return None

    for session in store.get("sessions", []):
        if str(session.get("chat_id", "")).strip() == chat_id:
            return session
    return None


def build_chat_title(messages: list[dict[str, Any]], fallback: str = "New chat") -> str:
    for message in messages:
        if str(message.get("role", "")).strip() != "user":
            continue

        text = " ".join(str(message.get("content", "")).split()).strip()
        if not text:
            continue

        if len(text) <= 48:
            return text
        return text[:45].rstrip() + "..."

    return fallback
