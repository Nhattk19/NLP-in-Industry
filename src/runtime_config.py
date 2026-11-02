"""Load project configuration and resolve API keys stored in text files."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_runtime_config() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    value = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not value:
        return
    if not (value.lower().endswith(".txt") or value.startswith(("./", "../", ".\\"))):
        return
    key_path = Path(value).expanduser()
    if not key_path.is_absolute():
        key_path = PROJECT_ROOT / key_path
    if not key_path.is_file():
        raise ValueError("GOOGLE_API_KEY points to a key file that does not exist.")
    key = key_path.read_text(encoding="utf-8-sig").strip()
    if not key:
        raise ValueError("The GOOGLE_API_KEY file is empty.")
    os.environ["GOOGLE_API_KEY"] = key
