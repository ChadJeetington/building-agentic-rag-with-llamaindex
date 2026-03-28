import os
from pathlib import Path

from dotenv import load_dotenv

# Load repo-root .env regardless of notebook cwd (e.g. lesson subfolders).
_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")


def get_anthropic_api_key() -> str:
    """Return the Anthropic API key from the environment (including values from `.env`)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to `.env` at the project root or export it in your environment."
        )
    return key
