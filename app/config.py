"""Single place for configuration. Everything else imports from here."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# --- Providers -------------------------------------------------------------
# "openai" or "ollama". These are independent: OpenAI for chat plus Ollama for
# embeddings is a good combination, since embeddings run once and chat runs
# constantly.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

_DEFAULT_LLM = {"openai": "gpt-4o-mini", "ollama": "llama3.1"}
_DEFAULT_EMB = {"openai": "text-embedding-3-small", "ollama": "nomic-embed-text"}

LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_LLM.get(LLM_PROVIDER, "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or _DEFAULT_EMB.get(
    EMBEDDING_PROVIDER, "nomic-embed-text"
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Storage ---------------------------------------------------------------
CHROMA_DIR = str(ROOT / os.getenv("CHROMA_DIR", "chroma_db"))
POLICY_DIR = ROOT / "data" / "policies"
COLLECTION_NAME = "policies"

# --- Retrieval -------------------------------------------------------------
# Tune these and watch the eval score move. That experiment is worth writing
# up in the README.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 4

# --- Guards ----------------------------------------------------------------
if LLM_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
        "Add it to .env, or set LLM_PROVIDER=ollama to run locally."
    )
if EMBEDDING_PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
    )