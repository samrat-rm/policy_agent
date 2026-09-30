"""Single place for configuration. Everything else imports from here."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# Must be a model with tool-calling support. phi4-mini, qwen2.5, mistral-nemo
# and phi4-mini work. phi3, gemma3 and tinyllama do NOT — they will fail with
# "does not support tools".
LLM_MODEL = os.getenv("LLM_MODEL", "phi4-mini:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHROMA_DIR = str(ROOT / os.getenv("CHROMA_DIR", "chroma_db"))
POLICY_DIR = ROOT / "data" / "policies"
COLLECTION_NAME = "policies"

# Retrieval settings. Tune these and watch your eval score move —
# that experiment is worth writing up in the README.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 4