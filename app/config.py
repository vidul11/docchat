"""
Centralised config — all settings loaded from environment variables.

Why a dedicated config module instead of os.getenv() everywhere?
- One place to see every tunable setting in the project.
- If you rename an env var, you fix it here, not in 10 files.
- You can set typed defaults (int, str) in one spot.

python-dotenv loads the .env file from disk so you don't have to
export variables in your shell before running the app.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load the .env file sitting one directory above this file (project root).
# Path(__file__) is this file's path; .parent is app/, .parent.parent is the root.
load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")


# ---------------------------------------------------------------------------
# Embeddings — always local, no API cost
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DOCS_DIR: str = os.getenv("DOCS_DIR", "./data/documents")


# ---------------------------------------------------------------------------
# Chunking hyperparameters
# Exposed as config so you can run the sensitivity analysis (300/500/800).
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))
