"""All settings loaded from environment variables via .env at the project root."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# LLM
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

# Embeddings
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Storage
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DOCS_DIR: str = os.getenv("DOCS_DIR", "./data/documents")

# Chunking — tweak these to tune retrieval quality
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "5"))
