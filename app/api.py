"""
FastAPI backend — three endpoints that expose the RAG pipeline over HTTP.

Why FastAPI?
- Automatic request/response validation via Pydantic models.
- Auto-generated docs at /docs (useful for testing without a UI).
- Async-native, so file uploads and LLM calls don't block each other.

Endpoint summary:
  POST /query   — ask a question, get a grounded answer + sources
  POST /ingest  — upload a new document into the knowledge base
  GET  /sources — list all documents currently in ChromaDB
"""

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingestor import ingest_file, list_sources
from app.rag import build_rag_chain, query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App + startup
# ---------------------------------------------------------------------------

app = FastAPI(title="DocChat", description="RAG over your personal documents")

# The chain is built once at startup and reused for every request.
# Why not build it inside each request handler?
# Loading ChromaDB + the embedding model takes ~2-3 seconds.
# You'd pay that cost on every single query — unacceptable for a chat app.
_chain = None


@app.on_event("startup")
async def startup():
    global _chain
    logger.info("Building RAG chain...")
    _chain = build_rag_chain()
    logger.info("RAG chain ready.")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

# Pydantic models define the shape of JSON that comes in and goes out.
# FastAPI validates them automatically — if the client sends the wrong type,
# it returns a 422 error before your code even runs.

class QueryRequest(BaseModel):
    question: str


class SourceSnippet(BaseModel):
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Ask a question against the knowledge base."""
    if _chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not ready")

    return query(_chain, request.question)


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile):
    """
    Upload a document (PDF, MD, TXT) and add it to the knowledge base.

    Why save to a temp file first?
    LangChain's loaders (PyPDFLoader etc.) need a real file path on disk —
    they can't read from an in-memory stream. So we write the upload to a
    temp file, ingest it, then delete it.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".txt"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    num_chunks = ingest_file(tmp_path)
    Path(tmp_path).unlink()

    return {"status": "ingested", "filename": file.filename, "chunks": num_chunks}


@app.get("/sources")
async def sources_endpoint():
    """List all documents currently stored in the knowledge base."""
    return {"sources": list_sources()}

