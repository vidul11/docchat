"""FastAPI backend — /query, /ingest, /sources endpoints over the RAG pipeline."""

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from app.ingestor import ingest_file, list_sources
from app.rag import build_rag_chain, query

logger = logging.getLogger(__name__)

app = FastAPI(title="DocChat", description="RAG over your personal documents")

_chain = None


@app.on_event("startup")
async def startup():
    global _chain
    logger.info("Building RAG chain...")
    _chain = build_rag_chain()
    logger.info("RAG chain ready.")


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Ask a question against the knowledge base."""
    if _chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not ready")

    return query(_chain, request.question)


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile):
    """Upload a document (PDF, MD, TXT) and add it to the knowledge base."""
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

