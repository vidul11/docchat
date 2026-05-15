"""
Document ingestion pipeline: files → chunks → embeddings → ChromaDB.

Pipeline summary
----------------
1. Load raw files into LangChain `Document` objects (each has .page_content + .metadata).
2. Split those documents into fixed-size, overlapping chunks.
3. Embed every chunk with a local sentence-transformer model (no API needed).
4. Persist the vectors + text in ChromaDB so retrieval survives restarts.

Why these choices?
------------------
- RecursiveCharacterTextSplitter: tries natural boundaries first (\n\n, \n, ". ")
  before resorting to hard character cuts, so chunks stay semantically coherent.
- Overlap (50 chars by default): a key sentence that sits near a chunk boundary
  appears in *both* adjacent chunks, preventing retrieval gaps.
- all-MiniLM-L6-v2: tiny (80 MB), fast, and surprisingly good for semantic search.
  It was trained specifically for sentence-level similarity — exactly what RAG needs.
- cosine similarity in Chroma: normalises for vector magnitude so short and long
  chunks are compared fairly.
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported file types → loader class
# ---------------------------------------------------------------------------
_LOADERS = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
}


def load_documents(doc_dir: str = DOCS_DIR) -> list[Document]:
    """
    Walk `doc_dir` recursively and load every supported file.

    Returns a flat list of LangChain Document objects.
    Each Document has:
      - .page_content: the raw text of that page/file
      - .metadata:     dict with at least {"source": "<filepath>"}
                       PyPDFLoader also adds {"page": <int>}
    """
    docs: list[Document] = []
    doc_path = Path(doc_dir)

    if not doc_path.exists():
        logger.warning("Documents directory %s does not exist.", doc_dir)
        return docs

    for file in sorted(doc_path.rglob("*")):
        if file.is_dir():
            continue
        loader_cls = _LOADERS.get(file.suffix.lower())
        if loader_cls is None:
            logger.debug("Skipping unsupported file: %s", file.name)
            continue
        try:
            loaded = loader_cls(str(file)).load()
            logger.info("Loaded %d page(s) from %s", len(loaded), file.name)
            docs.extend(loaded)
        except Exception:
            logger.exception("Failed to load %s", file)

    logger.info("Total pages/sections loaded: %d", len(docs))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Split raw documents into overlapping chunks.

    Why RecursiveCharacterTextSplitter?
    It tries separators in order: paragraph break → line break → sentence end → space.
    This keeps chunks semantically intact much better than a naive character split.

    chunk_size  = max characters per chunk  (configurable via CHUNK_SIZE env var)
    chunk_overlap = characters shared with the next chunk (prevents boundary gaps)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        "Split %d document pages into %d chunks (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Build the embedding model (downloaded once, cached locally by HuggingFace).

    all-MiniLM-L6-v2 produces 384-dimensional vectors.
    It's fast enough to embed thousands of chunks in seconds on CPU.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    """
    Open (or create) the persisted ChromaDB collection.

    ChromaDB stores both the raw text and its vector representation on disk.
    `collection_metadata={"hnsw:space": "cosine"}` tells Chroma to use cosine
    similarity rather than Euclidean distance — better for text embeddings where
    vector magnitude doesn't carry meaning.

    If the collection already has documents, this just opens it (no re-embedding).
    """
    if embeddings is None:
        embeddings = get_embeddings()

    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingest_documents(doc_dir: str = DOCS_DIR) -> int:
    """
    Full ingestion pipeline: load → split → embed → store.

    Returns the number of chunks added to ChromaDB.
    """
    docs = load_documents(doc_dir)
    if not docs:
        logger.warning("No documents found in %s — nothing ingested.", doc_dir)
        return 0

    chunks = split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    # add_documents() embeds each chunk and upserts into ChromaDB.
    # It uses the chunk text + metadata; IDs are auto-generated.
    vectorstore.add_documents(chunks)
    logger.info("Ingested %d chunks into ChromaDB at %s", len(chunks), CHROMA_PERSIST_DIR)
    return len(chunks)


def ingest_file(file_path: str) -> int:
    """
    Ingest a single file (used by the FastAPI /ingest endpoint after upload).
    """
    loader_cls = _LOADERS.get(Path(file_path).suffix.lower())
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")

    docs = loader_cls(file_path).load()
    chunks = split_documents(docs)

    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    vectorstore.add_documents(chunks)

    logger.info("Ingested %d chunks from %s", len(chunks), file_path)
    return len(chunks)


def list_sources() -> list[str]:
    """
    Return the unique source filenames stored in ChromaDB metadata.
    Used by the GET /sources endpoint.
    """
    vectorstore = get_vectorstore()
    # Chroma's .get() returns a dict with "metadatas" — a list of dicts.
    result = vectorstore.get()
    sources = {
        meta.get("source", "unknown")
        for meta in result.get("metadatas", [])
    }
    return sorted(sources)
