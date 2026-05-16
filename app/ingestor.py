"""Document ingestion pipeline: files → chunks → embeddings → ChromaDB."""

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

_LOADERS = {
    ".pdf": PyPDFLoader,
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
}


def load_documents(doc_dir: str = DOCS_DIR) -> list[Document]:
    """Walk doc_dir recursively and load every supported file."""
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
    """Split documents into overlapping chunks for embedding."""
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
    """Load the embedding model (cached locally after first download)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_vectorstore(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    """Open (or create) the persisted ChromaDB collection."""
    if embeddings is None:
        embeddings = get_embeddings()

    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def ingest_documents(doc_dir: str = DOCS_DIR) -> int:
    """Full ingestion pipeline: load → split → embed → store. Returns chunk count."""
    docs = load_documents(doc_dir)
    if not docs:
        logger.warning("No documents found in %s — nothing ingested.", doc_dir)
        return 0

    chunks = split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)

    vectorstore.add_documents(chunks)
    logger.info("Ingested %d chunks into ChromaDB at %s", len(chunks), CHROMA_PERSIST_DIR)
    return len(chunks)


def ingest_file(file_path: str) -> int:
    """Ingest a single file into ChromaDB. Returns chunk count."""
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
    """Return unique source filenames stored in ChromaDB metadata."""
    vectorstore = get_vectorstore()
    result = vectorstore.get()
    sources = {
        meta.get("source", "unknown")
        for meta in result.get("metadatas", [])
    }
    return sorted(sources)
