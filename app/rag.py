"""RAG chain: retrieve relevant chunks from ChromaDB, then generate an answer."""

import logging
import re
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL, RETRIEVAL_K
from app.ingestor import get_vectorstore

logger = logging.getLogger(__name__)

_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question using only the context below.
If the context does not contain enough information, say exactly:
"I don't have enough information in my documents to answer this."

Context:
{context}

Question: {question}

Answer (cite the source document when possible):""",
)


def _format_docs(docs) -> str:
    """Concatenate retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def _build_llm() -> BaseLanguageModel:
    """Return the configured LLM (ollama or openai based on LLM_PROVIDER)."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.1)

    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1, reasoning=False)


def build_rag_chain() -> dict:
    """Build and return the RAG chain. Call once at startup and reuse."""
    llm = _build_llm()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that qwen3 adds when thinking mode leaks through."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def query(rag: dict, question: str) -> dict:
    """Run a question through the RAG chain. Returns answer and source filenames."""
    answer = _strip_think(rag["chain"].invoke(question))
    docs = rag["retriever"].invoke(question)
    sources = sorted({Path(doc.metadata.get("source", "unknown")).name for doc in docs})

    return {"answer": answer, "sources": sources}
