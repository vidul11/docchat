"""
RAG chain: retrieve relevant chunks from ChromaDB, then generate an answer.

Flow for every user question:
  1. Embed the question (same model used at ingest time).
  2. Retrieve the top-K most similar chunks from ChromaDB.
  3. Stuff those chunks into a prompt as "context".
  4. Send the prompt to the LLM and return the answer + source docs.

Why LCEL (LangChain Expression Language) instead of RetrievalQA?
RetrievalQA was removed in LangChain 1.x. LCEL uses the | pipe operator to
chain steps explicitly — easier to read and more flexible.
"""

import logging
import re
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from app.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, OPENAI_API_KEY, OPENAI_MODEL, RETRIEVAL_K
from app.ingestor import get_vectorstore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# "Answer only from context" is the single most important instruction in a RAG
# prompt — it stops the LLM from mixing in its own training knowledge and
# hallucinating facts that aren't in your documents.
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


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm() -> BaseLanguageModel:
    """
    Return the configured LLM.

    LLM_PROVIDER=ollama  → local Ollama instance (default, free, private)
    LLM_PROVIDER=openai  → OpenAI API (fallback, requires OPENAI_API_KEY)

    Why a factory function instead of a module-level object?
    Importing this module doesn't start Ollama or validate the API key.
    The LLM is only created when you actually call build_rag_chain().
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.1)

    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1, reasoning=False)


# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_rag_chain() -> dict:
    """
    Assemble and return a RAG chain using LCEL (pipe syntax).

    Returns a dict with:
      "chain"     — the runnable that produces the answer string
      "retriever" — kept separately so query() can get source documents

    Call this once at startup and reuse — building it is expensive
    (opens ChromaDB, loads embedding model) but invoking it is cheap.
    """
    llm = _build_llm()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    # LCEL pipe: retrieve → format → prompt → LLM → parse to string
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _PROMPT
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that qwen3 adds when thinking mode leaks through."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def query(rag: dict, question: str) -> dict:
    """
    Run a question through the RAG chain and return a clean result dict.

    Returns:
        {
            "answer":  str,
            "sources": list[str]   # unique source filenames
        }
    """
    answer = _strip_think(rag["chain"].invoke(question))

    # Fetch sources separately using the same retriever
    docs = rag["retriever"].invoke(question)
    sources = sorted({Path(doc.metadata.get("source", "unknown")).name for doc in docs})

    return {"answer": answer, "sources": sources}
