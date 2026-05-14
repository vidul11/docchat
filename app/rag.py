"""
RAG chain: retrieve relevant chunks from ChromaDB, then generate an answer.

Flow for every user question:
  1. Embed the question (same model used at ingest time).
  2. Retrieve the top-K most similar chunks from ChromaDB.
  3. Stuff those chunks into a prompt as "context".
  4. Send the prompt to the LLM and return the answer + source docs.

Why "stuff" strategy (not map-reduce or refine)?
- Simple: all chunks go into one prompt in a single LLM call.
- Fast: no extra LLM calls to merge partial answers.
- Fine for our chunk sizes — the total context fits in the model's window.
"""

import logging

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

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

    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

# ---------------------------------------------------------------------------
# Chain builder
# ---------------------------------------------------------------------------

def build_rag_chain() -> RetrievalQA:
    """
    Assemble and return a RetrievalQA chain ready to answer questions.

    Call this once at startup and reuse the chain — building it is expensive
    (opens ChromaDB, loads embedding model) but invoking it is cheap.
    """
    llm = _build_llm()
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": _PROMPT},
        return_source_documents=True,
    )


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

def query(chain: RetrievalQA, question: str) -> dict:
    """
    Run a question through the chain and return a clean result dict.

    Returns:
        {
            "answer":  str,
            "sources": list[str]   # unique source filenames
        }

    Why a wrapper instead of calling chain.invoke() directly?
    The raw chain result has LangChain internals ("source_documents", "result").
    This function gives callers a stable, simple interface — the API and UI
    don't need to know about LangChain's internal structure.
    """
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = {
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    }
    sources = sorted(sources)

    return {"answer": answer, "sources": sources}
