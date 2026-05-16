# DocChat

A personal knowledge base you can chat with. Upload your PDFs, research papers, or notes — ask questions, get answers grounded in your documents.

This project was built to get hands-on with the tools that power most modern AI applications — LangChain, vector databases, local LLMs, and the MCP protocol. Rather than just reading about them, the goal was to actually wire them together and understand what's happening under the hood.

---

## How it works

You upload a document. DocChat breaks it into chunks, embeds them locally, and stores them in ChromaDB. When you ask a question, it finds the most relevant chunks and sends them to a local LLM to generate an answer — no hallucinations, no outside knowledge leaking in.

**Tested with "Attention Is All You Need" (Vaswani et al., 2017)**

```
You:     What is the main contribution of this paper?

DocChat: The Transformer is the first transduction model relying entirely on
         self-attention without using RNNs or convolution.

Sources: attention_is_all_you_need.pdf
```

---

## Stack

| Tool | What it does here |
|---|---|
| LangChain | Wires the RAG pipeline together |
| ChromaDB | Stores document chunks and their embeddings |
| sentence-transformers | Generates embeddings locally (free, no API) |
| Ollama + qwen3.5:9b | Runs the LLM locally |
| FastAPI | REST API backend |
| Gradio | Chat UI |
| MCP SDK | Exposes the knowledge base as a tool for Claude Desktop |

---

## Setup

**Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv), [Ollama](https://ollama.com)

```bash
# Pull the model
ollama pull qwen3.5:9b

# Clone and install
git clone https://github.com/vidul11/docchat.git
cd docchat
uv sync

# Configure
cp .env.example .env
```

---

## Run

```bash
ollama serve            # terminal 1
uv run python main.py   # terminal 2
```

Open `http://localhost:8000`, upload a document, and start asking questions.

---

## MCP Integration (Claude Desktop)

Once the app is running, add this to your Claude Desktop config:

```json
{
  "mcpServers": {
    "docchat": {
      "url": "http://localhost:8000/mcp/sse"
    }
  }
}
```

Claude will automatically query your knowledge base when you ask about your documents.

---

## REST API

Interactive docs at `http://localhost:8000/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/query` | Ask a question |
| POST | `/ingest` | Upload a document |
| GET | `/sources` | List indexed documents |

---

## Bring your own LLM

DocChat defaults to Ollama for fully local inference. Want to use OpenAI, Claude, or any other LLM? The provider is just a config switch — swap it in `.env` and the rest of the app stays the same:

```env
# Local (default) — any model available in Ollama
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5:9b

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-nano
```

Support for additional providers (Claude, Gemini, etc.) can be added in `app/rag.py` by extending the `_build_llm()` function.
