# DocChat — Personal Knowledge Base with RAG + MCP Server

## One-line pitch
A clean, minimal RAG system over your personal documents (research papers, thesis, course notes) using LangChain and ChromaDB, exposed as both a chat UI and an MCP server — so any MCP-compatible AI assistant can query your knowledge base as a tool.

---

## Why this project exists

This is deliberately the simplest of the four projects. Its purpose is not to impress with novelty — it's to cleanly demonstrate that you know how to build the standard RAG pattern and the emerging MCP protocol. Think of it as the "fundamentals" project that complements the three creative ones.

| What it demonstrates | Why it matters |
|---|---|
| LangChain RAG pipeline | Explicitly listed on many ML engineer job descriptions |
| ChromaDB (or any vector DB) in a standard RAG context | Table-stakes skill for 2026 AI engineering roles |
| MCP server implementation | Cutting-edge protocol (Anthropic, 2024) — very few people have this on their resume yet |
| Clean, well-documented code | Shows you can build production-quality tooling, not just research prototypes |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION                     │
│  Your PDFs, markdown files, text files                  │
│  LangChain document loaders → chunking → embedding      │
│  sentence-transformers (local, free) for embeddings     │
│  Stored in ChromaDB with metadata (source, page, date)  │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│              RAG CHAIN (LangChain)                        │
│  User question → embed → retrieve top-K chunks          │
│  → construct prompt with retrieved context              │
│  → local LLM (qwen3.5:9b via Ollama) generates answer  │
│  → return answer with source citations                  │
└──────────────┬──────────────────────────────────────────┘
               │
       ┌───────┴────────┐
       │                 │
┌──────▼──────┐  ┌───────▼───────┐
│  GRADIO UI  │  │  MCP SERVER   │
│  Chat with  │  │  Expose as    │
│  your docs  │  │  tool for     │
│  in browser │  │  Claude, etc. │
└─────────────┘  └───────────────┘
       │                 │
       └────────┬────────┘
                │
┌───────────────▼─────────────────────────────────────────┐
│              FastAPI BACKEND                              │
│  POST /query — ask a question, get grounded answer      │
│  POST /ingest — add new documents to the knowledge base │
│  GET  /sources — list all ingested documents            │
│  MCP endpoint — SSE-based tool server for external LLMs │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Tool | Why it's here | Role |
|---|---|---|
| **LangChain** | RAG pipeline orchestration — this is its strongest use case | Document loading, chunking, retrieval chain, prompt templates |
| **ChromaDB** | Vector storage for document chunks | Persistent local vector DB |
| **sentence-transformers** | Local embedding model (all-MiniLM-L6-v2 or similar) | Free, no API needed |
| **Local LLM (Ollama)** | Answer generation from retrieved context | qwen3.5:9b |
| **FastAPI** | Backend serving both the chat UI and MCP server | REST + SSE endpoints |
| **Gradio** | Simple chat interface | Upload docs + ask questions |
| **MCP SDK** | Model Context Protocol server implementation | Expose knowledge base as a tool |

**What's NOT included:**
- No OpenAI/Claude API by default — everything runs locally
- No complex agent loop — this is a straightforward retrieval-then-generate pipeline
- No custom model training — using off-the-shelf embeddings and LLM
- No Kubernetes/cloud deployment — runs on your laptop

**Fallback option:**
- `LLM_PROVIDER=ollama` (default) — fully local, no API cost
- `LLM_PROVIDER=openai` — uses gpt-4.1-nano as fallback if Ollama unavailable
- Switchable via environment variable, no code changes needed

---

## Implementation Details

### Document Ingestion Pipeline

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load documents
def load_documents(doc_dir: str):
    """Load all supported documents from a directory."""
    docs = []
    for file in Path(doc_dir).rglob("*"):
        if file.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load())
        elif file.suffix == ".md":
            docs.extend(UnstructuredMarkdownLoader(str(file)).load())
        elif file.suffix == ".txt":
            docs.extend(TextLoader(str(file)).load())
    return docs

# Chunk documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)
```

### RAG Chain

```python
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = Ollama(model="qwen3.5:9b", temperature=0.1)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question based only on the following context. 
If the context doesn't contain enough information, say "I don't have enough 
information in my documents to answer this."

Context:
{context}

Question: {question}

Answer (cite the source document when possible):"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)
```

### MCP Server

This is the cutting-edge piece. MCP (Model Context Protocol) lets external LLM clients (like Claude Desktop) connect to your knowledge base as a tool.

```python
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent

server = Server("docchat-knowledge-base")

@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="query_knowledge_base",
            description="Search and answer questions from the user's personal "
                       "document collection (research papers, thesis, notes). "
                       "Use this when the user asks about their own research, "
                       "course content, or personal documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer from the knowledge base"
                    }
                },
                "required": ["question"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "query_knowledge_base":
        result = qa_chain.invoke({"query": arguments["question"]})
        answer = result["result"]
        sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        return [TextContent(
            type="text",
            text=f"{answer}\n\nSources: {', '.join(set(sources))}"
        )]

# Run as SSE server on FastAPI
transport = SseServerTransport("/mcp")
app = FastAPI()

@app.route("/mcp", methods=["GET"])
async def handle_sse(request):
    async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
```

### FastAPI Endpoints

```python
@app.post("/query")
async def query(request: QueryRequest):
    """Ask a question against the knowledge base."""
    result = qa_chain.invoke({"query": request.question})
    return {
        "answer": result["result"],
        "sources": [
            {"content": doc.page_content[:200], "source": doc.metadata.get("source")}
            for doc in result["source_documents"]
        ]
    }

@app.post("/ingest")
async def ingest(file: UploadFile):
    """Add a new document to the knowledge base."""
    # Save file, load, chunk, embed, add to ChromaDB
    ...
    return {"status": "ingested", "chunks": num_chunks}

@app.get("/sources")
async def sources():
    """List all documents in the knowledge base."""
    ...
```

---

## What Makes the MCP Part Valuable

Once your MCP server is running, you can add it to Claude Desktop's config:

```json
{
  "mcpServers": {
    "my-research": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

Now when you chat with Claude Desktop, it can automatically query your personal knowledge base as a tool. You could ask Claude: "Based on my thesis research, what were the key findings about high impedance fault detection?" and Claude would call your MCP server, retrieve relevant chunks from your actual thesis PDF, and answer grounded in your documents.

This is a genuinely useful daily tool AND a cutting-edge protocol that very few people have implemented yet.

---

## Evaluation

### Retrieval Quality
- **Retrieval precision@5**: for 20 test questions you write, are the retrieved chunks actually relevant?
- **Answer grounding**: does the answer actually use information from the retrieved chunks (vs. hallucinating)?
- **Source attribution accuracy**: are the cited sources correct?

### Chunking Sensitivity Analysis
- Test 3 chunk sizes (300, 500, 800 tokens) and measure retrieval quality
- Document findings in README — shows you think about hyperparameters even in "simple" projects

### MCP Integration Test
- Demonstrate Claude Desktop successfully calling your MCP server
- Screenshot/video of the tool-use interaction for README

---

## Implementation Timeline (3-5 days)

### Day 1-2: Core RAG Pipeline
- [ ] Set up project structure
- [ ] Document ingestion pipeline (PDF, MD, TXT)
- [ ] ChromaDB setup with sentence-transformer embeddings
- [ ] LangChain RAG chain with Ollama (qwen3.5:9b)
- [ ] Test with 5-10 of your own documents
- **Deliverable**: working RAG that answers questions about your documents

### Day 3: FastAPI + Gradio
- [ ] FastAPI backend with /query, /ingest, /sources endpoints
- [ ] Gradio chat UI with document upload
- [ ] Source citation display in the UI
- **Deliverable**: interactive chat interface

### Day 4-5: MCP Server + Polish
- [ ] Implement MCP server using the MCP SDK
- [ ] Test with Claude Desktop integration
- [ ] Chunking sensitivity analysis
- [ ] Write comprehensive README with setup instructions
- [ ] Record demo (chat UI + Claude Desktop MCP integration)
- [ ] Push to GitHub
- **Deliverable**: portfolio-ready project with MCP demo

---

## Resume Bullet Points (draft)

**DocChat — Personal Knowledge Base with RAG + MCP Server**
- Built a RAG pipeline using LangChain and ChromaDB over personal research documents with local embeddings (sentence-transformers) and local LLM inference (Ollama), achieving grounded answers with source citations across X documents.
- Implemented a Model Context Protocol (MCP) server enabling external AI assistants to query the knowledge base as a tool, demonstrating production-ready LLM tool-use integration via FastAPI with SSE transport.

---

## New Tech for Your Resume from This Project

| Technology | How it's used | Resume signal |
|---|---|---|
| LangChain | RAG pipeline orchestration | Standard framework knowledge |
| MCP (Model Context Protocol) | Expose knowledge base as a tool for external LLMs | Cutting-edge protocol, very few have this |
| ChromaDB (in RAG context) | Vector storage for document chunks | Standard RAG pattern |
| FastAPI + SSE | Serve both REST API and MCP transport | Production serving |

---

## How the Four Projects Tell a Complete Story

| Project | What it proves | Key tech |
|---|---|---|
| **PitWall AI** | "I train custom models and build rigorous evaluation pipelines" | PyTorch, Monte Carlo, FastAPI, W&B, DuckDB |
| **SketchML** | "I fine-tune VLMs, build vector retrieval, and generate code" | QLoRA, ChromaDB, Ollama, OpenCV, FastAPI |
| **AniScout** | "I build agentic systems with multi-source data fusion and learned embeddings" | Agentic loop, contrastive learning, Polars, ChromaDB, Ollama |
| **DocChat** | "I know the standard tools and the cutting-edge protocols" | LangChain, RAG, MCP, ChromaDB |

No two projects overlap in their core ML technique. Every major resume gap is covered. The projects range from research-grade (PitWall AI) to creative (SketchML, AniScout) to practical (DocChat). A recruiter sees range. A technical interviewer sees depth.
