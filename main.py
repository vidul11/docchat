"""
Entry point — wires FastAPI, Gradio UI, and the MCP server together.

Why one entry point instead of running them separately?
- Single command to start the whole system: `uv run python main.py`
- Gradio mounts into FastAPI so both share the same port (8000)
- MCP server also mounts into FastAPI at /mcp
- One process, one port, everything works together

Port layout (all on localhost:8000):
  /          → Gradio chat UI
  /query     → FastAPI REST endpoint
  /ingest    → FastAPI REST endpoint
  /sources   → FastAPI REST endpoint
  /docs      → FastAPI auto-generated API docs (free, always available)
  /mcp/sse   → MCP SSE connection endpoint
"""

import logging

import gradio as gr
import uvicorn

from app.api import app
from app.ingestor import ingest_file
from app.mcp_server import create_mcp_app
from app.rag import build_rag_chain, query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_gradio_ui(chain) -> gr.Blocks:
    """
    Build the Gradio chat interface.

    gr.Blocks lets us compose multiple components (chatbot, file upload,
    sources display) into one page — more flexible than gr.ChatInterface.

    Why pass `chain` as an argument instead of importing it?
    The chain is already built by the time this function is called.
    Passing it in makes the dependency explicit and avoids circular imports.
    """
    def respond(message: str, history: list) -> tuple[str, list]:
        """Called every time the user sends a message."""
        if not message.strip():
            return "", history
        try:
            result = query(chain, message)
            sources = result["sources"]
            if sources:
                response = result["answer"] + "\n\n**Sources:** " + ", ".join(sources)
            else:
                response = result["answer"]
        except Exception as e:
            logger.exception("Query failed")
            response = f"Something went wrong while answering your question. Please try again.\n\n_(Error: {e})_"

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return "", history

    def ingest_uploaded_file(file) -> str:
        """Called when the user uploads a document via the UI."""
        if file is None:
            return "No file uploaded."
        try:
            n = ingest_file(file.name)
            return f"Ingested {file.name} — {n} chunks added."
        except Exception as e:
            return f"Error: {e}"

    with gr.Blocks(title="DocChat") as ui:
        gr.Markdown("# DocChat\nAsk me anything from what you provided.")

        chatbot = gr.Chatbot(height=450, type="messages")
        msg_box = gr.Textbox(placeholder="You know what to do...", show_label=False)
        clear_btn = gr.Button("Clear")

        gr.Markdown("---\n### Add a document")
        file_upload = gr.File(file_types=[".pdf", ".md", ".txt"], label="Upload document")
        upload_status = gr.Textbox(label="Upload status", interactive=False)

        msg_box.submit(respond, [msg_box, chatbot], [msg_box, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])
        file_upload.change(ingest_uploaded_file, [file_upload], [upload_status])

    return ui


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting DocChat...")

    # Build the RAG chain once — shared by both the API and the Gradio UI.
    # (The MCP server builds its own copy lazily on first request.)
    chain = build_rag_chain()

    app.mount("/mcp", create_mcp_app())

    ui = build_gradio_ui(chain)
    gr.mount_gradio_app(app, ui, path="/")

    logger.info("DocChat running at http://localhost:8000")
    logger.info("API docs at http://localhost:8000/docs")
    logger.info("MCP server at http://localhost:8000/mcp/sse")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
