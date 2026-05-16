"""Entry point — wires FastAPI, Gradio UI, and the MCP server together on port 8000."""

import logging
from pathlib import Path

import gradio as gr
import uvicorn

from app.api import app
from app.ingestor import ingest_file
from app.mcp_server import create_mcp_app
from app.rag import build_rag_chain, query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_gradio_ui(chain) -> gr.Blocks:
    """Build the Gradio chat interface."""
    def respond(message: str, history: list) -> tuple[str, list]:
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
        if file is None:
            return "No file uploaded."
        try:
            n = ingest_file(file.name)
            return f"Ingested {Path(file.name).name} — {n} chunks added."
        except Exception as e:
            return f"Error: {e}"

    css = """
    body, .gradio-container { background-color: #1e1e1e !important; }
    .gradio-container { max-width: 100% !important; padding: 24px 40px !important; }
    #chatbot { background-color: #2a2a2a !important; border: 1px solid #333 !important; border-radius: 12px !important; }
    #msg_box textarea { background-color: #2a2a2a !important; color: #f0f0f0 !important; border: 1px solid #444 !important; border-radius: 8px !important; }
    #send_btn { background-color: #00bcd4 !important; color: #000 !important; border-radius: 8px !important; font-weight: 600 !important; }
    #send_btn:hover { background-color: #00acc1 !important; }
    #clear_btn { background-color: #2a2a2a !important; color: #00bcd4 !important; border: 1px solid #00bcd4 !important; border-radius: 8px !important; }
    #clear_btn:hover { background-color: #00bcd4 !important; color: #000 !important; }
    #upload_row { background-color: #2a2a2a !important; border: 1px solid #333 !important; border-radius: 10px !important; padding: 12px !important; }
    .message.bot { background-color: #2a2a2a !important; border: 1px solid #333 !important; color: #f0f0f0 !important; }
    .message.user { background-color: transparent !important; color: #f0f0f0 !important; border: 1px solid #00bcd4 !important; }
    h1 { color: #f0f0f0 !important; }
    p { color: #aaa !important; }
    """

    with gr.Blocks(title="DocChat", css=css) as ui:
        gr.Markdown("# DocChat\nAsk questions about your uploaded documents.")

        chatbot = gr.Chatbot(height=600, type="messages", elem_id="chatbot", show_label=False)

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Ask a question...",
                show_label=False,
                elem_id="msg_box",
                scale=7,
                container=False,
            )
            send_btn = gr.Button("Send", elem_id="send_btn", scale=1, min_width=80)
            clear_btn = gr.Button("Clear", elem_id="clear_btn", scale=1, min_width=80)

        with gr.Column(elem_id="upload_row"):
            file_upload = gr.File(
                file_types=[".pdf", ".md", ".txt"],
                label="Upload a document (PDF, MD, TXT)",
            )
            upload_status = gr.Textbox(
                label=None,
                placeholder="Upload status will appear here...",
                interactive=False,
                show_label=False,
                container=False,
            )

        msg_box.submit(respond, [msg_box, chatbot], [msg_box, chatbot])
        send_btn.click(respond, [msg_box, chatbot], [msg_box, chatbot])
        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])
        file_upload.change(ingest_uploaded_file, [file_upload], [upload_status])

    return ui


def main():
    logger.info("Starting DocChat...")

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
