"""MCP server — exposes the knowledge base as a tool over SSE transport."""

import logging

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.applications import Starlette

from app.rag import build_rag_chain, query

logger = logging.getLogger(__name__)

server = Server("docchat-knowledge-base")

_chain = None


def get_chain():
    global _chain
    if _chain is None:
        logger.info("Building RAG chain for MCP server...")
        _chain = build_rag_chain()
    return _chain


@server.list_tools()
async def list_tools() -> list[Tool]:
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
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "query_knowledge_base":
        raise ValueError(f"Unknown tool: {name}")

    question = arguments["question"]
    result = query(get_chain(), question)
    answer = result["answer"]
    sources_str = ", ".join(result["sources"])
    return [TextContent(type="text", text=f"{answer}\n\nSources: {sources_str}")]


def create_mcp_app() -> Starlette:
    """Build the Starlette app that handles SSE transport for the MCP server."""
    transport = SseServerTransport("/mcp/messages")

    async def handle_sse(request: Request):
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    async def handle_messages(request: Request):
        await transport.handle_post_message(
            request.scope, request.receive, request._send
        )

    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=transport.handle_post_message),
        ]
    )
