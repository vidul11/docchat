"""
MCP (Model Context Protocol) server — exposes the knowledge base as a tool
that any MCP-compatible AI assistant (Claude Desktop, etc.) can call.

What is MCP?
MCP is an open protocol by Anthropic (2024) that standardises how AI assistants
discover and call external tools. Think of it like a USB-C standard — instead of
every AI app inventing its own plugin format, they all speak MCP.

How it works here:
1. This server advertises one tool: "query_knowledge_base"
2. Claude Desktop (or any MCP client) connects via SSE (Server-Sent Events)
3. When a user asks Claude about their documents, Claude calls our tool
4. We run the RAG query and return the answer + sources back to Claude

Transport — why SSE?
SSE (Server-Sent Events) is HTTP-based, works through firewalls, and needs
no special infrastructure. The MCP SDK handles the protocol framing; we just
define the tools and their handlers.
"""

import logging

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.applications import Starlette

from app.rag import build_rag_chain, query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

# The server name is what MCP clients display to the user.
# Keep it descriptive — Claude Desktop shows this in its tools list.
server = Server("docchat-knowledge-base")

# Built once and reused — same reasoning as api.py
_chain = None


def get_chain():
    global _chain
    if _chain is None:
        logger.info("Building RAG chain for MCP server...")
        _chain = build_rag_chain()
    return _chain


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    Advertise available tools to the MCP client.

    The client calls this once on connect to discover what tools exist.
    The inputSchema follows JSON Schema — it tells the client exactly what
    arguments to pass when calling the tool.
    """
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
    


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Handle a tool call from the MCP client.

    The client sends the tool name + arguments dict.
    We run the RAG query and return a list of TextContent objects.

    Why return a list?
    MCP tools can return multiple content blocks (text, images, etc.).
    We only need one text block here, but the protocol requires a list.
    """
    if name != "query_knowledge_base":
        raise ValueError(f"Unknown tool: {name}")

    question = arguments["question"]
    result = query(get_chain(), question)
    answer = result["answer"]
    sources_str = ", ".join(result["sources"])
    return [TextContent(type="text", text=f"{answer}\n\nSources: {sources_str}")]


# ---------------------------------------------------------------------------
# Starlette app (mounted into FastAPI in main.py)
# ---------------------------------------------------------------------------

def create_mcp_app() -> Starlette:
    """
    Build a Starlette app that handles the SSE transport for the MCP server.

    Why Starlette and not FastAPI directly?
    The MCP SDK's SseServerTransport works at the ASGI level — it needs raw
    request/response control that FastAPI's router abstracts away. Starlette
    gives us that low-level access while still being mountable into FastAPI.

    Two routes:
      GET  /sse      — client connects here to open the SSE stream
      POST /messages — client sends tool call requests here
    """
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
