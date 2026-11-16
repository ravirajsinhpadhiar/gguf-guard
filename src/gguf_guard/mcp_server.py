"""Optional MCP server (pip install 'gguf-guard[mcp]')."""
from __future__ import annotations

from typing import Any

from . import checks, manifest
from .parser import GGUFError, parse

try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

server = _Server("gguf-guard")


@server.tool()
def scan_model(path: str) -> dict[str, Any]:
    """Static safety checks on a GGUF model file: structure, chat-template injection
    constructs, hidden characters, provenance metadata. Does not load or run the model."""
    try:
        g = parse(path)
    except GGUFError as exc:
        return {"error": str(exc)}
    return {
        "file": str(g.path), "gguf_version": g.version, "tensors": len(g.tensors),
        "findings": [f.as_dict() for f in checks.scan(g)],
        "limitations": ["Static checks only: a clean scan does not mean the weights are unmodified or safe."],
    }


@server.tool()
def model_info(path: str) -> dict[str, Any]:
    """Metadata and tensor summary of a GGUF file."""
    try:
        return checks.describe(parse(path))
    except GGUFError as exc:
        return {"error": str(exc)}


@server.tool()
def diff_models(reference: str, candidate: str) -> dict[str, Any]:
    """Which tensors and metadata keys differ between two GGUF files. Hashes every
    tensor, so multi-gigabyte files take a while."""
    try:
        return manifest.compare(manifest.build(parse(reference)), manifest.build(parse(candidate)))
    except GGUFError as exc:
        return {"error": str(exc)}


def run() -> None:
    server.run()
