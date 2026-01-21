"""Simple agent wrapper to use MCP server tools."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MCPToolsAgent:
    """
    Wrapper for agents to call MCP tools directly.
    This is the simplest approach - no separate process needed.
    """

    @staticmethod
    def python_check(filepath: str, timeout: int = 30) -> dict[str, Any]:
        """
        Check a Python file for syntax/runtime errors.
        
        Args:
            filepath: Path to the Python file to check
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with check results
        """
        from ai_pedia_mcp_server.mcp_tools.python_checker import python_check
        
        try:
            result = python_check(filepath, timeout)
            return result
        except Exception as exc:
            logger.error(f"python_check failed: {exc}")
            return {
                "success": False,
                "path": filepath,
                "compile": {"success": False, "error": str(exc)},
                "run": None,
                "error_log": None,
            }

    @staticmethod
    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    @staticmethod
    def greeting(name: str) -> str:
        """Return a greeting message."""
        return f"Hello, {name}!"

    @staticmethod
    def rag_query(query: str, n_results: int = 5) -> str:
        """
        Search the local knowledge base.
        """
        from ai_pedia_mcp_server.mcp_tools.rag_search import rag_query
        try:
            return rag_query(query, n_results)
        except Exception as exc:
            logger.error(f"rag_query failed: {exc}")
            return f"Error: {exc}"


# For backward compatibility with async code
class MCPClient:
    """Async wrapper for MCPToolsAgent."""

    async def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Call a tool asynchronously."""
        agent = MCPToolsAgent()
        
        if tool_name == "python_check":
            return agent.python_check(**kwargs)
        elif tool_name == "add":
            return agent.add(**kwargs)
        elif tool_name == "greeting":
            return agent.greeting(**kwargs)
        elif tool_name == "rag_query":
            return agent.rag_query(**kwargs)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


class MCPClientSync:
    """Synchronous wrapper for calling MCP tools directly."""

    def __enter__(self) -> MCPClientSync:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def call_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Tool-specific arguments
            
        Returns:
            Tool result
        """
        agent = MCPToolsAgent()
        
        if tool_name == "python_check":
            return agent.python_check(**kwargs)
        elif tool_name == "add":
            return agent.add(**kwargs)
        elif tool_name == "greeting":
            return agent.greeting(**kwargs)
        elif tool_name == "rag_query":
            return agent.rag_query(**kwargs)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def list_tools(self) -> list[dict[str, Any]]:
        """List all available tools."""
        return [
            {
                "name": "python_check",
                "description": "Check a Python file for syntax/runtime errors",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Path to the Python file"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                    },
                    "required": ["filepath"],
                },
            },
            {
                "name": "add",
                "description": "Add two numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                },
            },
            {
                "name": "greeting",
                "description": "Return a greeting message",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "rag_query",
                "description": "Search the local knowledge base (RAG)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "n_results": {"type": "integer", "description": "Number of results", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        ]


__all__ = ["MCPToolsAgent", "MCPClient", "MCPClientSync"]

