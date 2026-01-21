from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .mcp_tools.python_checker import register as register_python_checker

# create an mcp server
mcp = FastMCP("Demo")


# add an addition tool
@mcp.tool()
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Return a greeting message for the given name."""
    return f"Hello, {name}!"



# activate the python checker tool, details are in mcp_tools/python_checker.py
# activate the python checker tool, details are in mcp_tools/python_checker.py
register_python_checker(mcp)

# RAG tool
from .mcp_tools.rag_search import register as register_rag_search
register_rag_search(mcp)


if __name__ == "__main__":
    mcp.run(transport="stdio")  # run the server using stdio transport
