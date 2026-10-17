"""MCP adapter: one tool, same pipeline as the REST API. Owner: Person B.

Run: PYTHONPATH=src python -m safepassage.mcp_server   (Streamable HTTP; inspect with MCP Inspector)
"""

from mcp.server import MCPServer

from .service import scan_public_url

mcp = MCPServer("SafePassage LK")


@mcp.tool()
def check_opportunity_url(url: str) -> dict:
    """Analyse a public job, study-abroad or visa offer page for scam risk.

    Returns an ML risk assessment, red-flag indicators, official-source verification and a novelty score.
    The result is a risk assessment, not a confirmed fraud determination.
    """
    return scan_public_url(url).model_dump()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
