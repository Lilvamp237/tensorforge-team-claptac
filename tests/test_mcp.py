import pytest
from mcp.server.mcpserver.exceptions import ToolError

from safepassage import mcp_server
from safepassage.schema import Risk, ScanResult


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_tool_is_listed():
    tools = await mcp_server.mcp.list_tools()
    assert [t.name for t in tools] == ["check_opportunity_url"]


def test_tool_uses_shared_pipeline(monkeypatch):
    fake = ScanResult(scan_id="scan_x", source="url", url="https://jobs.example", risk=Risk(label="high", probability=0.9),
                      needs_review=False, content={}, indicators=["Advance payment requested"], verification={}, novelty={})
    monkeypatch.setattr(mcp_server, "scan_public_url", lambda url: fake)
    out = mcp_server.check_opportunity_url("https://jobs.example")
    assert out["risk"]["label"] == "high" and out["disclaimer"]


def test_tool_refuses_internal_addresses():
    with pytest.raises(ToolError, match="not allowed"):
        mcp_server.check_opportunity_url("http://169.254.169.254/latest/meta-data")
