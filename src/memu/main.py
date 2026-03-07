from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from memu.mcp_server import app as mcp_app

app = mcp_app

# Mount MCP server — exposes all FastAPI routes as MCP tools at /mcp
mcp = FastApiMCP(app)
mcp.mount()