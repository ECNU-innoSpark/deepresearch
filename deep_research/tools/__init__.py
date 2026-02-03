"""
Tools module for Deep Research System.

Provides unified interfaces for various data sources:
- Web Search (Tavily, Google, Bing)
- RAGFlow API
- MCP Protocol
"""

from .search_provider import SearchProvider, SearchResult, get_search_provider
from .ragflow_provider import RAGFlowProvider, RAGFlowResult, get_ragflow_provider
from .mcp_client import MCPClient, MCPResult, get_mcp_client

__all__ = [
    "SearchProvider",
    "SearchResult",
    "get_search_provider",
    "RAGFlowProvider",
    "RAGFlowResult",
    "get_ragflow_provider",
    "MCPClient",
    "MCPResult",
    "get_mcp_client",
]
