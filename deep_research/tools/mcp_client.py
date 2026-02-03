"""
MCP (Model Context Protocol) Client for Deep Research System.

This module provides integration with MCP servers for accessing
external tools and resources through the Model Context Protocol.
"""

import asyncio
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime
import json

import httpx
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from core.utils import get_logger, generate_id

logger = get_logger(__name__)


class MCPTransportType(str, Enum):
    """MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class MCPResult(BaseModel):
    """Result from an MCP tool call."""
    
    id: str = Field(default_factory=lambda: generate_id("mcp"))
    
    # Content
    content: str = Field(..., description="Result content")
    content_type: str = Field(default="text", description="Content MIME type")
    
    # Source information
    server_name: str = Field(..., description="MCP server name")
    tool_name: str = Field(..., description="Tool that was called")
    
    # Status
    success: bool = Field(default=True, description="Whether the call succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=datetime.now)


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    
    name: str = Field(..., description="Server identifier")
    transport: MCPTransportType = Field(default=MCPTransportType.STDIO)
    enabled: bool = Field(default=True)
    
    # For stdio transport
    command: Optional[str] = Field(None, description="Command to run")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    # For SSE/HTTP transport
    url: Optional[str] = Field(None, description="Server URL")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")


class MCPTool(BaseModel):
    """An MCP tool definition."""
    
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    server_name: str = ""


class BaseMCPClient(ABC):
    """Abstract base class for MCP clients."""
    
    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the server."""
        pass
    
    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call a tool on the server."""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        pass


class StdioMCPClient(BaseMCPClient):
    """
    MCP client using stdio transport.
    
    Communicates with MCP servers via stdin/stdout using JSON-RPC.
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._tools: List[MCPTool] = []
    
    async def connect(self) -> bool:
        """Start the MCP server process."""
        if not self.config.command:
            logger.error("No command specified for stdio MCP server")
            return False
        
        try:
            cmd = [self.config.command] + self.config.args
            env = {**dict(subprocess.os.environ), **self.config.env}
            
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            
            # Initialize the server
            response = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "deep-research", "version": "1.0.0"},
            })
            
            if response.get("result"):
                logger.info(f"Connected to MCP server: {self.config.name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.config.name}", error=str(e))
            return False
    
    async def disconnect(self) -> None:
        """Stop the MCP server process."""
        if self._process:
            self._process.terminate()
            self._process = None
    
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC request to the server."""
        if not self._process:
            return {"error": "Not connected"}
        
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        
        try:
            # Write request
            request_bytes = (json.dumps(request) + "\n").encode()
            self._process.stdin.write(request_bytes)
            self._process.stdin.flush()
            
            # Read response
            response_line = self._process.stdout.readline()
            if response_line:
                return json.loads(response_line.decode())
            return {"error": "No response"}
            
        except Exception as e:
            return {"error": str(e)}
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the MCP server."""
        response = await self._send_request("tools/list", {})
        
        tools = []
        for tool_data in response.get("result", {}).get("tools", []):
            tools.append(MCPTool(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                parameters=tool_data.get("inputSchema", {}),
                server_name=self.config.name,
            ))
        
        self._tools = tools
        return tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call a tool on the MCP server."""
        response = await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        
        result = response.get("result", {})
        content_items = result.get("content", [])
        
        # Extract text content
        content_parts = []
        for item in content_items:
            if item.get("type") == "text":
                content_parts.append(item.get("text", ""))
        
        return MCPResult(
            content="\n".join(content_parts) if content_parts else str(result),
            server_name=self.config.name,
            tool_name=tool_name,
            success="error" not in response,
            error_message=response.get("error", {}).get("message") if "error" in response else None,
        )


class HttpMCPClient(BaseMCPClient):
    """
    MCP client using HTTP/SSE transport.
    
    Communicates with MCP servers via HTTP requests.
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._connected = False
        self._tools: List[MCPTool] = []
    
    async def connect(self) -> bool:
        """Verify connection to HTTP MCP server."""
        if not self.config.url:
            logger.error("No URL specified for HTTP MCP server")
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.config.url}/health",
                    headers=self.config.headers,
                )
                self._connected = response.status_code == 200
                return self._connected
        except Exception as e:
            logger.error(f"Failed to connect to HTTP MCP server", error=str(e))
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from HTTP MCP server."""
        self._connected = False
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools from the HTTP MCP server."""
        if not self.config.url:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.config.url}/tools",
                    headers=self.config.headers,
                )
                response.raise_for_status()
                data = response.json()
                
                tools = []
                for tool_data in data.get("tools", []):
                    tools.append(MCPTool(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        parameters=tool_data.get("inputSchema", {}),
                        server_name=self.config.name,
                    ))
                
                self._tools = tools
                return tools
                
        except Exception as e:
            logger.error("Failed to list MCP tools", error=str(e))
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call a tool on the HTTP MCP server."""
        if not self.config.url:
            return MCPResult(
                content="",
                server_name=self.config.name,
                tool_name=tool_name,
                success=False,
                error_message="Server URL not configured",
            )
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.config.url}/tools/{tool_name}",
                    headers=self.config.headers,
                    json=arguments,
                )
                response.raise_for_status()
                data = response.json()
                
                return MCPResult(
                    content=data.get("content", str(data)),
                    server_name=self.config.name,
                    tool_name=tool_name,
                    success=True,
                    metadata=data.get("metadata", {}),
                )
                
        except Exception as e:
            return MCPResult(
                content="",
                server_name=self.config.name,
                tool_name=tool_name,
                success=False,
                error_message=str(e),
            )


class MockMCPClient(BaseMCPClient):
    """Mock MCP client for testing."""
    
    def __init__(self, server_name: str = "mock"):
        self.server_name = server_name
        self._connected = False
    
    async def connect(self) -> bool:
        self._connected = True
        return True
    
    async def disconnect(self) -> None:
        self._connected = False
    
    async def list_tools(self) -> List[MCPTool]:
        return [
            MCPTool(
                name="mock_search",
                description="Mock search tool for testing",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                server_name=self.server_name,
            ),
            MCPTool(
                name="mock_fetch",
                description="Mock fetch tool for testing",
                parameters={"type": "object", "properties": {"url": {"type": "string"}}},
                server_name=self.server_name,
            ),
        ]
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        return MCPResult(
            content=f"Mock MCP result for tool '{tool_name}' with args: {arguments}",
            server_name=self.server_name,
            tool_name=tool_name,
            success=True,
        )


class MCPClient:
    """
    Unified MCP client that manages multiple MCP servers.
    
    This class handles the lifecycle of MCP server connections
    and provides a unified interface for tool discovery and invocation.
    """
    
    def __init__(self, use_mock: bool = False):
        """
        Initialize the MCP client manager.
        
        Args:
            use_mock: Whether to use mock clients for testing
        """
        self._config = get_config().get_mcp_config()
        self._use_mock = use_mock
        self._clients: Dict[str, BaseMCPClient] = {}
        self._all_tools: Dict[str, MCPTool] = {}  # tool_name -> tool
    
    @property
    def enabled(self) -> bool:
        """Check if MCP is enabled."""
        return self._config.get("enabled", False) or self._use_mock
    
    async def initialize(self) -> None:
        """Initialize all configured MCP servers."""
        if not self.enabled:
            logger.info("MCP is disabled")
            return
        
        if self._use_mock:
            mock_client = MockMCPClient()
            await mock_client.connect()
            self._clients["mock"] = mock_client
            tools = await mock_client.list_tools()
            for tool in tools:
                self._all_tools[tool.name] = tool
            return
        
        # Initialize real MCP servers
        for server_config in self._config.get("servers", []):
            if not server_config.get("enabled", True):
                continue
            
            config = MCPServerConfig(**server_config)
            
            # Create appropriate client based on transport
            if config.transport == MCPTransportType.STDIO:
                client = StdioMCPClient(config)
            else:  # HTTP/SSE
                client = HttpMCPClient(config)
            
            # Connect and list tools
            if await client.connect():
                self._clients[config.name] = client
                tools = await client.list_tools()
                for tool in tools:
                    self._all_tools[tool.name] = tool
                logger.info(f"MCP server initialized", server=config.name, tools=len(tools))
    
    async def shutdown(self) -> None:
        """Disconnect from all MCP servers."""
        for name, client in self._clients.items():
            try:
                await client.disconnect()
                logger.info(f"Disconnected from MCP server: {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}", error=str(e))
        self._clients.clear()
        self._all_tools.clear()
    
    def list_all_tools(self) -> List[MCPTool]:
        """List all available tools from all connected servers."""
        return list(self._all_tools.values())
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a specific tool by name."""
        return self._all_tools.get(tool_name)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """
        Call a tool by name.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            MCPResult with the tool output
        """
        tool = self._all_tools.get(tool_name)
        if not tool:
            return MCPResult(
                content="",
                server_name="unknown",
                tool_name=tool_name,
                success=False,
                error_message=f"Tool '{tool_name}' not found",
            )
        
        client = self._clients.get(tool.server_name)
        if not client:
            return MCPResult(
                content="",
                server_name=tool.server_name,
                tool_name=tool_name,
                success=False,
                error_message=f"Server '{tool.server_name}' not connected",
            )
        
        return await client.call_tool(tool_name, arguments)
    
    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Synchronous version of call_tool."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.call_tool(tool_name, arguments))


# Global MCP client instance
_mcp_client: Optional[MCPClient] = None


def get_mcp_client(use_mock: bool = False) -> MCPClient:
    """
    Get the global MCP client instance.
    
    Args:
        use_mock: Whether to use mock client
        
    Returns:
        MCPClient instance
    """
    global _mcp_client
    if _mcp_client is None or _mcp_client._use_mock != use_mock:
        _mcp_client = MCPClient(use_mock=use_mock)
    return _mcp_client


async def initialize_mcp(use_mock: bool = False) -> MCPClient:
    """
    Initialize and return the MCP client.
    
    Args:
        use_mock: Whether to use mock client
        
    Returns:
        Initialized MCPClient
    """
    client = get_mcp_client(use_mock)
    await client.initialize()
    return client
