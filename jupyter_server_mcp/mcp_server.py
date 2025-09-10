"""Simple MCP server for registering Python functions as tools."""

import logging
from inspect import iscoroutinefunction
from typing import Any, Callable, Dict, List, Optional, Union

from fastmcp import FastMCP
from traitlets import Bool, Int, Unicode
from traitlets.config.configurable import LoggingConfigurable

logger = logging.getLogger(__name__)


class MCPServer(LoggingConfigurable):
    """Simple MCP server that allows registering Python functions as tools."""

    # Configurable traits
    name = Unicode(
        default_value="Jupyter MCP Server", help="Name for the MCP server"
    ).tag(config=True)

    port = Int(
        default_value=3001, help="Port for the MCP server to listen on"
    ).tag(config=True)

    host = Unicode(
        default_value="localhost", help="Host for the MCP server to listen on"
    ).tag(config=True)

    enable_debug_logging = Bool(
        default_value=False, help="Enable debug logging for MCP operations"
    ).tag(config=True)

    def __init__(self, **kwargs):
        """Initialize the MCP server.

        Args:
            **kwargs: Configuration parameters
        """
        super().__init__(**kwargs)

        # Initialize FastMCP and tools registry
        self.mcp = FastMCP(self.name)
        self._registered_tools = {}
        self.log.info(
            f"Initialized MCP server '{self.name}' on {self.host}:{self.port}"
        )

    def register_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Register a Python function as an MCP tool.

        Args:
            func: Python function to register
            name: Optional tool name (defaults to function name)
            description: Optional tool description (defaults to function
                docstring)
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        self.log.info(f"Registering tool: {tool_name}")
        if self.enable_debug_logging:
            self.log.debug(
                f"Tool details - Name: {tool_name}, "
                f"Description: {tool_description}, Async: {iscoroutinefunction(func)}"
            )

        # Register with FastMCP
        self.mcp.tool(func)

        # Keep track for listing
        self._registered_tools[tool_name] = {
            "name": tool_name,
            "description": tool_description,
            "function": func,
            "is_async": iscoroutinefunction(func),
        }

    def register_tools(
        self, tools: Union[List[Callable], Dict[str, Callable]]
    ):
        """Register multiple Python functions as MCP tools.

        Args:
            tools: List of functions or dict mapping names to functions
        """
        if isinstance(tools, list):
            for func in tools:
                self.register_tool(func)
        elif isinstance(tools, dict):
            for name, func in tools.items():
                self.register_tool(func, name=name)
        else:
            raise ValueError(
                "tools must be a list of functions or dict mapping names to functions"
            )

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools."""
        return [
            {"name": tool["name"], "description": tool["description"]}
            for tool in self._registered_tools.values()
        ]

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        return self._registered_tools.get(tool_name)

    async def start_server(self, host: Optional[str] = None):
        """Start the MCP server on the specified host and port."""
        server_host = host or self.host

        self.log.info(
            f"Starting MCP server '{self.name}' on {server_host}:{self.port}"
        )
        self.log.info(
            f"Registered tools: {list(self._registered_tools.keys())}"
        )

        if self.enable_debug_logging:
            self.log.debug(
                f"Server configuration - Host: {server_host}, Port: {self.port}"
            )

        # Start FastMCP server with HTTP transport
        await self.mcp.run_http_async(host=server_host, port=self.port)
