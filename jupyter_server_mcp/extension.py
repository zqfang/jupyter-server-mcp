"""Jupyter Server extension for managing MCP server."""

import asyncio
import contextlib
import importlib
import logging

from jupyter_server.extension.application import ExtensionApp
from traitlets import Int, List, Unicode

from .mcp_server import MCPServer

# Import notebook execution tools for context setup
try:
    from . import notebook_execution_tools
    NOTEBOOK_TOOLS_AVAILABLE = True
except ImportError:
    NOTEBOOK_TOOLS_AVAILABLE = False
    notebook_execution_tools = None

logger = logging.getLogger(__name__)


class MCPExtensionApp(ExtensionApp):
    """The Jupyter Server MCP extension app."""

    name = "jupyter_server_mcp"
    description = "Jupyter Server extension providing MCP server for tool registration"

    # Configurable traits
    mcp_port = Int(default_value=3001, help="Port for the MCP server to listen on").tag(
        config=True
    )

    mcp_name = Unicode(
        default_value="Jupyter MCP Server", help="Name for the MCP server"
    ).tag(config=True)

    mcp_tools = List(
        trait=Unicode(),
        default_value=[],
        help=(
            "List of tools to register with the MCP server. "
            "Format: 'module_path:function_name' "
            "(e.g., 'os:getcwd', 'math:sqrt')"
        ),
    ).tag(config=True)

    mcp_server_instance: object | None = None
    mcp_server_task: asyncio.Task | None = None

    def _load_function_from_string(self, tool_spec: str):
        """Load a function from a string specification.

        Args:
            tool_spec: Function specification in format
                'module_path:function_name'

        Returns:
            The loaded function object

        Raises:
            ValueError: If tool_spec format is invalid
            ImportError: If module cannot be imported
            AttributeError: If function not found in module
        """
        if ":" not in tool_spec:
            msg = (
                f"Invalid tool specification '{tool_spec}'. "
                f"Expected format: 'module_path:function_name'"
            )
            raise ValueError(msg)

        module_path, function_name = tool_spec.rsplit(":", 1)

        try:
            module = importlib.import_module(module_path)
            return getattr(module, function_name)
        except ImportError as e:
            msg = f"Could not import module '{module_path}': {e}"
            raise ImportError(msg) from e
        except AttributeError as e:
            msg = f"Function '{function_name}' not found in module '{module_path}': {e}"
            raise AttributeError(msg) from e

    def _register_configured_tools(self):
        """Register tools specified in the mcp_tools configuration."""
        if not self.mcp_tools:
            return

        logger.info(f"Registering {len(self.mcp_tools)} configured tools")

        for tool_spec in self.mcp_tools:
            try:
                function = self._load_function_from_string(tool_spec)
                self.mcp_server_instance.register_tool(function)
                logger.info(f"✅ Registered tool: {tool_spec}")
            except Exception as e:
                logger.error(f"❌ Failed to register tool '{tool_spec}': {e}")
                continue

    def initialize(self):
        """Initialize the extension."""
        super().initialize()
        # serverapp will be available as self.serverapp after parent initialization

    def initialize_handlers(self):
        """Initialize the handlers for the extension."""
        # No HTTP handlers needed - MCP server runs on separate port

    def initialize_settings(self):
        """Initialize settings for the extension."""
        # Configuration is handled by traitlets

    async def start_extension(self):
        """Start the extension - called after Jupyter Server starts."""
        try:
            self.log.info(
                f"Starting MCP server '{self.mcp_name}' on port {self.mcp_port}"
            )

            # Set server context for notebook execution tools
            if NOTEBOOK_TOOLS_AVAILABLE:
                notebook_execution_tools.set_server_context(self.serverapp)
                self.log.info("Server context set for notebook execution tools")

            self.mcp_server_instance = MCPServer(
                parent=self, name=self.mcp_name, port=self.mcp_port
            )

            # Register configured tools
            self._register_configured_tools()

            # Start the MCP server in a background task
            self.mcp_server_task = asyncio.create_task(
                self.mcp_server_instance.start_server()
            )

            # Give the server a moment to start
            await asyncio.sleep(0.5)

            self.log.info(f"✅ MCP server started on port {self.mcp_port}")
            if self.mcp_tools:
                registered_count = len(self.mcp_server_instance._registered_tools)
                self.log.info(f"Registered {registered_count} tools from configuration")
            else:
                self.log.info("Use mcp_server_instance.register_tool() to add tools")

        except Exception as e:
            self.log.error(f"Failed to start MCP server: {e}")
            raise

    async def _start_jupyter_server_extension(self, serverapp):  # noqa: ARG002
        """Start the extension - called after Jupyter Server starts."""
        await self.start_extension()

    async def stop_extension(self):
        """Stop the extension - called when Jupyter Server shuts down."""
        if self.mcp_server_task and not self.mcp_server_task.done():
            self.log.info("Stopping MCP server")
            self.mcp_server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.mcp_server_task

        # Always clean up
        self.mcp_server_task = None
        self.mcp_server_instance = None
        self.log.info("MCP server stopped")
