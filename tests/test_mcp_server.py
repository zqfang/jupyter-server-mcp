"""Test the simplified MCP server functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from jupyter_server_mcp.mcp_server import MCPServer


def simple_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


async def async_function(name: str) -> str:
    """Async greeting function."""
    await asyncio.sleep(0.001)  # Small delay
    return f"Hello, {name}!"


def function_with_docstring(message: str) -> str:
    """Print a message to stdout.
    
    This is a more detailed description of what this function does.
    """
    print(message)
    return f"Printed: {message}"


class TestMCPServer:
    """Test MCPServer functionality."""
    
    def test_server_creation(self):
        """Test basic server creation."""
        server = MCPServer()
        
        assert server.name == "Jupyter MCP Server"
        assert server.port == 3001
        assert server.host == "localhost"
        assert server.enable_debug_logging == False
        assert server.mcp is not None
        assert len(server._registered_tools) == 0
    
    def test_server_creation_with_params(self):
        """Test server creation with custom parameters."""
        server = MCPServer(name="Test Server", port=3050, host="0.0.0.0")
        
        assert server.name == "Test Server" 
        assert server.port == 3050
        assert server.host == "0.0.0.0"
        assert server.mcp is not None
    
    def test_register_single_tool(self):
        """Test registering a single tool."""
        server = MCPServer()
        
        server.register_tool(simple_function)
        
        # Check tool was registered
        assert len(server._registered_tools) == 1
        assert "simple_function" in server._registered_tools
        
        tool_info = server._registered_tools["simple_function"]
        assert tool_info["name"] == "simple_function"
        assert tool_info["description"] == "Add two numbers."
        assert tool_info["function"] == simple_function
        assert tool_info["is_async"] == False
    
    def test_register_tool_with_custom_name(self):
        """Test registering a tool with custom name."""
        server = MCPServer()
        
        server.register_tool(simple_function, name="add_numbers")
        
        assert "add_numbers" in server._registered_tools
        assert "simple_function" not in server._registered_tools
        
        tool_info = server._registered_tools["add_numbers"]
        assert tool_info["name"] == "add_numbers"
    
    def test_register_tool_with_custom_description(self):
        """Test registering a tool with custom description.""" 
        server = MCPServer()
        
        server.register_tool(simple_function, description="Custom description")
        
        tool_info = server._registered_tools["simple_function"]
        assert tool_info["description"] == "Custom description"
    
    def test_register_async_tool(self):
        """Test registering an async tool."""
        server = MCPServer()
        
        server.register_tool(async_function)
        
        tool_info = server._registered_tools["async_function"]
        assert tool_info["is_async"] == True
    
    def test_register_tools_as_list(self):
        """Test registering multiple tools as a list."""
        server = MCPServer()
        
        server.register_tools([simple_function, async_function])
        
        assert len(server._registered_tools) == 2
        assert "simple_function" in server._registered_tools
        assert "async_function" in server._registered_tools
    
    def test_register_tools_as_dict(self):
        """Test registering multiple tools as a dict."""
        server = MCPServer()
        
        tools = {
            "add": simple_function,
            "greet": async_function
        }
        server.register_tools(tools)
        
        assert len(server._registered_tools) == 2
        assert "add" in server._registered_tools
        assert "greet" in server._registered_tools
        
        assert server._registered_tools["add"]["function"] == simple_function
        assert server._registered_tools["greet"]["function"] == async_function
    
    def test_register_tools_invalid_type(self):
        """Test registering tools with invalid type."""
        server = MCPServer()
        
        with pytest.raises(ValueError, match="tools must be a list of functions or dict"):
            server.register_tools("invalid")
    
    def test_list_tools(self):
        """Test listing registered tools."""
        server = MCPServer()
        
        # Initially empty
        assert server.list_tools() == []
        
        # After registering tools
        server.register_tool(simple_function)
        server.register_tool(function_with_docstring)
        
        tools = server.list_tools()
        assert len(tools) == 2
        
        tool_names = [t["name"] for t in tools]
        assert "simple_function" in tool_names
        assert "function_with_docstring" in tool_names
        
        # Check structure
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
    
    def test_get_tool_info(self):
        """Test getting tool information."""
        server = MCPServer()
        
        # Non-existent tool
        assert server.get_tool_info("nonexistent") is None
        
        # Existing tool
        server.register_tool(simple_function)
        info = server.get_tool_info("simple_function")
        
        assert info is not None
        assert info["name"] == "simple_function"
        assert info["function"] == simple_function


class TestMCPServerDirect:
    """Test direct MCPServer instantiation."""
    
    def test_create_server_defaults(self):
        """Test creating server with defaults."""
        server = MCPServer()
        
        assert isinstance(server, MCPServer)
        assert server.name == "Jupyter MCP Server"
        assert server.port == 3001
    
    def test_create_server_with_params(self):
        """Test creating server with custom parameters."""
        server = MCPServer(name="Custom Server", port=3055)
        
        assert server.name == "Custom Server"
        assert server.port == 3055


class TestMCPServerIntegration:
    """Integration tests for MCP server."""
    
    @pytest.mark.integration
    def test_server_with_multiple_tools(self):
        """Test server with multiple different types of tools."""
        server = MCPServer(name="Integration Test Server")
        
        # Register various types of tools
        server.register_tool(simple_function)
        server.register_tool(async_function)
        server.register_tool(function_with_docstring, name="printer")
        
        # Verify all registered
        assert len(server._registered_tools) == 3
        tools = server.list_tools()
        tool_names = [t["name"] for t in tools]
        
        assert "simple_function" in tool_names
        assert "async_function" in tool_names
        assert "printer" in tool_names
        
        # Verify async detection
        assert server._registered_tools["simple_function"]["is_async"] == False
        assert server._registered_tools["async_function"]["is_async"] == True
        assert server._registered_tools["printer"]["is_async"] == False