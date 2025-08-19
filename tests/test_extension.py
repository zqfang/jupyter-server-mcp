"""Test Jupyter Server extension functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from jupyter_server_docs_mcp.extension import MCPExtensionApp


class TestMCPExtensionApp:
    """Test MCPExtensionApp functionality."""
    
    def test_extension_creation(self):
        """Test creating extension with defaults."""
        extension = MCPExtensionApp()
        
        assert extension.name == "jupyter_server_docs_mcp"
        assert extension.mcp_port == 3001
        assert extension.mcp_name == "Jupyter MCP Server"
    
    def test_extension_trait_configuration(self):
        """Test configuring extension via traits."""
        extension = MCPExtensionApp()
        
        # Test port configuration
        extension.mcp_port = 3010
        assert extension.mcp_port == 3010
        
        # Test name configuration
        extension.mcp_name = "My Custom Server"
        assert extension.mcp_name == "My Custom Server"
        
        # Test tools configuration
        extension.mcp_tools = ["os:getcwd", "math:sqrt"]
        assert extension.mcp_tools == ["os:getcwd", "math:sqrt"]
    
    def test_initialize_handlers(self):
        """Test handler initialization (should be no-op)."""
        extension = MCPExtensionApp()
        # Should not raise any errors
        extension.initialize_handlers()
    
    def test_initialize_settings(self):
        """Test settings initialization (should be no-op)."""
        extension = MCPExtensionApp()
        # Should not raise any errors
        extension.initialize_settings()


class TestMCPExtensionLifecycle:
    """Test extension lifecycle methods."""
    
    @pytest.mark.asyncio
    async def test_start_extension_success(self):
        """Test successful extension startup."""
        extension = MCPExtensionApp()
        extension.mcp_port = 3098
        extension.mcp_name = "Test Server"
        
        # Mock the MCP server creation to avoid actual server startup
        with patch('jupyter_server_docs_mcp.extension.MCPServer') as mock_mcp_class:
            mock_server = Mock()
            mock_server.start_server = AsyncMock()
            mock_mcp_class.return_value = mock_server
            
            await extension.start_extension()
            
            # Verify server was created and started
            mock_mcp_class.assert_called_once_with(
                name=extension.mcp_name,
                port=extension.mcp_port
            )
            mock_server.start_server.assert_called_once()
            
            # Verify extension state
            assert extension.mcp_server_instance == mock_server
            assert extension.mcp_server_task is not None
    
    @pytest.mark.asyncio
    async def test_start_extension_failure(self):
        """Test extension startup failure handling."""
        extension = MCPExtensionApp()
        
        # Mock server creation to raise an exception
        with patch('jupyter_server_docs_mcp.extension.MCPServer') as mock_mcp_class:
            mock_mcp_class.side_effect = Exception("Server creation failed")
            
            with pytest.raises(Exception, match="Server creation failed"):
                await extension.start_extension()
    
    @pytest.mark.asyncio
    async def test_stop_extension_with_running_server(self):
        """Test stopping extension with running server."""
        extension = MCPExtensionApp()
        
        # Create a real asyncio task that can be cancelled
        async def dummy_task():
            await asyncio.sleep(10)  # Long running task
        
        task = asyncio.create_task(dummy_task())
        
        # Set up extension state
        extension.mcp_server_task = task
        extension.mcp_server_instance = Mock()
        
        await extension.stop_extension()
        
        # Verify cleanup
        assert task.cancelled()
        assert extension.mcp_server_task is None
        assert extension.mcp_server_instance is None
    
    @pytest.mark.asyncio
    async def test_stop_extension_no_server(self):
        """Test stopping extension when no server is running."""
        extension = MCPExtensionApp()
        
        # No server running
        extension.mcp_server_task = None
        extension.mcp_server_instance = None
        
        # Should not raise any errors
        await extension.stop_extension()
    
    @pytest.mark.asyncio
    async def test_stop_extension_completed_task(self):
        """Test stopping extension with completed task."""
        extension = MCPExtensionApp()
        
        # Create a mock completed task
        mock_task = Mock()
        mock_task.done.return_value = True
        
        extension.mcp_server_task = mock_task
        extension.mcp_server_instance = Mock()
        
        await extension.stop_extension()
        
        # Should not try to cancel completed task
        mock_task.cancel.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test complete start -> stop lifecycle."""
        extension = MCPExtensionApp()
        extension.mcp_port = 3099
        extension.mcp_name = "Lifecycle Test Server"
        
        # Mock the MCP server
        with patch('jupyter_server_docs_mcp.extension.MCPServer') as mock_mcp_class:
            mock_server = Mock()
            mock_server.start_server = AsyncMock()
            mock_mcp_class.return_value = mock_server
            
            # Start extension
            await extension.start_extension()
            
            # Verify started
            assert extension.mcp_server_instance is not None
            assert extension.mcp_server_task is not None
            
            # The task should be created by start_extension
            original_task = extension.mcp_server_task
            
            # Stop extension  
            await extension.stop_extension()
            
            # Verify stopped
            assert extension.mcp_server_instance is None
            assert extension.mcp_server_task is None
            assert original_task.cancelled()


class TestExtensionIntegration:
    """Integration tests for extension."""
    
    @pytest.mark.integration
    def test_extension_with_real_configuration(self):
        """Test extension with realistic configuration."""
        extension = MCPExtensionApp()
        
        # Configure like a real deployment
        extension.mcp_port = 3020
        extension.mcp_name = "Production MCP Server"
        
        # Should initialize without errors
        extension.initialize_handlers()
        extension.initialize_settings()
        
        # Configuration should be preserved
        assert extension.mcp_port == 3020
        assert extension.mcp_name == "Production MCP Server"


class TestToolLoading:
    """Test tool loading functionality."""
    
    def test_load_function_from_string_valid(self):
        """Test loading valid functions from string specs."""
        extension = MCPExtensionApp()
        
        # Test loading os.getcwd
        func = extension._load_function_from_string("os:getcwd")
        assert callable(func)
        assert func.__name__ == "getcwd"
        
        # Test loading math.sqrt
        func = extension._load_function_from_string("math:sqrt")
        assert callable(func)
        assert func.__name__ == "sqrt"
    
    def test_load_function_from_string_invalid_format(self):
        """Test loading functions with invalid format."""
        extension = MCPExtensionApp()
        
        with pytest.raises(ValueError, match="Invalid tool specification"):
            extension._load_function_from_string("invalid_format")
        
        with pytest.raises(ValueError, match="Invalid tool specification"):
            extension._load_function_from_string("no_colon_here")
    
    def test_load_function_from_string_invalid_module(self):
        """Test loading functions from non-existent modules."""
        extension = MCPExtensionApp()
        
        with pytest.raises(ImportError, match="Could not import module"):
            extension._load_function_from_string("nonexistent_module:some_func")
    
    def test_load_function_from_string_invalid_function(self):
        """Test loading non-existent functions from valid modules."""
        extension = MCPExtensionApp()
        
        with pytest.raises(AttributeError, match="Function.*not found"):
            extension._load_function_from_string("os:nonexistent_function")
    
    def test_load_function_with_nested_module(self):
        """Test loading functions from nested modules."""
        extension = MCPExtensionApp()
        
        # Test loading from json.dumps 
        func = extension._load_function_from_string("json:dumps")
        assert callable(func)
        assert func.__name__ == "dumps"
    
    def test_register_configured_tools_empty(self):
        """Test registering tools when mcp_tools is empty."""
        extension = MCPExtensionApp()
        extension.mcp_server_instance = Mock()
        extension.mcp_tools = []
        
        # Should not call register_tool
        extension._register_configured_tools()
        extension.mcp_server_instance.register_tool.assert_not_called()
    
    def test_register_configured_tools_valid(self):
        """Test registering valid configured tools."""
        extension = MCPExtensionApp()
        extension.mcp_server_instance = Mock()
        extension.mcp_tools = ["os:getcwd", "math:sqrt"]
        
        # Capture log output
        import logging
        with patch('jupyter_server_docs_mcp.extension.logger') as mock_logger:
            extension._register_configured_tools()
            
            # Should register both tools
            assert extension.mcp_server_instance.register_tool.call_count == 2
            
            # Check log messages
            mock_logger.info.assert_any_call("Registering 2 configured tools")
            mock_logger.info.assert_any_call("✅ Registered tool: os:getcwd")
            mock_logger.info.assert_any_call("✅ Registered tool: math:sqrt")
    
    def test_register_configured_tools_with_errors(self):
        """Test registering tools when some fail to load."""
        extension = MCPExtensionApp()
        extension.mcp_server_instance = Mock()
        extension.mcp_tools = ["os:getcwd", "invalid:function", "math:sqrt"]
        
        with patch('jupyter_server_docs_mcp.extension.logger') as mock_logger:
            extension._register_configured_tools()
            
            # Should register 2 valid tools (os:getcwd and math:sqrt)
            assert extension.mcp_server_instance.register_tool.call_count == 2
            
            # Check error logging
            mock_logger.error.assert_any_call("❌ Failed to register tool 'invalid:function': Could not import module 'invalid': No module named 'invalid'")


class TestExtensionWithTools:
    """Test extension lifecycle with configured tools."""
    
    @pytest.mark.asyncio
    async def test_start_extension_with_tools(self):
        """Test extension startup with configured tools."""
        extension = MCPExtensionApp()
        extension.mcp_port = 3089
        extension.mcp_name = "Test Server With Tools"
        extension.mcp_tools = ["os:getcwd", "math:sqrt"]
        
        with patch('jupyter_server_docs_mcp.extension.MCPServer') as mock_mcp_class:
            mock_server = Mock()
            mock_server.start_server = AsyncMock()
            mock_server._registered_tools = {"getcwd": {}, "sqrt": {}}  # Mock registered tools
            mock_mcp_class.return_value = mock_server
            
            await extension.start_extension()
            
            # Verify server creation
            mock_mcp_class.assert_called_once_with(
                name="Test Server With Tools",
                port=3089
            )
            
            # Verify tools were registered
            assert mock_server.register_tool.call_count == 2
    
    @pytest.mark.asyncio 
    async def test_start_extension_no_tools(self):
        """Test extension startup with no configured tools."""
        extension = MCPExtensionApp()
        extension.mcp_port = 3088
        extension.mcp_tools = []
        
        with patch('jupyter_server_docs_mcp.extension.MCPServer') as mock_mcp_class:
            mock_server = Mock()
            mock_server.start_server = AsyncMock()
            mock_server._registered_tools = {}
            mock_mcp_class.return_value = mock_server
            
            await extension.start_extension()
            
            # Should not register any tools
            mock_server.register_tool.assert_not_called()