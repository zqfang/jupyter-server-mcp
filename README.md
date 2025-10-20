# Jupyter Server MCP Extension

A configurable MCP (Model Context Protocol) server extension for Jupyter Server that allows dynamic registration of Python functions as tools accessible to MCP clients from a running Jupyter Server.

https://github.com/user-attachments/assets/aa779b1c-a443-48d7-b3eb-13f27a4333b3

## Overview

This extension provides a simplified, trait-based approach to exposing Jupyter functionality through the MCP protocol. It can dynamically load and register tools from various Python packages, making them available to AI assistants and other MCP clients.

## Key Features

- **Simplified Architecture**: Direct function registration without complex abstractions
- **Configurable Tool Loading**: Register tools via string specifications (`module:function`)  
- **Jupyter Integration**: Seamless integration with Jupyter Server extension system
- **HTTP Transport**: FastMCP-based HTTP server with proper MCP protocol support
- **Traitlets Configuration**: Full configuration support through Jupyter's traitlets system

## Installation

### Option 1: Standard Installation

```bash
pip install jupyter-server-mcp
```

### Option 2: Development Setup with uv (Recommended)

For faster dependency installation and development:

See [INSTALL.md](INSTALL.md) for detailed instructions.

## Quick Start

### 1. Basic Configuration

Create a `jupyter_config.py` file:

```python
c = get_config()

# Basic MCP server settings
c.MCPExtensionApp.mcp_name = "My Jupyter MCP Server"
c.MCPExtensionApp.mcp_port = 8080

# Register tools from existing packages
c.MCPExtensionApp.mcp_tools = [
    # Standard library tools
    "os:getcwd",
    "json:dumps",
    "time:time",

    # Notebook execution and kernel management tools
    "jupyter_server_mcp.notebook_execution_tools:list_running_kernels",
    "jupyter_server_mcp.notebook_execution_tools:list_available_kernels",
    "jupyter_server_mcp.notebook_execution_tools:execute_notebook_code",
    "jupyter_server_mcp.notebook_execution_tools:setup_notebook",
    "jupyter_server_mcp.notebook_execution_tools:switch_notebook_kernel",
    "jupyter_server_mcp.notebook_execution_tools:shutdown_notebook",
    "jupyter_server_mcp.notebook_execution_tools:modify_notebook_cells",
    "jupyter_server_mcp.notebook_execution_tools:query_notebook"
]
```

### 2. Start Jupyter Server

```bash
jupyter lab --config examples/jupyter_config_with_excution_tools.py
```

The MCP server will start automatically on `http://localhost:8080/mcp`.

### 3. Connect MCP Clients

**Claude Code Configuration:**

Set the following configuration:

```json
"mcpServers": {
  "jupyter-mcp": {
    "type": "http",
    "url": "http://localhost:8080/mcp"
  }
}
```

Or use the `claude` CLI:

```bash
claude mcp add --transport http jupyter-mcp http://localhost:8080/mcp
```

**Gemini CLI Configuration:**

Add the following to `.gemini/settings.json`:

```json
{
  "mcpServers": {
    "jupyter-mcp": {
      "httpUrl": "http://localhost:8080/mcp"
    }
  }
}
```

## Architecture

### Core Components

#### MCPServer (`jupyter_server_mcp.mcp_server.MCPServer`)

A simplified LoggingConfigurable class that manages FastMCP integration:

```python
from jupyter_server_mcp.mcp_server import MCPServer

# Create server
server = MCPServer(name="My Server", port=8080)

# Register functions
def my_tool(message: str) -> str:
    return f"Hello, {message}!"

server.register_tool(my_tool)

# Start server
await server.start_server()
```

**Key Methods:**
- `register_tool(func, name=None, description=None)` - Register a Python function
- `register_tools(tools)` - Register multiple functions (list or dict)
- `list_tools()` - Get list of registered tools
- `start_server(host=None)` - Start the HTTP MCP server

#### MCPExtensionApp (`jupyter_server_mcp.extension.MCPExtensionApp`)

Jupyter Server extension that manages the MCP server lifecycle:

**Configuration Traits:**
- `mcp_name` - Server name (default: "Jupyter MCP Server")
- `mcp_port` - Server port (default: 3001)  
- `mcp_tools` - List of tools to register (format: "module:function")

### Tool Loading System

Tools are loaded using string specifications in the format `module_path:function_name`:

```python
# Examples
"os:getcwd"                                           # Standard library
"jupyter_ai_tools.toolkits.notebook:read_notebook"   # External package
"math:sqrt"                                           # Built-in modules
```

The extension dynamically imports the module and registers the function with FastMCP.

## Configuration Examples

### Minimal Setup
```python
c = get_config()
c.MCPExtensionApp.mcp_port = 8080
```

### Full Configuration
```python
c = get_config()

# MCP Server Configuration
c.MCPExtensionApp.mcp_name = "Advanced Jupyter MCP Server"
c.MCPExtensionApp.mcp_port = 8080
c.MCPExtensionApp.mcp_tools = [
    # File system operations (jupyter-ai-tools)
    "jupyter_ai_tools.toolkits.file_system:read",
    "jupyter_ai_tools.toolkits.file_system:write", 
    "jupyter_ai_tools.toolkits.file_system:edit",
    "jupyter_ai_tools.toolkits.file_system:ls",
    "jupyter_ai_tools.toolkits.file_system:glob",
    
    # Notebook operations (jupyter-ai-tools)
    "jupyter_ai_tools.toolkits.notebook:read_notebook",
    "jupyter_ai_tools.toolkits.notebook:edit_cell",
    "jupyter_ai_tools.toolkits.notebook:add_cell", 
    "jupyter_ai_tools.toolkits.notebook:delete_cell",
    "jupyter_ai_tools.toolkits.notebook:create_notebook",
    
    # Git operations (jupyter-ai-tools)
    "jupyter_ai_tools.toolkits.git:git_status",
    "jupyter_ai_tools.toolkits.git:git_add",
    "jupyter_ai_tools.toolkits.git:git_commit",
    "jupyter_ai_tools.toolkits.git:git_push",
    
    # JupyterLab operations (jupyterlab-commands-toolkit)
    "jupyterlab_commands_toolkit.tools:clear_all_outputs_in_notebook",
    "jupyterlab_commands_toolkit.tools:open_document",
    "jupyterlab_commands_toolkit.tools:open_markdown_file_in_preview_mode",
    "jupyterlab_commands_toolkit.tools:show_diff_of_current_notebook",
    
    # Utility functions  
    "os:getcwd",
    "json:dumps",
    "time:time",
    "platform:system",
]
```

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=jupyter_server_mcp tests/
```

### Project Structure

```
jupyter_server_mcp/
├── jupyter_server_mcp/
│   ├── __init__.py
│   ├── mcp_server.py      # Core MCP server implementation
│   └── extension.py       # Jupyter Server extension
├── tests/
│   ├── test_mcp_server.py # MCPServer tests
│   └── test_extension.py  # Extension tests  
├── demo/
│   ├── jupyter_config.py  # Example configuration
│   └── *.py              # Debug/diagnostic scripts
└── pyproject.toml         # Package configuration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass: `pytest tests/`
5. Submit a pull request
