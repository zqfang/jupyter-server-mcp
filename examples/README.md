# Notebook Execution Tools Examples

This directory contains example configurations for using the Jupyter Server MCP extension with notebook execution tools.

## Quick Start

### 1. Install Dependencies

```bash
# Install the package with all dependencies
pip install -e .
```

### 2. Start Jupyter with Execution Tools

```bash
jupyter lab --config=examples/jupyter_config_with_execution_tools.py
```

The MCP server will start on port 8080 at `http://localhost:8080/mcp`.

### 3. Connect Your AI Assistant

**For Claude Code:**

```bash
claude mcp add --transport http jupyter-mcp http://localhost:8080/mcp
```

Or add to your MCP configuration:
```json
{
  "mcpServers": {
    "jupyter-mcp": {
      "type": "http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

**For Gemini CLI:**

Add to `~/.gemini/settings.json`:
```json
{
  "mcpServers": {
    "jupyter-mcp": {
      "httpUrl": "http://localhost:8080/mcp"
    }
  }
}
```

## Available Tools

### list_running_kernels
Lists all currently running Jupyter kernels.

**Returns:**
- Kernel IDs, names, states, and connection counts
- Last activity timestamps

**Example use case:**
```
User: "What kernels are currently running?"
Assistant: [calls list_running_kernels] "You have 2 kernels running: python3 and julia-1.9"
```

### execute_cell
Executes Python code in a Jupyter kernel.

**Parameters:**
- `code` (required): The code to execute
- `kernel_id` (optional): Use existing kernel or start new one
- `timeout` (optional): Maximum execution time in seconds (default: 30)
- `kernel_name` (optional): Kernel to start if creating new (default: "python3")

**Returns:**
- Execution status, outputs, errors, and execution count
- Kernel ID used for execution

**Example use case:**
```
User: "Calculate the fibonacci sequence up to 100"
Assistant: [calls execute_cell with code="def fib(n):..."]
         "Here's the result: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]"
```

### execute_notebook
Executes all cells in a Jupyter notebook file.

**Parameters:**
- `notebook_path` (required): Path to notebook file
- `kernel_name` (optional): Kernel to use (default: "python3")
- `timeout` (optional): Maximum time per cell in seconds (default: 600)
- `output_path` (optional): Where to save executed notebook

**Returns:**
- Execution status, cell counts, execution time
- Path to saved notebook with outputs
- Any errors encountered

**Example use case:**
```
User: "Run the analysis notebook and save the results"
Assistant: [calls execute_notebook with path="analysis.ipynb"]
         "Executed all 15 cells successfully in 12.3 seconds"
```

### shutdown_kernel
Shuts down a running Jupyter kernel.

**Parameters:**
- `kernel_id` (required): ID of kernel to shutdown

**Returns:**
- Status and confirmation message

**Example use case:**
```
User: "Clean up the test kernel"
Assistant: [calls shutdown_kernel with kernel_id="abc123"]
         "Kernel shutdown successfully"
```

## Workflow Examples

### Example 1: Interactive Data Analysis

```
User: "Create a script to analyze the sales data in data.csv"