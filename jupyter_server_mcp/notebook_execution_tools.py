"""Notebook execution tools for Jupyter Server MCP extension.

This module provides tools for executing code and managing Jupyter kernels.
"""

import asyncio
import inspect
import logging
import os
import time
from typing import Any

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from nbformat.notebooknode import from_dict
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

logger = logging.getLogger(__name__)

# Global context to store the Jupyter ServerApp instance
_server_context = {"serverapp": None}

# Track active kernels per notebook to support multi-kernel workflows
# Keyed by notebook_path (absolute path)
_notebook_kernels: dict[str, str] = {}  # Maps notebook_path -> kernel_id


# ============================================================================
# Helper Functions
# ============================================================================

def _normalize_path(path: str) -> str:
    """Normalize a file path to an absolute path."""
    return os.path.abspath(path)


def _read_notebook(notebook_path: str):
    """Read a notebook file and return the notebook object."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


def _write_notebook(notebook_path: str, nb) -> None:
    """Write a notebook object to a file."""
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def _track_kernel(notebook_path: str, kernel_id: str) -> None:
    """Track a kernel for a notebook."""
    if kernel_id:
        _notebook_kernels[notebook_path] = kernel_id


def _get_tracked_kernel(notebook_path: str) -> str | None:
    """Get the tracked kernel ID for a notebook."""
    return _notebook_kernels.get(notebook_path)


def set_server_context(serverapp):
    """Set the Jupyter ServerApp context for tools to use.

    Args:
        serverapp: The Jupyter ServerApp instance
    """
    _server_context["serverapp"] = serverapp
    logger.info("Server context set for notebook execution tools")


def get_server_context():
    """Get the Jupyter ServerApp context.

    Returns:
        The Jupyter ServerApp instance or None if not set
    """
    return _server_context.get("serverapp")


async def list_running_kernels() -> dict[str, Any]:
    """List all running Jupyter kernels.

    Returns a list of all currently running kernel sessions with their
    IDs, names, and current state.

    Returns:
        dict: A dictionary containing:
            - kernels: List of kernel information dicts with:
                - id: Kernel ID
                - name: Kernel name (e.g., 'python3')
                - last_activity: Last activity timestamp
                - execution_state: Current execution state
                - connections: Number of connections

    Example:
        >>> result = await list_running_kernels()
        >>> print(result)
        {
            "kernels": [
                {
                    "id": "abc123",
                    "name": "python3",
                    "last_activity": "2024-01-01T12:00:00",
                    "execution_state": "idle",
                    "connections": 1
                }
            ]
        }
    """
    serverapp = get_server_context()
    if serverapp is None:
        return {"error": "Server context not available", "kernels": []}

    try:
        kernel_manager = serverapp.kernel_manager

        # Get list of kernel IDs
        kernel_ids = list(kernel_manager.list_kernel_ids())

        kernels = []
        for kernel_id in kernel_ids:
            try:
                # Get kernel info
                kernel = kernel_manager.get_kernel(kernel_id)

                kernel_info = {
                    "id": kernel_id,
                    "name": kernel.kernel_name if hasattr(kernel, "kernel_name") else "unknown",
                    "last_activity": kernel.last_activity.isoformat() if hasattr(kernel, "last_activity") else None,
                    "execution_state": getattr(kernel, "execution_state", "unknown"),
                    "connections": kernel.connection_count if hasattr(kernel, "connection_count") else 0,
                }
                kernels.append(kernel_info)
            except Exception as e:
                logger.warning(f"Error getting info for kernel {kernel_id}: {e}")
                continue

        logger.info(f"Listed {len(kernels)} kernels")
        return {"kernels": kernels}

    except Exception as e:
        logger.error(f"Error listing kernels: {e}")
        return {"error": str(e), "kernels": []}


async def list_available_kernels() -> dict[str, Any]:
    """List all available (installed) kernel specifications.

    Returns information about all kernel specifications that are installed
    and available for use. This is useful for discovering what kernels
    can be used with setup_notebook or switch_notebook_kernel.

    This function can be used to check whether a specific kernel (e.g., 'python3',
    'ir', 'julia-1.9') is available before creating a notebook or starting a kernel.

    Returns:
        dict: A dictionary containing:
            - kernelspecs: Dict mapping kernel names to kernel info with:
                - name: Kernel name (e.g., 'python3', 'ir', 'julia-1.9')
                - resource_dir: Path to kernel resource directory
                - spec: Full kernel specification dict with:
                    - display_name: Human-readable name
                    - language: Programming language
                    - argv: Command line arguments to start kernel
                    - env: Environment variables (optional)
                    - metadata: Additional metadata (optional)

    Example:
        >>> result = await list_available_kernels()
        >>> print(result)
        {
            "kernelspecs": {
                "python3": {
                    "name": "python3",
                    "resource_dir": "/usr/local/share/jupyter/kernels/python3",
                    "spec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "argv": ["/usr/bin/python3", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                        "env": {},
                        "metadata": {}
                    }
                },
                "ir": {
                    "name": "ir",
                    "resource_dir": "/usr/local/share/jupyter/kernels/ir",
                    "spec": {
                        "display_name": "R",
                        "language": "R",
                        "argv": ["/usr/bin/R", "--slave", "-e", "IRkernel::main()", "--args", "{connection_file}"]
                    }
                }
            }
        }

        >>> # Check if a kernel is available
        >>> result = await list_available_kernels()
        >>> if "python3" in result["kernelspecs"]:
        ...     print("Python 3 kernel is available!")
    """
    try:
        # Create KernelSpecManager to access installed kernel specifications
        ksm = KernelSpecManager()

        # Get all kernel specs
        all_specs = ksm.get_all_specs()

        # Format the output to be more user-friendly
        kernelspecs = {}
        for kernel_name, spec_info in all_specs.items():
            kernelspecs[kernel_name] = {
                "name": kernel_name,
                "resource_dir": spec_info.get("resource_dir", ""),
                "spec": spec_info.get("spec", {})
            }

        logger.info(f"Found {len(kernelspecs)} available kernel specifications")
        return {"kernelspecs": kernelspecs}

    except Exception as e:
        logger.error(f"Error listing available kernels: {e}")
        return {"error": str(e), "kernelspecs": {}}


async def execute_cell(
    code: str,
    kernel_id: str | None = None,
    timeout: int = 30,
    kernel_name: str = "python3"
) -> dict[str, Any]:
    """Execute code in a Jupyter kernel.

    Executes the provided code in a specified kernel or starts a new one.
    Returns the execution results including outputs, errors, and execution count.

    Args:
        code: The code to execute
        kernel_id: Optional kernel ID to use. If None, starts a new kernel.
        timeout: Maximum time in seconds to wait for execution (default: 30)
        kernel_name: Kernel name to use when starting new kernel (default: "python3")

    Returns:
        dict: A dictionary containing:
            - status: "ok" or "error"
            - kernel_id: ID of the kernel used
            - execution_count: Execution counter value
            - outputs: List of outputs (text, images, errors, etc.)
            - error: Error message if execution failed

    Example:
        >>> result = await execute_cell("print('Hello, World!')")
        >>> print(result)
        {
            "status": "ok",
            "kernel_id": "abc123",
            "execution_count": 1,
            "outputs": [{"output_type": "stream", "name": "stdout", "text": "Hello, World!\\n"}]
        }
    """
    serverapp = get_server_context()
    if serverapp is None:
        return {"error": "Server context not available", "status": "error"}

    kernel_manager = serverapp.kernel_manager
    started_kernel = False

    try:
        # Get or start kernel
        if kernel_id is None:
            # Start a new kernel
            logger.info(f"Starting new kernel: {kernel_name}")
            kernel_id = await kernel_manager.start_kernel(kernel_name=kernel_name)
            started_kernel = True
            logger.info(f"Started kernel: {kernel_id}")

            # Wait a bit for kernel to be ready
            await asyncio.sleep(1)

        # Get the kernel
        kernel = kernel_manager.get_kernel(kernel_id)

        # Create a client to communicate with the kernel
        client = kernel.client()
        client.start_channels()

        try:
            # Wait for kernel to be ready
            await asyncio.wait_for(client.wait_for_ready(), timeout=10)

            # Execute the code
            logger.info(f"Executing code in kernel {kernel_id}")
            msg_id = client.execute(code)

            # Collect outputs
            outputs = []
            execution_count = None
            status = "ok"
            error_msg = None

            # Wait for execution to complete
            start_time = asyncio.get_event_loop().time()
            while True:
                if asyncio.get_event_loop().time() - start_time > timeout:
                    status = "error"
                    error_msg = f"Execution timed out after {timeout} seconds"
                    break

                try:
                    # Directly await the async method - no executor needed for AsyncKernelClient
                    msg = await asyncio.wait_for(
                        client.get_iopub_msg(timeout=0.5),
                        timeout=1
                    )

                    # Validate message structure
                    if not isinstance(msg, dict):
                        logger.error(f"Invalid message type from kernel: {type(msg)}")
                        status = "error"
                        error_msg = f"Invalid message type: {type(msg).__name__}"
                        break

                    try:
                        msg_type = msg["header"]["msg_type"]
                        content = msg["content"]
                    except (KeyError, TypeError) as e:
                        logger.error(f"Malformed kernel message: {e}")
                        status = "error"
                        error_msg = f"Malformed kernel message: {str(e)}"
                        break

                    # Check if this is a reply to our execute request
                    if msg["parent_header"].get("msg_id") != msg_id:
                        continue

                    # Handle different message types
                    if msg_type == "execute_result":
                        execution_count = content.get("execution_count")
                        outputs.append({
                            "output_type": "execute_result",
                            "data": content.get("data", {}),
                            "metadata": content.get("metadata", {}),
                            "execution_count": execution_count
                        })

                    elif msg_type == "stream":
                        outputs.append({
                            "output_type": "stream",
                            "name": content.get("name", "stdout"),
                            "text": content.get("text", "")
                        })

                    elif msg_type == "display_data":
                        outputs.append({
                            "output_type": "display_data",
                            "data": content.get("data", {}),
                            "metadata": content.get("metadata", {})
                        })

                    elif msg_type == "error":
                        status = "error"
                        error_msg = "\n".join(content.get("traceback", []))
                        outputs.append({
                            "output_type": "error",
                            "ename": content.get("ename", ""),
                            "evalue": content.get("evalue", ""),
                            "traceback": content.get("traceback", [])
                        })

                    elif msg_type == "status":
                        if content.get("execution_state") == "idle":
                            # Execution complete
                            break

                except asyncio.TimeoutError:
                    # No message received in 1 second, check if we should continue
                    continue

            logger.info(f"Execution complete: status={status}, outputs={len(outputs)}")

            result = {
                "status": status,
                "kernel_id": kernel_id,
                "execution_count": execution_count,
                "outputs": outputs,
                "started_new_kernel": started_kernel
            }

            if error_msg:
                result["error"] = error_msg

            return result

        finally:
            # Clean up client
            client.stop_channels()

    except Exception as e:
        logger.error(f"Error executing cell: {e}")
        return {
            "error": str(e),
            "status": "error",
            "kernel_id": kernel_id,
            "started_new_kernel": started_kernel
        }


async def execute_notebook(
    notebook_path: str,
    kernel_name: str = "python3",
    timeout: int = 600,
    output_path: str | None = None,
    stop_on_error: bool = False
) -> dict[str, Any]:
    """Execute all cells in a Jupyter notebook.

    Runs all cells in the specified notebook file and optionally saves
    the executed notebook with outputs to a new file.

    REFACTORED: Uses _execute_cell_by_position() (via execute_notebook_code pattern)
    instead of NotebookClient.

    Args:
        notebook_path: Path to the notebook file to execute
        kernel_name: Name of the kernel to use (default: "python3")
        timeout: Maximum time in seconds per cell (default: 600)
        output_path: Optional path to save executed notebook. If None, overwrites original.
        stop_on_error: If True, stop execution on first error (default: False)

    Returns:
        dict: A dictionary containing:
            - status: "completed", "error", or "partial"
            - notebook_path: Path to the original notebook
            - output_path: Path where executed notebook was saved
            - cells_executed: Number of cells executed
            - cells_total: Total number of code cells
            - execution_time: Total execution time in seconds
            - kernel_id: ID of the kernel used
            - errors: List of errors encountered (if any)
            - error_count: Number of errors (if any)

    Example:
        >>> result = await execute_notebook("analysis.ipynb", output_path="executed.ipynb")
        >>> print(result)
        {
            "status": "completed",
            "notebook_path": "analysis.ipynb",
            "output_path": "executed.ipynb",
            "cells_executed": 10,
            "cells_total": 10,
            "execution_time": 5.23,
            "kernel_id": "abc123"
        }
    """
    kernel_id = None
    cells_executed = 0

    try:
        # ===================================================================
        # PHASE 1: VALIDATION & SETUP
        # ===================================================================

        # Validate notebook path
        if not os.path.exists(notebook_path):
            return {
                "error": f"Notebook not found: {notebook_path}",
                "status": "error"
            }

        # Read the notebook to get cell positions
        logger.info(f"Reading notebook: {notebook_path}")
        nb = _read_notebook(notebook_path)

        # Find all code cells and their positions
        code_cell_indices = [
            i for i, cell in enumerate(nb.cells)
            if cell.cell_type == "code"
        ]
        total_cells = len(code_cell_indices)

        # Handle empty notebook
        if total_cells == 0:
            logger.info("No code cells to execute")
            return {
                "status": "completed",
                "notebook_path": notebook_path,
                "output_path": output_path or notebook_path,
                "cells_executed": 0,
                "cells_total": 0,
                "execution_time": 0.0,
                "message": "No code cells to execute"
            }

        logger.info(f"Found {total_cells} code cells to execute")

        # Initialize tracking
        start_time = time.time()
        errors = []
        status = "completed"

        # ===================================================================
        # PHASE 2: KERNEL INITIALIZATION
        # ===================================================================

        # Get server context
        serverapp = get_server_context()
        if serverapp is None:
            return {
                "error": "Server context not available",
                "status": "error"
            }

        kernel_manager = serverapp.kernel_manager

        # Check if notebook already has a tracked kernel
        kernel_id = _get_tracked_kernel(notebook_path)

        # If no tracked kernel, start one and track it
        if kernel_id is None:
            logger.info(f"Starting new kernel: {kernel_name}")
            kernel_id = await kernel_manager.start_kernel(kernel_name=kernel_name)
            _track_kernel(notebook_path, kernel_id)
            logger.info(f"Started and tracked kernel: {kernel_id}")

            # Wait for kernel to be ready
            await asyncio.sleep(1)
        else:
            logger.info(f"Using existing tracked kernel: {kernel_id}")

        # ===================================================================
        # PHASE 3: EXECUTE ALL CODE CELLS
        # Uses _execute_cell_by_position which handles notebook I/O per cell
        # ===================================================================

        for idx, cell_position in enumerate(code_cell_indices):
            logger.info(f"Executing cell {idx + 1}/{total_cells} (position {cell_position})")

            try:
                # Use _execute_cell_by_position which:
                # - Reads notebook
                # - Executes cell via execute_cell()
                # - Updates cell outputs
                # - Saves notebook
                result = await _execute_cell_by_position(
                    notebook_path=notebook_path,
                    position_index=cell_position,
                    timeout=timeout
                )

                # Check execution result
                if result.get("status") == "ok":
                    cells_executed += 1
                    logger.info(f"Cell {idx + 1} executed successfully")

                elif result.get("status") == "error":
                    cells_executed += 1

                    # Track error
                    error_info = {
                        "cell_index": cell_position,
                        "cell_number": idx + 1,
                        "error": result.get("error", "Unknown error")
                    }
                    errors.append(error_info)
                    logger.error(f"Cell {idx + 1} failed: {error_info['error']}")

                    # Decide whether to continue or stop
                    if stop_on_error:
                        status = "partial"
                        logger.info("Stopping execution after error (stop_on_error=True)")
                        break
                    else:
                        status = "error"

                # Track kernel_id from first execution
                if kernel_id is None and result.get("kernel_id"):
                    kernel_id = result.get("kernel_id")

            except Exception as e:
                # Unexpected exception during cell execution
                cells_executed += 1
                error_info = {
                    "cell_index": cell_position,
                    "cell_number": idx + 1,
                    "error": f"Exception: {str(e)}",
                    "exception_type": type(e).__name__
                }
                errors.append(error_info)
                logger.error(f"Exception executing cell {idx + 1}: {e}")

                if stop_on_error:
                    status = "partial"
                    break
                else:
                    status = "error"

        # ===================================================================
        # PHASE 4: FINALIZATION
        # ===================================================================

        # Calculate total execution time
        execution_time = time.time() - start_time
        logger.info(f"Notebook execution complete: {cells_executed}/{total_cells} cells in {execution_time:.2f}s")

        # Handle output_path (copy if different from input)
        final_output_path = output_path or notebook_path
        if output_path and output_path != notebook_path:
            import shutil
            shutil.copy2(notebook_path, output_path)
            logger.info(f"Copied executed notebook to: {output_path}")

        # Build comprehensive result
        result = {
            "status": status,
            "notebook_path": notebook_path,
            "output_path": final_output_path,
            "cells_executed": cells_executed,
            "cells_total": total_cells,
            "execution_time": round(execution_time, 2),
            "kernel_id": kernel_id
        }

        # Add errors if any
        if errors:
            result["errors"] = errors
            result["error_count"] = len(errors)

        logger.info(f"Notebook execution result: {result['status']}")
        return result

    except Exception as e:
        # Top-level exception handler
        logger.error(f"Error in execute_notebook: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

        return {
            "error": str(e),
            "status": "error",
            "notebook_path": notebook_path,
            "cells_executed": cells_executed,
            "kernel_id": kernel_id
        }


async def shutdown_kernel(kernel_id: str) -> dict[str, Any]:
    """Shutdown a running Jupyter kernel.

    Stops and removes the specified kernel from the kernel manager.

    This deletes any active Jupyter session(s) for the given notebook and stops the
    corresponding local KernelClient, ensuring a clean slate before switching
    kernels or performing other operations.

    Args:
        kernel_id: The ID of the kernel to shutdown

    Returns:
        dict: A dictionary containing:
            - status: "shutdown" or "error"
            - kernel_id: The ID of the kernel
            - message: Status message
            - error: Error message if shutdown failed

    Example:
        >>> result = await shutdown_kernel("abc123")
        >>> print(result)
        {
            "status": "shutdown",
            "kernel_id": "abc123",
            "message": "Kernel shutdown successfully"
        }
    """
    serverapp = get_server_context()
    if serverapp is None:
        return {
            "error": "Server context not available",
            "status": "error",
            "kernel_id": kernel_id
        }

    try:
        kernel_manager = serverapp.kernel_manager

        # Check if kernel exists
        if kernel_id not in kernel_manager.list_kernel_ids():
            return {
                "error": f"Kernel not found: {kernel_id}",
                "status": "error",
                "kernel_id": kernel_id
            }

        # Shutdown the kernel
        logger.info(f"Shutting down kernel: {kernel_id}")
        await kernel_manager.shutdown_kernel(kernel_id)

        return {
            "status": "shutdown",
            "kernel_id": kernel_id,
            "message": "Kernel shutdown successfully"
        }

    except Exception as e:
        logger.error(f"Error shutting down kernel {kernel_id}: {e}")
        return {
            "error": str(e),
            "status": "error",
            "kernel_id": kernel_id
        }




async def shutdown_notebook(
    notebook_path: str,
    shutdown_kernel: bool = True
) -> dict[str, Any]:
    """Close a notebook session and clean local kernel state.

    This shuts down the kernel associated with a notebook and removes
    the notebook from the active kernel tracking.

    Args:
        notebook_path: Path to the notebook file (relative or absolute)
        shutdown_kernel: If True, shutdown the kernel (default: True)

    Returns:
        dict: Status information containing:
            - message: Status message
            - kernel_id: ID of the kernel that was shut down (if any)

    Example:
        >>> result = await shutdown_notebook("demo.ipynb")
        >>> print(result)
        {
            "message": "Notebook session closed",
            "kernel_id": "abc123"
        }
    """
    notebook_path = _normalize_path(notebook_path)

    serverapp = get_server_context()
    if serverapp is None:
        return {"message": "Server context not available"}

    # Get kernel ID if tracked
    kernel_id = _get_tracked_kernel(notebook_path)

    result = {"message": "Notebook session closed"}

    if kernel_id and shutdown_kernel:
        try:
            kernel_manager = serverapp.kernel_manager
            if kernel_id in kernel_manager.list_kernel_ids():
                await kernel_manager.shutdown_kernel(kernel_id)
                result["kernel_id"] = kernel_id
                logger.info(f"Shut down kernel {kernel_id} for notebook {notebook_path}")
        except Exception as e:
            logger.warning(f"Error shutting down kernel {kernel_id}: {e}")

    # Remove from tracking
    if notebook_path in _notebook_kernels:
        del _notebook_kernels[notebook_path]

    return result


async def switch_notebook_kernel(
    notebook_path: str,
    kernel_name: str
) -> dict[str, Any]:
    """Switch a notebook to use a different kernel.

    This updates the notebook's kernelspec metadata and creates a new
    kernel session with the specified kernel.

    Args:
        notebook_path: Path to the notebook file (relative or absolute)
        kernel_name: Name of the kernel to switch to (e.g., "python3", "ir", "julia-1.10")

    Returns:
        dict: Status information containing:
            - message: Status message
            - kernel_name: Name of the new kernel
            - kernel_id: ID of the new kernel

    Example:
        >>> # Switch from Python to R
        >>> result = await switch_notebook_kernel("analysis.ipynb", "ir")
        >>> print(result)
        {
            "message": "Kernel switched to ir",
            "kernel_name": "ir",
            "kernel_id": "xyz789"
        }
    """
    notebook_path = _normalize_path(notebook_path)

    # First, shutdown existing kernel
    await shutdown_notebook(notebook_path, shutdown_kernel=True)

    # Read the notebook
    nb = _read_notebook(notebook_path)

    # Determine language name based on kernel
    if kernel_name.lower() in ["ir", "r", "irkernel"]:
        language_name = "R"
        display_name = "R"
    elif kernel_name.startswith("python"):
        language_name = "python"
        display_name = f"Python 3" if kernel_name == "python3" else kernel_name
    elif kernel_name.startswith("julia"):
        language_name = "julia"
        display_name = kernel_name
    else:
        language_name = kernel_name
        display_name = kernel_name

    # Update kernelspec metadata
    nb.metadata["kernelspec"] = {
        "name": kernel_name,
        "display_name": display_name,
        "language": language_name
    }
    nb.metadata["language_info"] = {
        "name": language_name
    }

    # Save the updated notebook
    _write_notebook(notebook_path, nb)

    # Start a new kernel with the specified kernel name
    serverapp = get_server_context()
    if serverapp is None:
        raise RuntimeError("Server context not available")

    kernel_manager = serverapp.kernel_manager
    kernel_id = await kernel_manager.start_kernel(kernel_name=kernel_name)

    # Track the new kernel
    _track_kernel(notebook_path, kernel_id)

    logger.info(f"Switched notebook {notebook_path} to kernel {kernel_name} (ID: {kernel_id})")

    return {
        "message": f"Kernel switched to {kernel_name}",
        "kernel_name": kernel_name,
        "kernel_id": kernel_id
    }

async def query_notebook(
    notebook_path: str,
    query_type: str,
    execution_count: int | None = None,
    position_index: int | None = None,
    cell_id: str | None = None,
) -> dict[str, Any] | list | str | int:
    """Query notebook information and metadata.

    This consolidates all read-only operations into a single tool following MCP best practices.

    Args:
        notebook_path: Path to the notebook file (relative or absolute)
        query_type: Type of query to perform. Options:
            - 'view_source': View source code of notebook (single cell or all cells)
            - 'check_server': Check if Jupyter server is running and accessible
            - 'list_sessions': List all notebook sessions on the server
            - 'list_kernels': List all available kernelspecs on the server
            - 'get_position_index': Get the index of a code cell
        execution_count: (For view_source/get_position_index) The execution count to look for.
            IMPORTANT: This is the number shown in square brackets like [3] in Jupyter UI.
            Only available for executed code cells. Must be an integer (e.g., 3).
            COMMON MISTAKE: Don't confuse with position_index!
            - execution_count=3 finds the cell that was executed 3rd (shows [3] in Jupyter)
            - position_index=3 finds the 4th cell in the notebook (0-indexed position)
        position_index: (For view_source) The position index to look for.
            This is the cell's physical position in the notebook (0-indexed).
            Examples: first cell = 0, second cell = 1, third cell = 2, etc.
            Works for all cell types (code, markdown, raw). Must be an integer.
        cell_id: (For get_position_index) Cell ID like "205658d6-093c-4722-854c-90b149f254ad".
            This is a unique identifier for each cell, visible in notebook metadata.

    Returns:
        Union[dict, list, str, int]:
            - view_source: dict (single cell) or list[dict] (all cells) with cell contents/metadata
            - check_server: str status message
            - list_sessions: list of notebook sessions
            - list_kernels: dict of available kernelspecs
            - get_position_index: int positional index

    Examples:
        # View all cells in notebook
        await query_notebook("my_notebook.ipynb", "view_source")

        # View cell by execution count (the [3] shown in Jupyter UI)
        await query_notebook("my_notebook.ipynb", "view_source", execution_count=3)

        # View cell by position (first cell=0, second=1, etc)
        await query_notebook("my_notebook.ipynb", "view_source", position_index=0)

        # Get position index of cell with execution count [5]
        await query_notebook("my_notebook.ipynb", "get_position_index", execution_count=5)

        # List available kernels
        await query_notebook("my_notebook.ipynb", "list_kernels")

    Raises:
        ValueError: If invalid query_type or missing required parameters
    """
    if query_type == "view_source":
        return await _query_view_source(notebook_path, execution_count, position_index)
    elif query_type == "check_server":
        return _query_check_server()
    elif query_type == "list_sessions":
        return await _query_list_sessions()
    elif query_type == "list_kernels":
        return await list_available_kernels()
    elif query_type == "get_position_index":
        return await _query_get_position_index(notebook_path, execution_count, cell_id)
    else:
        raise ValueError(
            f"Invalid query_type: {query_type}. Must be one of: view_source, check_server, list_sessions, list_kernels, get_position_index"
        )


async def _query_view_source(
    notebook_path: str,
    execution_count: int | None = None,
    position_index: int | None = None,
) -> dict | list[dict]:
    """View the source code of a Jupyter notebook (either single cell or all cells)."""
    if execution_count is not None and position_index is not None:
        raise ValueError("Cannot provide both execution_count and position_index.")

    serverapp = get_server_context()
    if serverapp is None:
        raise RuntimeError("Server context not available")

    # Normalize path
    notebook_path = _normalize_path(notebook_path)

    # Read the notebook
    try:
        nb = _read_notebook(notebook_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    # View all cells if no specific cell requested
    if execution_count is None and position_index is None:
        logger.info("Viewing all cells")
        return _filter_cell_outputs(nb.cells)

    # Find cell by execution_count if provided
    if position_index is None and execution_count is not None:
        position_indices = []
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and cell.get("execution_count") == execution_count:
                position_indices.append(i)

        if len(position_indices) == 0:
            cells_info = _get_available_execution_counts(nb)
            error_msg = f"No cells found with execution count {execution_count}."
            if cells_info["execution_counts"]:
                error_msg += f" Available execution counts: {cells_info['execution_counts']}"
            else:
                error_msg += " No cells have been executed yet."
            raise ValueError(error_msg)
        elif len(position_indices) > 1:
            raise ValueError(f"Multiple cells found with execution count {execution_count}")

        position_index = position_indices[0]

    # Get cell by position index
    if position_index >= len(nb.cells):
        raise IndexError(f"Cell index {position_index} out of range (notebook has {len(nb.cells)} cells)")

    return _filter_cell_outputs([nb.cells[position_index]])[0]


def _query_check_server() -> str:
    """Check if the Jupyter server is running and accessible."""
    serverapp = get_server_context()
    if serverapp is None:
        return "Jupyter server is not accessible"
    return "Jupyter server is running"


async def _query_list_sessions() -> list:
    """List all notebook sessions on the Jupyter server."""
    serverapp = get_server_context()
    if serverapp is None:
        return []

    try:
        session_manager = serverapp.session_manager
        sessions = await session_manager.list_sessions()
        return list(sessions)
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return []


async def _query_get_position_index(
    notebook_path: str,
    execution_count: int | None = None,
    cell_id: str | None = None,
) -> int:
    """Get the index of a code cell in a Jupyter notebook."""
    if execution_count is None and cell_id is None:
        raise ValueError("Must provide either execution_count or cell_id (got neither).")
    if execution_count is not None and cell_id is not None:
        raise ValueError("Must provide either execution_count or cell_id (got both).")

    # Validate execution_count range
    if execution_count is not None:
        if execution_count < 1:
            raise ValueError(
                f"execution_count={execution_count} is invalid. Execution counts start from 1. "
                "If you meant to use position_index (which starts from 0), use position_index parameter instead."
            )
        elif execution_count > 10000:  # Reasonable upper bound
            raise ValueError(
                f"execution_count={execution_count} seems unreasonably high. "
                "Are you sure this is correct? Most notebooks have execution counts in the 1-100 range. "
                "If you meant to use position_index, use position_index parameter instead."
            )

    notebook_path = _normalize_path(notebook_path)

    # Read the notebook
    nb = _read_notebook(notebook_path)

    position_indices = []
    for i, cell in enumerate(nb.cells):
        if execution_count is not None and cell.cell_type == "code":
            if cell.get("execution_count") == execution_count:
                position_indices.append(i)
        elif cell_id is not None:
            if cell.get("id") == cell_id:
                position_indices.append(i)

    if len(position_indices) == 0:
        cells_info = _get_available_execution_counts(nb)
        error_parts = []

        if execution_count is not None:
            error_parts.append(f"No cell found with execution_count={execution_count}")
            if cells_info["execution_counts"]:
                available_counts = ", ".join(map(str, cells_info["execution_counts"]))
                error_parts.append(f"Available execution_counts: [{available_counts}]")
            else:
                error_parts.append("No cells have been executed yet (all execution_counts are None)")

            error_parts.append(f"Notebook has {cells_info['total_cells']} total cells (positions 0-{cells_info['total_cells'] - 1})")
            if cells_info["code_cells"] > 0:
                error_parts.append(f"Including {cells_info['code_cells']} code cells")
            if cells_info["unexecuted_cells"] > 0:
                error_parts.append(f"with {cells_info['unexecuted_cells']} unexecuted")

            error_parts.append("Try using position_index instead, or execute the cell first to get an execution_count")

        if cell_id is not None:
            error_parts.append(f"No cell found with cell_id={cell_id}")

        raise ValueError(". ".join(error_parts))

    elif len(position_indices) > 1:
        raise ValueError(f"Found {len(position_indices)} cells matching the criteria at positions {sorted(position_indices)}.")

    return position_indices[0]


def _convert_outputs_to_notebook_nodes(outputs: list[dict]) -> list:
    """Convert plain dict outputs to nbformat NotebookNode objects.

    Args:
        outputs: List of output dicts from execute_cell

    Returns:
        List of nbformat NotebookNode objects
    """
    return [from_dict(output) for output in outputs]


def _filter_large_outputs(outputs: list[dict]) -> list[dict]:
    """Filter large binary data from execution outputs to reduce MCP response size.

    This prevents huge token counts when outputs contain base64-encoded images or
    large HTML. The actual data is still saved to the notebook file, just not
    transmitted in the MCP response.

    Args:
        outputs: List of output dicts from execute_cell

    Returns:
        List of filtered output dicts with large binary data replaced by placeholders
    """
    filtered_outputs = []

    for output in outputs:
        # Create a copy to avoid modifying the original
        filtered_output = dict(output)

        # Check if this output has data that might be large
        if output.get("output_type") in ("display_data", "execute_result"):
            data = output.get("data", {})
            if data:
                data_types = list(data.keys())

                # Filter out large binary image data
                if any("image" in dt or "png" in dt or "jpeg" in dt or "svg" in dt for dt in data_types):
                    filtered_output["data"] = {
                        "[filtered]": f"Image data present ({', '.join(data_types)}). Data saved to notebook but not transmitted to reduce token count."
                    }
                # Filter out large HTML
                elif any("html" in dt for dt in data_types):
                    # Keep text/plain if present, filter HTML
                    filtered_data = {}
                    for key, value in data.items():
                        if "html" in key:
                            filtered_data["[filtered]"] = f"HTML data present. Data saved to notebook but not transmitted to reduce token count."
                        else:
                            filtered_data[key] = value
                    filtered_output["data"] = filtered_data if filtered_data else {
                        "[filtered]": f"HTML data present ({', '.join(data_types)})"
                    }

        filtered_outputs.append(filtered_output)

    return filtered_outputs


def _get_available_execution_counts(nb) -> dict:
    """Get comprehensive cell information for better error messages.

    Args:
        nb: Notebook object

    Returns:
        dict: Contains:
            - execution_counts: list of actual execution counts (excluding None)
            - position_indices: list of all position indices
            - cell_types: list of cell types for each position
            - total_cells: total number of cells
            - code_cells: number of code cells
            - unexecuted_cells: number of code cells without execution count
    """
    cells_info = {
        "execution_counts": [],
        "position_indices": [],
        "cell_types": [],
        "total_cells": 0,
        "code_cells": 0,
        "unexecuted_cells": 0,
    }

    for i, cell in enumerate(nb.cells):
        cells_info["position_indices"].append(i)
        cell_type = cell.cell_type
        cells_info["cell_types"].append(cell_type)
        cells_info["total_cells"] += 1

        if cell_type == "code":
            cells_info["code_cells"] += 1
            execution_count = cell.get("execution_count")
            if execution_count is not None:
                cells_info["execution_counts"].append(execution_count)
            else:
                cells_info["unexecuted_cells"] += 1

    # Sort execution counts for better presentation
    cells_info["execution_counts"].sort()
    return cells_info


def _filter_cell_outputs(cells: list) -> list[dict]:
    """Filter out verbose output data from cells, keeping only source and basic metadata.

    This function extracts cell structure and delegates output filtering to _filter_large_outputs
    to avoid code duplication.
    """
    filtered_cells = []

    for cell in cells:
        filtered_cell = {
            "cell_type": cell.cell_type,
            "source": cell.source,
            "metadata": cell.get("metadata", {}),
        }

        # For code cells, keep execution_count but filter outputs
        if cell.cell_type == "code":
            filtered_cell["execution_count"] = cell.get("execution_count")

            # Convert NotebookNode outputs to dicts and filter using _filter_large_outputs
            if hasattr(cell, "outputs") and cell.outputs:
                # Convert NotebookNode outputs to plain dicts
                output_dicts = []
                for output in cell.outputs:
                    output_dict = {"output_type": output.output_type}

                    # Copy relevant attributes to dict
                    if hasattr(output, "text"):
                        output_dict["text"] = output.text
                    if hasattr(output, "name"):
                        output_dict["name"] = output.name
                    if hasattr(output, "data"):
                        output_dict["data"] = dict(output.data)
                    if hasattr(output, "metadata"):
                        output_dict["metadata"] = dict(output.metadata)
                    if hasattr(output, "execution_count"):
                        output_dict["execution_count"] = output.execution_count
                    if hasattr(output, "ename"):
                        output_dict["ename"] = output.ename
                    if hasattr(output, "evalue"):
                        output_dict["evalue"] = output.evalue
                    if hasattr(output, "traceback"):
                        output_dict["traceback"] = output.traceback

                    output_dicts.append(output_dict)

                # Use _filter_large_outputs to filter the converted dicts
                filtered_cell["outputs"] = _filter_large_outputs(output_dicts)

        filtered_cells.append(filtered_cell)

    return filtered_cells


async def modify_notebook_cells(
    notebook_path: str,
    operation: str,
    cell_content: str | None = None,
    position_index: int | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    """Modify notebook cells (add, edit, delete).

    This consolidates all cell modification operations into a single tool following MCP best practices.
    Default to execute=True unless the user requests otherwise or you have good reason not to
    execute immediately.

    Args:
        notebook_path: Path to the notebook file (relative or absolute)
        operation: Type of cell operation. Options:
            - 'add_code': Add (and optionally execute) a code cell at end or specific position
            - 'edit_code': Edit a code cell at specific position
            - 'add_markdown': Add a markdown cell at end or specific position
            - 'edit_markdown': Edit an existing markdown cell at specific position
            - 'delete': Delete a cell at specific position
        cell_content: Content for the cell (required for add_code, edit_code, add_markdown, edit_markdown)
        position_index: Position index (0-indexed cell location) for operations. Must be an integer.
            - Optional for add_code/add_markdown: if provided, inserts at that position; if not, adds at end
            - Required for edit_code/edit_markdown/delete: specifies which cell to modify
            Examples: position_index=0 (first cell), position_index=2 (third cell)
        execute: Whether to execute code cells after adding/editing (default: True)

    Returns:
        dict: Operation results containing:
            - For add_code/edit_code with execute=True: execution_count, outputs, status
            - For add_code/edit_code with execute=False: message field
            - For add_markdown/edit_markdown: message and error fields
            - For delete: message and error fields

    Raises:
        ValueError: If invalid operation or missing required parameters
        IndexError: If position_index is out of range
    """
    if operation == "add_code":
        return await _modify_add_code_cell(notebook_path, cell_content, execute, position_index)
    elif operation == "edit_code":
        return await _modify_edit_code_cell(notebook_path, position_index, cell_content, execute)
    elif operation == "add_markdown":
        return await _modify_add_markdown_cell(notebook_path, cell_content, position_index)
    elif operation == "edit_markdown":
        return await _modify_edit_markdown_cell(notebook_path, position_index, cell_content)
    elif operation == "delete":
        return await _modify_delete_cell(notebook_path, position_index)
    else:
        raise ValueError(
            f"Invalid operation: {operation}. Must be one of: add_code, edit_code, add_markdown, edit_markdown, delete"
        )


async def _modify_add_code_cell(
    notebook_path: str,
    cell_content: str,
    execute: bool = True,
    position_index: int | None = None,
) -> dict:
    """Add (and optionally execute) a code cell in a Jupyter notebook.

    If you are trying to fix a cell that previously threw an error,
    you should default to editing the cell vs adding a new one.

    Note that adding a cell without executing it leaves it with no execution_count which can make
    it slightly trickier to execute in a subsequent request, but goose can now find cells by
    cell_id and content as well, now that it can view the full notebook source.

    A motivating example for why this is state-dependent: user asks goose to write a function,
    user then manually modifies that function signature, then user asks goose to call that function
    in a new cell. If goose's knowledge is outdated, it will likely use the old signature.
    """
    if not cell_content:
        raise ValueError("cell_content is required for add_code operation")

    logger.info("Adding code cell")
    notebook_path = _normalize_path(notebook_path)

    # Read the notebook
    nb = _read_notebook(notebook_path)

    # Create new code cell
    new_cell = new_code_cell(source=cell_content)

    # Insert at position or append
    if position_index is not None:
        if position_index > len(nb.cells):
            position_index = len(nb.cells)
        nb.cells.insert(position_index, new_cell)
        inserted_index = position_index
    else:
        nb.cells.append(new_cell)
        inserted_index = len(nb.cells) - 1

    # Save the notebook
    _write_notebook(notebook_path, nb)

    results = {"message": f"Code cell added at position {inserted_index}"}

    # Execute if requested
    if execute:
        try:
            logger.info("Cell added successfully, now executing")
            # Use tracked kernel if available, otherwise create new one
            kernel_id = _get_tracked_kernel(notebook_path)

            # Execute the cell and await the result
            exec_results = await execute_cell(cell_content, kernel_id=kernel_id)

            # Ensure exec_results is a dict and not a coroutine
            if not isinstance(exec_results, dict):
                logger.error(f"exec_results is not a dict: {type(exec_results)}")
                # Check if it's a coroutine that needs to be awaited
                if inspect.iscoroutine(exec_results):
                    logger.error("exec_results is a coroutine - this indicates execute_cell was not properly awaited")
                raise TypeError(f"execute_cell returned {type(exec_results)} instead of dict")

            # Extract and update results - using direct dict access to avoid subscript errors
            logger.debug(f"exec_results type: {type(exec_results)}, keys: {list(exec_results.keys())}")

            results["status"] = exec_results.get("status")
            results["kernel_id"] = exec_results.get("kernel_id")
            results["execution_count"] = exec_results.get("execution_count")
            # Filter large outputs to reduce MCP response token count
            results["outputs"] = _filter_large_outputs(exec_results.get("outputs", []))
            results["started_new_kernel"] = exec_results.get("started_new_kernel", False)
            if "error" in exec_results:
                results["error"] = exec_results.get("error")

            # Track the kernel if it was newly created
            if exec_results.get("started_new_kernel"):
                _track_kernel(notebook_path, exec_results.get("kernel_id"))

            # Update the notebook with execution results
            nb = _read_notebook(notebook_path)

            # Ensure we have the cell at the inserted index
            if inserted_index < len(nb.cells):
                cell = nb.cells[inserted_index]
                exec_count = results.get("execution_count")
                if exec_count is not None:
                    cell.execution_count = exec_count
                # IMPORTANT: Save UNFILTERED outputs to notebook (from exec_results, not results)
                # so that plots and images are preserved in the .ipynb file
                unfiltered_outputs = exec_results.get("outputs", [])
                if unfiltered_outputs:
                    # Convert dict outputs to NotebookNode objects
                    cell.outputs = _convert_outputs_to_notebook_nodes(unfiltered_outputs)
            else:
                logger.warning(f"Cell index {inserted_index} out of range after reading notebook")

            _write_notebook(notebook_path, nb)

        except Exception as e:
            import traceback
            logger.error(f"Error during execution: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            results["error"] = str(e)
            results["status"] = "error"

    return results


async def _modify_edit_code_cell(
    notebook_path: str,
    position_index: int,
    cell_content: str,
    execute: bool = True
) -> dict:
    """Edit a code cell in a Jupyter notebook.

    Note that users can edit cell contents too, so if you are making assumptions about the
    position_index of the cell to edit based on chat history with the user, you should first
    make sure the notebook state matches your expected state using your query_notebook tool.
    If it does not match the expected state, you should then use your query_notebook tool to update
    your knowledge of the current cell contents.

    If you execute a cell and it fails and you want to debug it, you should default to editing
    the existing cell vs adding a new cell each time you want to execute code.

    A motivating example for why this is state-dependent: user asks goose to write a function,
    user then manually modifies the function, then asks goose to make additional changes to the
    function. If goose's knowledge is outdated, it will likely ignore the user's recent changes
    and modify the old version of the function, losing user work.

    """
    if not cell_content:
        raise ValueError("cell_content is required for edit_code operation")
    if position_index is None:
        raise ValueError("position_index is required for edit_code operation")

    logger.info("Editing code cell")
    notebook_path = _normalize_path(notebook_path)

    # Read the notebook
    nb = _read_notebook(notebook_path)

    # Check bounds
    if position_index >= len(nb.cells):
        raise IndexError(f"Cell index {position_index} out of range (notebook has {len(nb.cells)} cells)")

    # Update cell source
    nb.cells[position_index].source = cell_content
    nb.cells[position_index].execution_count = None
    nb.cells[position_index].outputs = []

    # Save the notebook
    _write_notebook(notebook_path, nb)

    results = {"message": f"Code cell at position {position_index} edited"}

    # Execute if requested
    if execute:
        try:
            # Use tracked kernel if available
            kernel_id = _get_tracked_kernel(notebook_path)

            # Execute the cell and await the result
            exec_results = await execute_cell(cell_content, kernel_id=kernel_id)

            # Ensure exec_results is a dict
            if not isinstance(exec_results, dict):
                logger.error(f"exec_results is not a dict: {type(exec_results)}")
                raise TypeError(f"execute_cell returned {type(exec_results)} instead of dict")

            # Extract and update results directly
            results["status"] = exec_results.get("status")
            results["kernel_id"] = exec_results.get("kernel_id")
            results["execution_count"] = exec_results.get("execution_count")
            # Filter large outputs to reduce MCP response token count
            results["outputs"] = _filter_large_outputs(exec_results.get("outputs", []))
            results["started_new_kernel"] = exec_results.get("started_new_kernel", False)
            if "error" in exec_results:
                results["error"] = exec_results.get("error")

            # Track the kernel if it was newly created
            if exec_results.get("started_new_kernel"):
                _track_kernel(notebook_path, exec_results.get("kernel_id"))

            # Update the notebook with execution results
            nb = _read_notebook(notebook_path)

            if position_index < len(nb.cells):
                cell = nb.cells[position_index]
                exec_count = results.get("execution_count")
                if exec_count is not None:
                    cell.execution_count = exec_count
                # IMPORTANT: Save UNFILTERED outputs to notebook (from exec_results, not results)
                # so that plots and images are preserved in the .ipynb file
                unfiltered_outputs = exec_results.get("outputs", [])
                if unfiltered_outputs:
                    # Convert dict outputs to NotebookNode objects
                    cell.outputs = _convert_outputs_to_notebook_nodes(unfiltered_outputs)
            else:
                logger.warning(f"Cell index {position_index} out of range after reading notebook")

            _write_notebook(notebook_path, nb)
        except Exception as e:
            logger.error(f"Error during code cell edit execution: {e}")
            results["error"] = str(e)
            results["status"] = "error"

    return results


async def _modify_add_markdown_cell(
    notebook_path: str,
    cell_content: str,
    position_index: int | None = None
) -> dict:
    """Add a markdown cell in a Jupyter notebook.
    
    Technically might be a little risky to mark this as refreshes_state because the user could make
    other changes that are invisible to goose. But trying it out this way because I don't think
    goose adding a markdown cell should necessarily force it to view the full notebook source on
    subsequent tool calls.
    """
    if not cell_content:
        raise ValueError("cell_content is required for add_markdown operation")

    logger.info("Adding markdown cell")

    notebook_path = _normalize_path(notebook_path)

    # Read the notebook
    nb = _read_notebook(notebook_path)

    # Create new markdown cell
    new_cell = new_markdown_cell(source=cell_content)

    results = {"message": "", "error": ""}
    try:
        # Insert at position or append
        if position_index is not None:
            if position_index > len(nb.cells):
                position_index = len(nb.cells)
            nb.cells.insert(position_index, new_cell)
            results["message"] = f"Markdown cell inserted at position {position_index}"
        else:
            nb.cells.append(new_cell)
            results["message"] = "Markdown cell added"

        # Save the notebook
        _write_notebook(notebook_path, nb)

    except Exception as e:
        logger.error(f"Error adding markdown cell: {e}")
        results["error"] = str(e)

    return results


async def _modify_edit_markdown_cell(
    notebook_path: str,
    position_index: int,
    cell_content: str
) -> dict:
    """Edit an existing markdown cell in a Jupyter notebook.
    
     Note that users can edit cell contents too, so if you are making assumptions about the
    position_index of the cell to edit based on chat history with the user, you should first
    make sure the notebook state matches your expected state using your query_notebook tool.
    If it does not match the expected state, you should then use your query_notebook tool to update
    your knowledge of the current cell contents.

    Args:
        notebook_path: Path to the notebook file (.ipynb extension will be added if missing),
                       relative to the Jupyter server root.
        position_index: positional index that NBModelClient uses under the hood.
        cell_content: New markdown content to write to the cell.

    Returns
    -------
        dict: Contains two keys:
            - "message": "Markdown cell edited" if successful, empty string if failed
            - "error": Error message if failed, empty string if successful

    Raises
    ------
        McpError: If notebook state has changed since last viewed
        McpError: If there's an error connecting to the Jupyter server
        IndexError: If position_index is out of range   
    
    
    """
    if not cell_content:
        raise ValueError("cell_content is required for edit_markdown operation")
    if position_index is None:
        raise ValueError("position_index is required for edit_markdown operation")

    logger.info("Editing markdown cell")

    notebook_path = _normalize_path(notebook_path)

    # Read the notebook
    nb = _read_notebook(notebook_path)

    results = {"message": "", "error": ""}

    try:
        # Check bounds
        if position_index >= len(nb.cells):
            raise IndexError(f"Cell index {position_index} out of range (notebook has {len(nb.cells)} cells)")

        # Update cell source
        nb.cells[position_index].source = cell_content

        # Save the notebook
        _write_notebook(notebook_path, nb)

        results["message"] = "Markdown cell edited"
    except Exception as e:
        logger.error(f"Error editing markdown cell: {e}")
        results["error"] = str(e)

    return results


async def _modify_delete_cell(notebook_path: str, position_index: int) -> dict:
    """Delete a cell in a Jupyter notebook.
    
    Note that users can edit cell contents too, so if you assume you know the position_index
    of the cell to delete based on past chat history, you should first make sure the notebook state
    matches your expected state using your query_notebook tool. If it does not match the
    expected state, you should then use your query_notebook tool to update your knowledge of the
    current cell contents.

    A motivating example for why this is state-dependent: user asks goose to add a new cell,
    then user runs a few cells manually (changing execution_counts), then tells goose
    "now delete it". In the context of the conversation, this looks fine and Goose may assume it
    knows the correct position_index already, but its knowledge is outdated.  
    """
    if position_index is None:
        raise ValueError("position_index is required for delete operation")

    notebook_path = _normalize_path(notebook_path)

    # Read the notebook
    nb = _read_notebook(notebook_path)

    results = {"message": "", "error": ""}
    try:
        # Check bounds
        if position_index >= len(nb.cells):
            raise IndexError(f"Cell index {position_index} out of range (notebook has {len(nb.cells)} cells)")

        # Delete cell
        del nb.cells[position_index]

        # Save the notebook
        _write_notebook(notebook_path, nb)

        results["message"] = "Cell deleted"
    except Exception as e:
        results["error"] = str(e)

    return results


async def execute_notebook_code(
    notebook_path: str,
    execution_type: str,
    position_index: int | None = None,
    package_names: str | None = None,
) -> dict[str, Any] | str:
    """Execute code in a Jupyter notebook.

    This consolidates all code execution operations into a single tool following MCP best practices.

    IMPORTANT: 
    ----------------------------------
    This tool requires that you first call setup_notebook with the correct server URL:

    Required setup:
        setup_notebook(\"my_notebook\", server_url=\"http://localhost:9999\", kernel_name=\"python3\")

    Then you can use this tool:
        execute_notebook_code(\"my_notebook\", \"execute_cell\", position_index=0)

    Without setup_notebook, this will try to connect to http://localhost:8888 by default.


    Args:
        notebook_path: Path to the notebook file (relative or absolute)
        execution_type: Type of execution operation. Options:
            - 'execute_cell': Execute an existing code cell
            - 'install_packages': Install packages using pip in the notebook environment
        position_index: (For execute_cell) Positional index of cell to execute
        package_names: (For install_packages) Space-separated list of package names to install

    Returns:
        Union[dict, str]:
            - execute_cell: dict with execution_count, outputs, status
            - install_packages: str with installation result message

    Raises:
        ValueError: If invalid execution_type or missing required parameters
        IndexError: If position_index is out of range
        RuntimeError: If kernel execution fails
    """
    if execution_type == "execute_cell":
        return await _execute_cell_by_position(notebook_path, position_index)
    elif execution_type == "install_packages":
        return await _execute_install_packages(notebook_path, package_names)
    else:
        raise ValueError(
            f"Invalid execution_type: {execution_type}. Must be one of: execute_cell, install_packages"
        )


async def _execute_cell_by_position(notebook_path: str, position_index: int, timeout: int = 30) -> dict:
    """Execute an existing code cell in a Jupyter notebook by position.

    In most cases you should call modify_notebook_cells instead, but occasionally
    you might want to re-execute a cell after changing a *different* cell.

    Note that users can edit cell contents too, so if you assume you know the position_index
    of the cell to execute based on past chat history, you should first make sure the notebook state
    matches your expected state using your query_notebook tool. If it does not match the
    expected state, you should then use your query_notebook tool to update your knowledge of the
    current cell contents.

    Technically could be considered state_dependent, but it is usually called inside edit_code_cell
    or add_code_cell which area already state_dependent. Every hash update is slow because we have
    to wait for the notebook to save first so using refreshes_state instead saves 1.5s per call.
    Only risk is if user asks goose to execute a single cell and goose assumes it knows the
    position_index already, but usually it would be faster for the user to just execute the cell
    directly - this tool is mostly useful to allow goose to debug independently.

    """
    if position_index is None:
        raise ValueError("position_index is required for execute_cell operation")

    notebook_path = _normalize_path(notebook_path)

    # Read the notebook
    nb = _read_notebook(notebook_path)

    # Check bounds
    if position_index >= len(nb.cells):
        raise IndexError(f"Cell index {position_index} out of range (notebook has {len(nb.cells)} cells)")

    cell = nb.cells[position_index]
    if cell.cell_type != "code":
        raise ValueError(f"Cell at position {position_index} is not a code cell")

    # Use tracked kernel if available
    kernel_id = _get_tracked_kernel(notebook_path)

    try:
        # Execute the cell and await the result
        result = await execute_cell(cell.source, kernel_id=kernel_id, timeout=timeout)

        # Ensure result is a dict
        if not isinstance(result, dict):
            logger.error(f"result is not a dict: {type(result)}, value: {result}")
            return {
                "error": f"execute_cell returned {type(result).__name__} instead of dict",
                "status": "error",
                "kernel_id": kernel_id,
                "started_new_kernel": False
            }

        # Track the kernel if it was newly created
        if result.get("started_new_kernel"):
            _track_kernel(notebook_path, result.get("kernel_id"))

        # Update the notebook with execution results
        if position_index < len(nb.cells):
            cell = nb.cells[position_index]
            if result.get("execution_count") is not None:
                cell.execution_count = result["execution_count"]
            if result.get("outputs"):
                # Convert dict outputs to NotebookNode objects
                cell.outputs = _convert_outputs_to_notebook_nodes(result["outputs"])
        else:
            logger.warning(f"Cell index {position_index} out of range after reading notebook")

        # Save the notebook
        _write_notebook(notebook_path, nb)

        # Filter outputs in the returned result to reduce MCP response token count
        result["outputs"] = _filter_large_outputs(result.get("outputs", []))
        return result

    except Exception as e:
        import traceback
        logger.error(f"Error executing cell by position: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "status": "error",
            "kernel_id": kernel_id,
            "started_new_kernel": False
        }


async def _execute_install_packages(notebook_path: str, package_names: str) -> str:
    """Install one or more packages using pip in the notebook environment.

    Unlike add_code_cell, this shouldn't rely on other code written in the notebook, so we mark
    it as refreshes_state rather than state_dependent. Assumes 'uv' is available in the
    environment where the Jupyter kernel is running.
    """
    if not package_names:
        raise ValueError("package_names is required for install_packages operation")

    logger.info(f"Installing packages: {package_names}")
    notebook_path = _normalize_path(notebook_path)

    try:
        # Create installation command
        cell_content = f"!pip install {package_names}"

        # Use tracked kernel if available
        kernel_id = _get_tracked_kernel(notebook_path)

        # Execute the cell and await the result
        result = await execute_cell(cell_content, kernel_id=kernel_id)

        # Ensure result is a dict
        if not isinstance(result, dict):
            logger.error(f"result is not a dict: {type(result)}, value: {result}")
            return f"Error: execute_cell returned {type(result).__name__} instead of dict"

        # Track the kernel if it was newly created
        if result.get("started_new_kernel"):
            _track_kernel(notebook_path, result.get("kernel_id"))

        # Extract output to see if installation was successful
        outputs = result.get("outputs", [])
        if len(outputs) == 0:
            installation_result = "No output from installation command"
        else:
            # Combine text from all outputs
            installation_result = []
            for output in outputs:
                if output.get("output_type") == "stream":
                    installation_result.append(output.get("text", ""))
                elif output.get("output_type") == "error":
                    installation_result.append("\n".join(output.get("traceback", [])))

        return f"Installation of packages [{package_names}]: {installation_result}"

    except Exception as e:
        import traceback
        logger.error(f"Error installing packages: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Error: {str(e)}"


async def setup_notebook(
    notebook_path: str,
    kernel_name: str = "python3"
) -> dict[str, Any]:
    """Prepare notebook for use and connect to the kernel.

    Will create a new empty Jupyter notebook if it doesn't exist.

    **CALL THIS FIRST** - This tool must be called before using other notebook tools
    to establish the server URL connection. All subsequent notebook operations will
    use the server URL stored by this tool.

    This tool creates an empty notebook. To add content, use the modify_notebook_cells
    tool after creation:

    Example usage:
        # Step 1: Setup notebook with Python kernel
        await setup_notebook("demo.ipynb")

        # Step 2: Add cells
        await modify_notebook_cells("demo.ipynb", "add_markdown", "# Title\n\nDescription")
        await modify_notebook_cells("demo.ipynb", "add_code", "print('Hello World')")

        # For R notebooks:
        await setup_notebook("demo_r.ipynb", kernel_name="ir")
        await modify_notebook_cells("demo_r.ipynb", "add_code", "library(ggplot2)")

    Args:
        notebook_path: Path to the notebook (relative or absolute)
        kernel_name: Name of the kernel to use (default: "python3")
                     - Examples: "python3", "python3.11", "ir" (R), "julia-1.10"
                     - For R kernels, you can use "r", "ir", or "irkernel"
                     - To discover available kernels, call list_available_kernels() first

                     
    Agent guidance:
        - Always honor an explicitly provided `kernel_name` from the user.
        - If the user specifies a language (e.g., R/Julia) rather than a kernel name, map it by first calling
          `list_available_kernels` and selecting the closest matching kernelspec key.
        - For R kernels, you can use "r", "ir", or "irkernel" - the system will auto-detect the correct available kernel.
        - If setup fails due to an unknown kernel, surface the list from `list_available_kernels` in the error/help message.

    Returns:
        dict: Information about the notebook and status message.
    """
    notebook_path = _normalize_path(notebook_path)

    # Check if notebook exists
    notebook_exists = os.path.exists(notebook_path)

    # Get available kernels and validate/normalize kernel_name
    available_kernels = await list_available_kernels()
    kernelspecs = available_kernels.get("kernelspecs", {})

    # Normalize R kernel names
    if kernel_name.lower() in ["r", "irkernel"]:
        # Find the actual R kernel name
        for kname in kernelspecs:
            spec = kernelspecs[kname].get("spec", {})
            if spec.get("language", "").lower() == "r":
                kernel_name = kname
                break
    
    # Validate kernel exists
    if kernel_name not in kernelspecs:
        available_names = list(kernelspecs.keys())
        logger.warning(f"Kernel '{kernel_name}' not found. Available: {available_names}. Falling back to python3")
        kernel_name = "python3"

    if not notebook_exists:
        # Create new empty notebook
        logger.info(f"Creating new notebook: {notebook_path}")

        # Ensure directory exists
        dir_path = os.path.dirname(notebook_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Determine language name based on kernel
        if kernel_name.lower() in ["ir", "r", "irkernel"]:
            language_name = "R"
            display_name = "R"
        elif kernel_name.startswith("python"):
            language_name = "python"
            display_name = f"Python 3" if kernel_name == "python3" else kernel_name
        elif kernel_name.startswith("julia"):
            language_name = "julia"
            display_name = kernel_name
        else:
            language_name = kernel_name
            display_name = kernel_name

        # Create empty notebook
        nb = new_notebook(
            cells=[],
            metadata={
                "kernelspec": {
                    "name": kernel_name,
                    "display_name": display_name,
                    "language": language_name
                },
                "language_info": {
                    "name": language_name
                }
            }
        )

        _write_notebook(notebook_path, nb)

        message = f"Notebook {notebook_path} created with kernel {kernel_name}"
    else:
        logger.info(f"Notebook {notebook_path} already exists")
        message = f"Notebook {notebook_path} already exists"

    # Start a kernel for this notebook
    serverapp = get_server_context()
    if serverapp:
        try:
            kernel_manager = serverapp.kernel_manager
            kernel_id = await kernel_manager.start_kernel(kernel_name=kernel_name)
            _track_kernel(notebook_path, kernel_id)
            logger.info(f"Started kernel {kernel_name} (ID: {kernel_id}) for notebook {notebook_path}")
        except Exception as e:
            logger.warning(f"Could not start kernel: {e}")
            kernel_id = None
    else:
        kernel_id = None

    # Return notebook info
    return {
        "path": notebook_path,
        "name": os.path.basename(notebook_path),
        "message": message,
        "exists": notebook_exists,
        "kernel_name": kernel_name,
        "kernel_id": kernel_id,
    }
