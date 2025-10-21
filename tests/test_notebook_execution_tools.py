"""Test notebook execution tools functionality."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from jupyter_server_mcp import notebook_execution_tools
from jupyter_server_mcp.notebook_execution_tools import (
    _convert_outputs_to_notebook_nodes,
    _filter_cell_outputs,
    _get_available_execution_counts,
    _normalize_path,
    _read_notebook,
    _write_notebook,
    execute_cell,
    get_server_context,
    list_available_kernels,
    list_running_kernels,
    modify_notebook_cells,
    query_notebook,
    set_server_context,
    setup_notebook,
    shutdown_notebook,
    switch_notebook_kernel,
    _kernel_tracker
)


class MockKernelManager:
    """Mock kernel manager for testing."""

    def __init__(self):
        self.kernels = {}
        self._next_id = 0

    async def start_kernel(self, kernel_name="python3"):
        """Mock start kernel."""
        kernel_id = f"kernel-{self._next_id}"
        self._next_id += 1
        self.kernels[kernel_id] = MockKernel(kernel_id, kernel_name)
        return kernel_id

    async def shutdown_kernel(self, kernel_id):
        """Mock shutdown kernel."""
        if kernel_id in self.kernels:
            del self.kernels[kernel_id]

    def get_kernel(self, kernel_id):
        """Mock get kernel."""
        return self.kernels.get(kernel_id)

    def list_kernel_ids(self):
        """Mock list kernel IDs."""
        return list(self.kernels.keys())


class MockKernel:
    """Mock kernel for testing."""

    def __init__(self, kernel_id, kernel_name="python3"):
        from datetime import datetime
        self.kernel_id = kernel_id
        self.kernel_name = kernel_name
        self.last_activity = datetime.now()
        self.execution_state = "idle"
        self.connection_count = 0
        self._client = MockKernelClient()  # Persistent client to maintain execution_count

    def client(self):
        """Return a mock client."""
        return self._client


class MockKernelClient:
    """Mock kernel client for testing."""

    def __init__(self):
        self.channels_started = False
        self.execution_count = 1
        self._current_msg_id = None
        self._shell_messages = []

    def start_channels(self):
        """Mock start channels."""
        self.channels_started = True

    def stop_channels(self):
        """Mock stop channels."""
        self.channels_started = False

    async def wait_for_ready(self):
        """Mock wait for ready."""
        await asyncio.sleep(0.01)

    def execute(self, code):
        """Mock execute - prepares shell reply with execution_count."""
        msg_id = f"msg-id-{self.execution_count}"
        self._current_msg_id = msg_id

        # Queue the execute_reply for shell channel
        # This always contains execution_count per Jupyter protocol
        self._shell_messages.append({
            "header": {
                "msg_type": "execute_reply",
                "msg_id": f"reply-{self.execution_count}"
            },
            "parent_header": {"msg_id": msg_id},
            "content": {
                "status": "ok",
                "execution_count": self.execution_count
            }
        })

        exec_count = self.execution_count
        self.execution_count += 1
        return msg_id

    async def get_shell_msg(self, timeout=None):
        """Mock get shell message - returns execute_reply."""
        await asyncio.sleep(0.01)  # Simulate async

        if self._shell_messages:
            return self._shell_messages.pop(0)

        # No messages available
        raise asyncio.TimeoutError("No shell messages available")

    async def get_iopub_msg(self, timeout=None):
        """Mock get iopub message."""
        await asyncio.sleep(0.01)  # Simulate async

        # Simulate execution complete
        return {
            "header": {"msg_type": "status", "msg_id": "msg-id-456"},
            "parent_header": {"msg_id": self._current_msg_id or "msg-id-123"},
            "content": {"execution_state": "idle"},
        }


class MockServerApp:
    """Mock server app for testing."""

    def __init__(self):
        self.kernel_manager = MockKernelManager()
        self.session_manager = MockSessionManager()


class MockSessionManager:
    """Mock session manager."""

    async def list_sessions(self):
        """Mock list sessions."""
        return []


@pytest.fixture
def temp_notebook(tmp_path):
    """Create a temporary notebook for testing."""
    nb_path = tmp_path / "test.ipynb"
    nb = new_notebook(cells=[])
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)
    return str(nb_path)


@pytest.fixture
def mock_serverapp():
    """Create a mock server app."""
    return MockServerApp()


@pytest.fixture(autouse=True)
def reset_context():
    """Reset global context before each test."""
    notebook_execution_tools._server_context = {"serverapp": None}
    notebook_execution_tools._notebook_kernels = {}
    yield
    notebook_execution_tools._server_context = {"serverapp": None}
    notebook_execution_tools._notebook_kernels = {}


class TestHelperFunctions:
    """Test helper functions."""

    def test_normalize_path(self):
        """Test path normalization."""
        path = "test.ipynb"
        normalized = _normalize_path(path)
        assert os.path.isabs(normalized)
        assert normalized.endswith("test.ipynb")

    def test_read_write_notebook(self, temp_notebook):
        """Test reading and writing notebooks."""
        # Read the notebook
        nb = _read_notebook(temp_notebook)
        assert nb is not None
        assert len(nb.cells) == 0

        # Add a cell
        nb.cells.append(new_code_cell("print('hello')"))

        # Write the notebook
        _write_notebook(temp_notebook, nb)

        # Read it again
        nb2 = _read_notebook(temp_notebook)
        assert len(nb2.cells) == 1
        assert nb2.cells[0].source == "print('hello')"

    def test_track_kernel(self):
        """Test kernel tracking."""
        path = "/test/path.ipynb"
        kernel_id = "kernel-123"

        _kernel_tracker.track(path, kernel_id)
        assert _kernel_tracker.get(path) == kernel_id

    def test_get_tracked_kernel_none(self):
        """Test getting non-existent kernel."""
        result = _kernel_tracker.get("/nonexistent/path.ipynb")
        assert result is None

    def test_convert_outputs_to_notebook_nodes(self):
        """Test converting outputs to notebook nodes."""
        outputs = [
            {"output_type": "stream", "name": "stdout", "text": "hello\n"},
            {"output_type": "execute_result", "data": {"text/plain": "42"}},
        ]
        nodes = _convert_outputs_to_notebook_nodes(outputs)
        assert len(nodes) == 2
        assert nodes[0]["output_type"] == "stream"

    def test_get_available_execution_counts(self, temp_notebook):
        """Test getting available execution counts."""
        nb = _read_notebook(temp_notebook)
        nb.cells.append(new_code_cell("x = 1"))
        nb.cells.append(new_code_cell("y = 2"))
        nb.cells[0].execution_count = 1
        nb.cells[1].execution_count = 2

        info = _get_available_execution_counts(nb)
        assert info["total_cells"] == 2
        assert info["code_cells"] == 2
        assert info["execution_counts"] == [1, 2]
        assert info["unexecuted_cells"] == 0

    def test_filter_cell_outputs(self, temp_notebook):
        """Test filtering cell outputs."""
        nb = _read_notebook(temp_notebook)
        cell = new_code_cell("print('test')")
        cell.execution_count = 1
        cell.outputs = [
            nbformat.v4.new_output("stream", name="stdout", text="test\n")
        ]
        nb.cells.append(cell)

        filtered = _filter_cell_outputs(nb.cells)
        assert len(filtered) == 1
        assert filtered[0]["cell_type"] == "code"
        assert filtered[0]["execution_count"] == 1


class TestServerContext:
    """Test server context management."""

    def test_set_get_server_context(self, mock_serverapp):
        """Test setting and getting server context."""
        set_server_context(mock_serverapp)
        assert get_server_context() == mock_serverapp

    def test_get_server_context_none(self):
        """Test getting server context when not set."""
        assert get_server_context() is None


class TestListFunctions:
    """Test list functions."""

    @pytest.mark.asyncio
    async def test_list_running_kernels_no_context(self):
        """Test listing kernels without server context."""
        result = await list_running_kernels()
        assert "error" in result
        assert result["kernels"] == []

    @pytest.mark.asyncio
    async def test_list_running_kernels(self, mock_serverapp):
        """Test listing running kernels."""
        set_server_context(mock_serverapp)
        kernel_id = await mock_serverapp.kernel_manager.start_kernel()

        result = await list_running_kernels()
        assert "kernels" in result
        assert len(result["kernels"]) == 1
        assert result["kernels"][0]["id"] == kernel_id

    @pytest.mark.asyncio
    async def test_list_available_kernels(self):
        """Test listing available kernel specs."""
        result = await list_available_kernels()
        assert "kernelspecs" in result
        # Should have at least python3
        assert isinstance(result["kernelspecs"], dict)


class TestExecuteCell:
    """Test execute_cell function."""

    @pytest.mark.asyncio
    async def test_execute_cell_no_context(self):
        """Test execute_cell without server context."""
        result = await execute_cell("print('test')")
        assert result["status"] == "error"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_cell_with_new_kernel(self, mock_serverapp):
        """Test executing cell with new kernel."""
        set_server_context(mock_serverapp)

        with patch.object(
            MockKernelClient, "get_iopub_msg"
        ) as mock_get_msg:
            # Simulate execution complete message
            mock_get_msg.return_value = {
                "header": {"msg_type": "status"},
                "parent_header": {"msg_id": "msg-id-123"},
                "content": {"execution_state": "idle"},
            }

            result = await execute_cell("print('hello')", kernel_name="python3")
            assert "kernel_id" in result
            assert result["started_new_kernel"] is True

    @pytest.mark.asyncio
    async def test_execute_cell_captures_execution_count_from_shell(self, mock_serverapp):
        """Test that execution_count is captured from shell channel execute_reply.

        This test verifies the fix for the issue where execution_count was not
        being updated in notebook files. The execute_reply on the shell channel
        ALWAYS contains execution_count, unlike execute_result on iopub which
        only exists when code produces a result.
        """
        set_server_context(mock_serverapp)

        # Execute code that only produces stream output (no execute_result)
        # This was the problematic case - execution_count would be null
        code = "print('Hello, World!')"

        result = await execute_cell(code, kernel_name="python3")

        # Verify execution_count was captured from shell reply
        assert result["status"] == "ok"
        assert result["execution_count"] == 1, "execution_count should be 1 from shell reply"
        assert result["execution_count"] is not None, "execution_count should never be None"

    @pytest.mark.asyncio
    async def test_execute_cell_increments_execution_count(self, mock_serverapp):
        """Test that execution_count increments with each execution."""
        set_server_context(mock_serverapp)

        # Start a kernel first
        kernel_id = await mock_serverapp.kernel_manager.start_kernel()

        # Execute multiple cells and verify count increments
        result1 = await execute_cell("x = 1", kernel_id=kernel_id)
        assert result1["execution_count"] == 1

        result2 = await execute_cell("x = 2", kernel_id=kernel_id)
        assert result2["execution_count"] == 2

        result3 = await execute_cell("x = 3", kernel_id=kernel_id)
        assert result3["execution_count"] == 3

    @pytest.mark.asyncio
    async def test_execute_cell_with_only_print_has_execution_count(self, mock_serverapp):
        """Test that code with only print statements gets execution_count.

        This is a regression test for the bug where execution_count was null
        for cells that only produced stream output (no execute_result).
        """
        set_server_context(mock_serverapp)

        # Code with only print (no return value)
        result = await execute_cell("print('test')", kernel_name="python3")

        # Should have execution_count even though there's no execute_result
        assert result["execution_count"] is not None
        assert result["execution_count"] >= 1


class TestSetupNotebook:
    """Test setup_notebook function."""

    @pytest.mark.asyncio
    async def test_setup_notebook_create_new(self, tmp_path, mock_serverapp):
        """Test creating a new notebook."""
        set_server_context(mock_serverapp)
        nb_path = tmp_path / "new_notebook.ipynb"

        result = await setup_notebook(str(nb_path), kernel_name="python3")
        assert result["exists"] is False
        assert result["kernel_name"] == "python3"
        assert os.path.exists(nb_path)

    @pytest.mark.asyncio
    async def test_setup_notebook_existing(self, temp_notebook, mock_serverapp):
        """Test setup with existing notebook."""
        set_server_context(mock_serverapp)
        result = await setup_notebook(temp_notebook, kernel_name="python3")
        assert result["exists"] is True


class TestSwitchNotebookKernel:
    """Test switch_notebook_kernel function."""

    @pytest.mark.asyncio
    async def test_switch_notebook_kernel(self, temp_notebook, mock_serverapp):
        """Test switching notebook kernel."""
        set_server_context(mock_serverapp)

        result = await switch_notebook_kernel(temp_notebook, "ir")
        assert result["kernel_name"] == "ir"
        assert "kernel_id" in result

        # Verify notebook metadata was updated
        nb = _read_notebook(temp_notebook)
        assert nb.metadata["kernelspec"]["name"] == "ir"


class TestShutdownNotebook:
    """Test shutdown_notebook function."""

    @pytest.mark.asyncio
    async def test_shutdown_notebook_no_kernel(self, temp_notebook):
        """Test shutdown notebook with no tracked kernel."""
        result = await shutdown_notebook(temp_notebook)
        assert "message" in result

    @pytest.mark.asyncio
    async def test_shutdown_notebook_with_kernel(self, temp_notebook, mock_serverapp):
        """Test shutdown notebook with tracked kernel."""
        set_server_context(mock_serverapp)
        kernel_id = await mock_serverapp.kernel_manager.start_kernel()
        _kernel_tracker.track(_normalize_path(temp_notebook), kernel_id)

        result = await shutdown_notebook(temp_notebook, shutdown_kernel=True)
        assert "kernel_id" in result


class TestModifyNotebookCells:
    """Test modify_notebook_cells function."""

    @pytest.mark.asyncio
    async def test_add_code_cell(self, temp_notebook, mock_serverapp):
        """Test adding a code cell."""
        set_server_context(mock_serverapp)

        with patch("jupyter_server_mcp.notebook_execution_tools.execute_cell") as mock_exec:
            mock_exec.return_value = {
                "status": "ok",
                "kernel_id": "kernel-123",
                "execution_count": 1,
                "outputs": [],
                "started_new_kernel": True,
            }

            result = await modify_notebook_cells(
                temp_notebook, "add_code", "print('hello')", execute=True
            )
            assert "status" in result or "message" in result

    @pytest.mark.asyncio
    async def test_add_markdown_cell(self, temp_notebook):
        """Test adding a markdown cell."""
        result = await modify_notebook_cells(
            temp_notebook, "add_markdown", "# Title"
        )
        assert "message" in result

        # Verify cell was added
        nb = _read_notebook(temp_notebook)
        assert len(nb.cells) == 1
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[0].source == "# Title"

    @pytest.mark.asyncio
    async def test_edit_code_cell(self, temp_notebook, mock_serverapp):
        """Test editing a code cell."""
        # First add a cell
        nb = _read_notebook(temp_notebook)
        nb.cells.append(new_code_cell("x = 1"))
        _write_notebook(temp_notebook, nb)

        set_server_context(mock_serverapp)

        with patch("jupyter_server_mcp.notebook_execution_tools.execute_cell") as mock_exec:
            mock_exec.return_value = {
                "status": "ok",
                "kernel_id": "kernel-123",
                "execution_count": 1,
                "outputs": [],
                "started_new_kernel": True,
            }

            result = await modify_notebook_cells(
                temp_notebook, "edit_code", "x = 2", position_index=0, execute=True
            )
            assert "status" in result or "message" in result

    @pytest.mark.asyncio
    async def test_delete_cell(self, temp_notebook):
        """Test deleting a cell."""
        # First add a cell
        nb = _read_notebook(temp_notebook)
        nb.cells.append(new_code_cell("x = 1"))
        _write_notebook(temp_notebook, nb)

        result = await modify_notebook_cells(
            temp_notebook, "delete", position_index=0
        )
        assert "message" in result

        # Verify cell was deleted
        nb = _read_notebook(temp_notebook)
        assert len(nb.cells) == 0

    @pytest.mark.asyncio
    async def test_invalid_operation(self, temp_notebook):
        """Test invalid operation."""
        with pytest.raises(ValueError, match="Invalid operation"):
            await modify_notebook_cells(temp_notebook, "invalid_op")


class TestQueryNotebook:
    """Test query_notebook function."""

    @pytest.mark.asyncio
    async def test_query_view_source_all_cells(self, temp_notebook, mock_serverapp):
        """Test viewing all cells."""
        set_server_context(mock_serverapp)

        nb = _read_notebook(temp_notebook)
        nb.cells.append(new_code_cell("x = 1"))
        nb.cells.append(new_markdown_cell("# Header"))
        _write_notebook(temp_notebook, nb)

        result = await query_notebook(temp_notebook, "view_source")
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_query_view_source_by_position(self, temp_notebook, mock_serverapp):
        """Test viewing cell by position."""
        set_server_context(mock_serverapp)

        nb = _read_notebook(temp_notebook)
        nb.cells.append(new_code_cell("x = 1"))
        _write_notebook(temp_notebook, nb)

        result = await query_notebook(temp_notebook, "view_source", position_index=0)
        assert isinstance(result, dict)
        assert result["source"] == "x = 1"

    @pytest.mark.asyncio
    async def test_query_check_server(self, mock_serverapp):
        """Test checking server status."""
        set_server_context(mock_serverapp)
        result = await query_notebook("dummy.ipynb", "check_server")
        assert "running" in result.lower()

    @pytest.mark.asyncio
    async def test_query_list_sessions(self, mock_serverapp):
        """Test listing sessions."""
        set_server_context(mock_serverapp)
        result = await query_notebook("dummy.ipynb", "list_sessions")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_query_list_kernels(self):
        """Test listing kernel specs."""
        result = await query_notebook("dummy.ipynb", "list_kernels")
        assert "kernelspecs" in result

    @pytest.mark.asyncio
    async def test_query_invalid_type(self, temp_notebook):
        """Test invalid query type."""
        with pytest.raises(ValueError, match="Invalid query_type"):
            await query_notebook(temp_notebook, "invalid_type")


@pytest.mark.asyncio
async def test_filter_large_outputs():
    """Test that _filter_large_outputs properly filters image data."""
    from jupyter_server_mcp.notebook_execution_tools import _filter_large_outputs

    # Create mock outputs with large image data
    outputs = [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": "Hello World\n"
        },
        {
            "output_type": "display_data",
            "data": {
                "text/plain": "Plot",
                "image/png": "iVBORw0KGgoAAAANS..." * 1000  # Simulated large base64 data
            }
        },
        {
            "output_type": "execute_result",
            "data": {
                "text/html": "<div>Large HTML content...</div>" * 100
            },
            "execution_count": 1
        }
    ]

    # Filter the outputs
    filtered = _filter_large_outputs(outputs)

    # Verify filtering
    assert len(filtered) == 3

    # Stream output should be unchanged
    assert filtered[0]["output_type"] == "stream"
    assert filtered[0]["text"] == "Hello World\n"

    # Image data should be filtered
    assert filtered[1]["output_type"] == "display_data"
    assert "[filtered]" in filtered[1]["data"]
    assert "image/png" in filtered[1]["data"]["[filtered]"]
    assert "text/plain" not in filtered[1]["data"]  # All data replaced with placeholder

    # HTML data should be filtered
    assert filtered[2]["output_type"] == "execute_result"
    assert "[filtered]" in filtered[2]["data"]
    assert "execution_count" in filtered[2]  # Metadata preserved
