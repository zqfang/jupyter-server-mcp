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
    _get_tracked_kernel,
    _normalize_path,
    _read_notebook,
    _track_kernel,
    _write_notebook,
    execute_cell,
    execute_notebook,
    get_server_context,
    list_available_kernels,
    list_running_kernels,
    modify_notebook_cells,
    query_notebook,
    set_server_context,
    setup_notebook,
    shutdown_kernel,
    shutdown_notebook,
    switch_notebook_kernel,
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

    def client(self):
        """Return a mock client."""
        return MockKernelClient()


class MockKernelClient:
    """Mock kernel client for testing."""

    def __init__(self):
        self.channels_started = False
        self.execution_count = 1

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
        """Mock execute."""
        return "msg-id-123"

    def get_iopub_msg(self):
        """Mock get iopub message."""
        # Simulate execution complete
        return {
            "header": {"msg_type": "status", "msg_id": "msg-id-456"},
            "parent_header": {"msg_id": "msg-id-123"},
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

        _track_kernel(path, kernel_id)
        assert _get_tracked_kernel(path) == kernel_id

    def test_get_tracked_kernel_none(self):
        """Test getting non-existent kernel."""
        result = _get_tracked_kernel("/nonexistent/path.ipynb")
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


class TestExecuteNotebook:
    """Test execute_notebook function."""

    @pytest.mark.asyncio
    async def test_execute_notebook_not_found(self):
        """Test executing non-existent notebook."""
        result = await execute_notebook("/nonexistent/notebook.ipynb")
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_notebook_empty(self, temp_notebook):
        """Test executing empty notebook."""
        with patch("jupyter_server_mcp.notebook_execution_tools.NotebookClient") as mock_client_class:
            mock_client = Mock()
            mock_client.async_execute = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await execute_notebook(temp_notebook)
            assert result["status"] == "completed"
            assert result["cells_executed"] == 0
            assert result["cells_total"] == 0


class TestShutdownKernel:
    """Test shutdown_kernel function."""

    @pytest.mark.asyncio
    async def test_shutdown_kernel_no_context(self):
        """Test shutdown without server context."""
        result = await shutdown_kernel("kernel-123")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_shutdown_kernel_not_found(self, mock_serverapp):
        """Test shutting down non-existent kernel."""
        set_server_context(mock_serverapp)
        result = await shutdown_kernel("nonexistent-kernel")
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_shutdown_kernel_success(self, mock_serverapp):
        """Test successful kernel shutdown."""
        set_server_context(mock_serverapp)
        kernel_id = await mock_serverapp.kernel_manager.start_kernel()

        result = await shutdown_kernel(kernel_id)
        assert result["status"] == "shutdown"
        assert result["kernel_id"] == kernel_id


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
        _track_kernel(_normalize_path(temp_notebook), kernel_id)

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
