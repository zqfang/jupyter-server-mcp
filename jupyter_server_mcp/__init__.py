"""Jupyter Server MCP Extension with configurable tools."""

__version__ = "0.1.0"

from .extension import MCPExtensionApp
from typing import Any, Dict, List


def _jupyter_server_extension_points() -> List[Dict[str, Any]]:  # pragma: no cover
    return [
        {
            "module": "jupyter_server_docs_mcp.extension",
            "app": MCPExtensionApp,
        },
    ]

__all__ = ["MCPExtensionApp"]