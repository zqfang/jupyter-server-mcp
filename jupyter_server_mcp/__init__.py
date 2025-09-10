"""Jupyter Server MCP Extension with configurable tools."""

from typing import Any, Dict, List

from .extension import MCPExtensionApp

__version__ = "0.1.0"


def _jupyter_server_extension_points() -> List[Dict[str, Any]]:
    # pragma: no cover
    return [
        {
            "module": "jupyter_server_mcp.extension",
            "app": MCPExtensionApp,
        },
    ]


__all__ = ["MCPExtensionApp"]
