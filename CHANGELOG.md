# Changelog



## 0.2.0 (Unreleased)

### Bug Fixes

- **[CRITICAL]** Fixed code execution failure with "Invalid message type: coroutine" error
  - **Root cause:** Incorrectly using `run_in_executor()` with async `AsyncKernelClient.get_iopub_msg()` method
  - **Solution:** Directly await the async method instead of wrapping it in a thread pool executor
  - **Impact:** Restores all code execution functionality for both Python and R kernels
  - **Files changed:** `jupyter_server_mcp/notebook_execution_tools.py` (line 309-313)
  - **Affects:** `execute_cell()`, `modify_notebook_cells()` with `execute=True`, `execute_notebook_code()`

- **[CRITICAL]** Fixed missing `execution_count` in notebook cells after code execution
  - **Root cause:** Only capturing `execution_count` from `execute_result` messages on IOPub channel, which are not sent for code that only produces print output
  - **Solution:** Read `execution_count` from `execute_reply` message on Shell channel, which is ALWAYS present according to Jupyter messaging protocol
  - **Impact:** Execution counts now properly populate in notebook files for all types of code execution
  - **Files changed:**
    - `jupyter_server_mcp/notebook_execution_tools.py` (lines 293-325, 368-383)
    - `tests/test_notebook_execution_tools.py` (lines 76, 83-145, 343-397)
  - **Affects:** All code execution functions: `execute_cell()`, `modify_notebook_cells()`, `execute_notebook()`, `execute_notebook_code()`
  - **User benefit:** Cells can now be referenced by execution count `[1]`, `[2]`, etc. in JupyterLab UI


## 0.1.2

([Full Changelog](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/compare/v0.1.1...754ead215d3a9bb6e46f9e8d874bb9911caec3eb))

### Enhancements made

- Add automatic tool discovery via Python entrypoints [#10](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/pull/10) ([@Zsailer](https://github.com/Zsailer))

### Documentation improvements

- Add screencast to the README [#9](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/pull/9) ([@jtpio](https://github.com/jtpio))
- Add Gemini CLI configuration [#6](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/pull/6) ([@jtpio](https://github.com/jtpio))
- Update README to the new toolkit [#5](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/pull/5) ([@jtpio](https://github.com/jtpio))

### Contributors to this release

([GitHub contributors page for this release](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/graphs/contributors?from=2025-09-11&to=2025-10-24&type=c))

[@jtpio](https://github.com/search?q=repo%3Ajupyter-ai-contrib%2Fjupyter-server-mcp+involves%3Ajtpio+updated%3A2025-09-11..2025-10-24&type=Issues) | [@Zsailer](https://github.com/search?q=repo%3Ajupyter-ai-contrib%2Fjupyter-server-mcp+involves%3AZsailer+updated%3A2025-09-11..2025-10-24&type=Issues)


## 0.1.1

([Full Changelog](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/compare/fb0b9a59e08150ce6c285a9af41dac2342e7d181...6bf57b34817b2b631f91afb8bab6a72d48fa7a5b))

### Maintenance and upkeep improvements

- Add releaser workflows [#3](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/pull/3) ([@jtpio](https://github.com/jtpio))

### Other merged PRs

- Add automatic JSON argument conversion for MCP tools [#4](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/pull/4) ([@Zsailer](https://github.com/Zsailer))
- Rename to jupyter-server-mcp [#2](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/pull/2) ([@Zsailer](https://github.com/Zsailer))
- Initial implementation of an MCP server as an extension [#1](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/pull/1) ([@Zsailer](https://github.com/Zsailer))

### Contributors to this release

([GitHub contributors page for this release](https://github.com/jupyter-ai-contrib/jupyter-server-mcp/graphs/contributors?from=2025-08-14&to=2025-09-11&type=c))

[@jtpio](https://github.com/search?q=repo%3Ajupyter-ai-contrib%2Fjupyter-server-mcp+involves%3Ajtpio+updated%3A2025-08-14..2025-09-11&type=Issues) | [@Zsailer](https://github.com/search?q=repo%3Ajupyter-ai-contrib%2Fjupyter-server-mcp+involves%3AZsailer+updated%3A2025-08-14..2025-09-11&type=Issues)
