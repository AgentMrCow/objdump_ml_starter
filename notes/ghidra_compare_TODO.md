# Ghidra Function Export TODO

`analyzeHeadless` is not available on this host, so postpone the headless export step. When Ghidra is installed, run a command like:

```
analyzeHeadless <project_dir> <project_name> -import data/build/linux/O3/hello_stripped \
    -scriptPath tools -postScript ghidra_export_functions_headless.py out/ghidra/hello_stripped.csv
```

Key expectations:
- Place helper scripts under `tools/` (e.g., `ghidra_export_functions_headless.py`).
- Export CSV with columns: `binary`, `func_start`, `func_end`, `func_name`, `source` (optional note on how the symbol was discovered).
- Emit outputs in `out/ghidra/`, one CSV per binary (matching stripped binary basenames).
- Capture the `analyzeHeadless` stdout/stderr to `out/logs/ghidra_<name>.log` for traceability.

Follow this template for each stripped build once Ghidra tooling lands.
