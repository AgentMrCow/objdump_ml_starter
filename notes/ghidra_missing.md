Ghidra headless tooling is now installed under `third_party/ghidra_11.4.2_PUBLIC`. Use the helper script to regenerate exports:

```
GHIDRA_HOME=third_party/ghidra_11.4.2_PUBLIC scripts/tools_ghidra.sh
```

This runs `analyzeHeadless` for each `data/build/linux/O3/*_stripped` binary, writes CSVs to `out/ghidra/<stem>.csv`, and logs to `out/logs/ghidra_<stem>.log`. Feed those CSVs into `src/mining/make_hard_negatives.py` to refresh `out/mining/<stem>_hardnegs.json`.
