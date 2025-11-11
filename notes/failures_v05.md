# v0.5 Failure Snippets

1. `hello` (O3) still misses `_init` at `0x0` because the stripped binary lacks a preceding code byte for candidate features; without xrefs and with zero context, the classifier never scores it above threshold.
2. `mathlib` shows false positives at `0x1110`/`0x1150` (see `out/error_analysis/mathlib_fp.tsv`): both live inside alignment sleds where `align16=1`, zero xrefs, and no prologue pattern; RF still elevates them due to padding heuristics.
3. `sort` has a persistent FP at `0x4011e0` where the jump-table landing pad mimics a prologue (windowed xref counts >0). Without CFG awareness, the model flags it even though Ghidra classifies it as part of the dispatch table.
