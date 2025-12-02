# objdump-ML Starter (Function Start Detection)

This repo augments `objdump` (linear sweep) with lightweight ML models to detect **function starts** on x86‑64 ELF binaries. It ships both a tiny starter demo and a scaled corpus (v0.6) with program-level splits.

## Quickstart (starter demo)

```bash
cd objdump_ml_starter

# 1) env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) build the small sample set (O0/O3; symbols + stripped)
python src/build_dataset.py

# 3) train on O0 symbol binaries
python src/train_start_detector.py --train_glob "data/build/*/O0/*_sym" --model_path models/start_detector.joblib

# 4) predict on a stripped O3 binary
python src/predict_starts.py --bin "data/build/linux/O3/hello_stripped" --model_path models/start_detector.joblib --out functions_pred.json

# 5) evaluate vs truth (±8-byte tolerance)
python src/eval_starts.py --pred functions_pred.json --truth_glob "data/labels/*/O3/hello_sym.functions_truth.json" --tolerance 8
```

## Scaled pipeline (v0.6+)

- Corpus: ~1.6k real C programs (RosettaCode) built at O0–O3, with DWARF truth, asm JSON, and program-level splits (no cross-program leakage). See `data/program_manifest_v06.json`, `splits/v06.json`, inventory `out/dataset_inventory_v06.tsv`.
- Models: Logistic Regression, Random Forest, XGBoost; post-filters for padding/jump-table artifacts; threshold sweeps and macro aggregation; optional Ghidra headless exports for agreement checks.

Common commands (after `source .venv/bin/activate` and `export PYTHONPATH=src`):

```bash
# Build/refresh the large dataset (supports --start/--end/--opt_levels)
python src/build_dataset.py

# Train tuned models on train split (O0/O1/O2) -> models/start_detector_v06f_*.joblib
python scripts/train_models_v06_tuned.py --split splits/v06.json --train_opts O0,O1,O2 --tag v06f --out_dir models

# Sweep thresholds on O3 test set
python scripts/evaluate_model_thresholds_v06.py \
  --bins_list out/test_bins_O3.txt \
  --model models/start_detector_v06f_rf.joblib \
  --out_prefix out/summary_thr_v06f_rf_O3 \
  --thresholds "0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40,0.45,0.50,0.55"

# Aggregate macro P/R/F1 across sweeps
python tools/aggregate_macros_v06.py \
  --patterns out/summary_thr_v06f_rf_O3_{thr}.tsv \
             out/summary_thr_v06f_xgb_O3_{thr}.tsv \
             out/summary_thr_v06f_logreg_O3_{thr}.tsv \
  --thresholds 0.15 0.18 0.20 0.22 0.25 0.28 0.30 0.35 0.40 0.45 0.50 0.55 \
  --out out/macro_v06f_O3.tsv

# Optional: Ghidra headless export + agreement checks (if analyzeHeadless is available)
scripts/tools_ghidra.sh          # uses bin lists under out/
python tools/compare_to_ghidra.py --out out/ghidra_compare_v06d_O3.tsv
```

Key artifacts
- Data: `data/build/linux/O*/...`, `data/labels/linux/O*/...`, `data/program_manifest_v06.json`
- Splits: `splits/v06.json` (train/val/test by program)
- Models: `models/start_detector*.joblib` (v06f_* are the tuned ones)
- Results: sweep TSVs under `out/summary_thr_*`, macro tables `out/macro_v06*.tsv`, plots under `out/plots_v06/`, Ghidra exports under `out/ghidra/`

## Project layout
```
src/
  build_dataset.py      # builds binaries (O0–O3), symbol & stripped
  parse_objdump.py      # parses `objdump -d` to JSON
  elf_labels.py         # extracts function start/end from DWARF/.symtab
  features.py           # candidates + features (context, alignment, n-grams, CFG extras)
  predict_starts.py     # inference with post-filters and merging
  eval_starts.py        # P/R/F1 with tolerance (mean/median offset)
  train_start_detector.py
scripts/
  train_models_v06_*.py # model training variants
  evaluate_model_thresholds_v06.py, tools_ghidra.sh, etc.
samples/
  hello.c, mathlib.c, sort.c, real_v06/... (RosettaCode corpus)
data/
  build/...             # binaries/asm
  labels/...            # truth JSON
  program_manifest_v06.json
splits/
  v06.json              # program-level split (no leakage)
models/
  start_detector.joblib # starter
  start_detector_v06*.joblib
out/
  # evaluation summaries, plots, ghidra exports, macro tables
```

## Notes
- DWARF truth is the primary reference; Ghidra exports are used for agreement/error analysis.
- Start with the starter demo, then move to the v0.6 pipeline for larger-scale experiments.
- Architecture scope: x86-64 ELF. Extend candidates/features for other ISAs as needed.
