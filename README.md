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
- Current best validated O3 setup: `models/start_detector_v06h_xgb.joblib` with `THRESH=0.75`, `POST_FILTER=on`, `MERGE_WINDOW=4`, giving macro `P=0.9867 / R=0.9565 / F1=0.9701` on the 149-binary O3 test split.

Common commands (after `source .venv/bin/activate` and `export PYTHONPATH=src`):

```bash
# Build/refresh the large dataset (supports --start/--end/--opt_levels)
python src/build_dataset.py

# Train tuned models on train split -> models/start_detector_v06h_*.joblib
python scripts/train_models_v06_tuned.py --split splits/v06.json --train_opts O0,O1,O2,O3 --tag v06h --out_dir models

# Optional ablation: retrain without the `reachable` feature
python scripts/train_models_v06_tuned.py \
  --split splits/v06.json \
  --train_opts O0,O1,O2,O3 \
  --tag v06i \
  --drop_features reachable \
  --out_dir models

# Sweep thresholds on O3 test set
python scripts/evaluate_model_thresholds_v06.py \
  --bins_list out/binlist_test_O3_stripped.txt \
  --model models/start_detector_v06h_xgb.joblib \
  --out_prefix out/v06h_rescue_sweep \
  --thresholds "0.55,0.60,0.65,0.70,0.75,0.80" \
  --merge_window 4 \
  --post_filter on

# Batch prediction with the current best settings
MODEL_PATH=models/start_detector_v06h_xgb.joblib \
THRESH=0.75 \
POST_FILTER=on \
MERGE_WINDOW=4 \
BIN_LIST=out/binlist_test_O3_stripped.txt \
src/run_batch_predict.sh

# Optional: Ghidra headless export + agreement checks (if analyzeHeadless is available)
scripts/tools_ghidra.sh          # uses bin lists under out/
python tools/compare_to_ghidra.py --out out/ghidra_compare_v06d_O3.tsv

# Streamlit demo (local)
streamlit run web/streamlit_app.py  # PYTHONPATH=src, models/ + data/ present
```

Key artifacts
- Data: `data/build/linux/O*/...`, `data/labels/linux/O*/...`, `data/program_manifest_v06.json`
- Splits: `splits/v06.json` (train/val/test by program)
- Models: `models/start_detector*.joblib` (`v06h_xgb` is the current best O3 model; `v06i_*` is the no-`reachable` ablation)
- Results: sweep TSVs under `out/*sweep*.tsv`, macro tables such as `out/macro_v06_compare.tsv` and `out/macro_v06_current_thresholds.tsv`, batch summaries like `out/summary_v06_best_O3.tsv`, plots under `out/plots_v06/`, Ghidra exports under `out/ghidra/`

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
  evaluate_model_thresholds_v06.py # threshold sweeps using current rescue/filter path
  tools_ghidra.sh, etc.
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
- `elf_labels.py` now filters out undefined/imported zero-address symbols, so `functions_truth.json` does not count PLT/import names as fake misses.
- Start with the starter demo, then move to the v0.6 pipeline for larger-scale experiments.
- Architecture scope: x86-64 ELF. Extend candidates/features for other ISAs as needed.
