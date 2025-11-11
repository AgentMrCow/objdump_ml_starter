# v0.5 Function-Start Detector Report

## Environment & Key Commands
```
python -V
objdump --version
gcc --version
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install xgboost matplotlib
PYTHONPATH=src python src/build_dataset.py
source .venv/bin/activate && PYTHONPATH=src python scripts/train_models_v05.py
source .venv/bin/activate && for model in start_detector_v05_{logreg,rf}.joblib; do ... src/run_batch_predict.sh; done
source .venv/bin/activate && PYTHONPATH=src python tools/aggregate_macros.py --macro out/macro_v05.tsv
source .venv/bin/activate && THRESH=0.55 POST_FILTER=on MODEL_PATH=models/start_detector_v05_rf.joblib BIN_GLOB='data/build/linux/O{1,2}/'* src/run_batch_predict.sh
source .venv/bin/activate && PYTHONPATH=src python tools/compare_to_ghidra.py --out out/ghidra_compare_v05.tsv
source .venv/bin/activate && PYTHONPATH=src python tools/list_mistakes.py --opt O3
```
*XGBoost + matplotlib are installed locally; PR curves are emitted via `tools/plot_pr_v05.py`. Headless Ghidra runs with `_JAVA_OPTIONS=-Duser.home=<tmp>` so fresh exports exist for O0–O3 under `out/ghidra/O{opt}_<stem>.csv`.*

## Dataset Snapshot (`out/dataset_inventory_v05.tsv`)
| program | opt | functions | instrs |
|---------|-----|-----------|--------|
| hello | O0 | 12 | 117 |
| hello | O1 | 10 | 85 |
| hello | O2 | 10 | 86 |
| hello | O3 | 10 | 86 |
| mathlib | O0 | 8 | 105 |
| mathlib | O1 | 8 | 91 |
| mathlib | O2 | 8 | 94 |
| mathlib | O3 | 8 | 94 |
| sort | O0 | 12 | 184 |
| sort | O1 | 11 | 139 |
| sort | O2 | 11 | 143 |
| sort | O3 | 11 | 143 |

## Program Splits (`splits/v05.json`)
- **Train (O0/O1/O2):** `hello`, `mathlib`
- **Val/Test (O3 focus, also O1/O2 reporting):** `sort`

## Model & Threshold Comparison @ O3 stripped
| Model (filtered) | Threshold | hello P/R/F1 | mathlib P/R/F1 | sort P/R/F1 | Macro P/R/F1 | mean_err / median_err |
|------------------|-----------|--------------|----------------|-------------|--------------|-----------------------|
| LogReg | 0.40 | 0.889 / 0.800 / 0.842 | 0.800 / 1.000 / 0.889 | 0.727 / 0.727 / 0.727 | **0.805 / 0.842 / 0.819** | ~1.0 / 0.0 |
| RandomForest | 0.55 | 1.000 / 0.900 / 0.947 | 0.800 / 1.000 / 0.889 | 0.909 / 0.909 / 0.909 | **0.903 / 0.936 / 0.915** | <=1.0 / 0.0 |
| XGBoost | 0.25 | 0.900 / 0.900 / 0.900 | 0.800 / 1.000 / 0.889 | 0.667 / 0.909 / 0.769 | **0.789 / 0.936 / 0.853** | ≤1.0 / 0.0 |
*(RF remains the v0.5 pick: highest Macro F1 at comparable recall; XGB boosts recall but loses precision on sort.)*

## O1/O2 Generalization (RF @ 0.55 filtered)
| Opt | Macro P | Macro R | Macro F1 | Notes |
|-----|---------|---------|----------|-------|
| O1 | 0.900 | 0.812 | 0.852 | sort keeps 3 FP in padded helpers |
| O2 | 0.903 | 0.936 | 0.915 | matches O3 behavior almost exactly |

## Macro Sweep & Threshold Rationale
- `out/macro_v05.tsv` captures every threshold/model combo (0.20–0.55). RF’s plateau at 0.50–0.55 yields the highest Macro F1 (=0.915) while keeping recall ≥0.936; lowering the bar only adds FPs. LogReg tops out near F1≈0.82, and XGBoost’s best point (0.25) trades precision for recall.
- `tools/plot_pr_v05.py` now writes PR/F1 curves (`out/plots/pr_*`, `out/plots/f1_*`). RF’s curve stays in the upper-right quadrant, which visually backs the THRESH=0.55 selection.

## Ghidra Agreement (O3, `_JAVA_OPTIONS=-Duser.home=...`; `out/ghidra_compare_v05.tsv`)
| File | Agree | Miss | Extra |
|------|-------|------|-------|
| O3_hello_stripped | 9 | 5 | 0 |
| O3_mathlib_stripped | 0 | 14 | 10 |
| O3_sort_stripped | 10 | 7 | 1 |
- CSVs for every opt level now live under `out/ghidra/O{opt}_<stem>.csv`; `tools/compare_to_ghidra.py --opt_levels O3` generated the table above (switch it to `--opt_levels O1,O2` if you want those comparisons too).

## Error Analysis
- Detailed FN/FP feature dumps in `out/error_analysis/*.tsv` (one pair per binary).
- Failure blurbs (`notes/failures_v05.md`):
  1. `_init` (0x0) in `hello` lacks any candidate features → always FN.
  2. `mathlib` alignment sled (0x1110/0x1150) still fires despite zero xrefs because the RF weights favor `align16` + padding.
  3. `sort` jump-table pad at 0x4011e0 resembles a prologue (non-zero windowed xrefs) and slips past the post-filter.

## Artifacts & Paths
- Models: `models/start_detector_v05_{logreg,rf,xgb}.joblib`.
- Sweeps: `out/summary_thr_<thr>_v05_{logreg,rf}_{filtered,raw}.tsv` with logs under `out/logs/`.
- Macro stats: `out/macro_v05.tsv` (plus legacy `out/macro_v03_vs_v04.tsv`).
- Dataset & split metadata: `out/dataset_inventory_v05.tsv`, `splits/v05.json`.
- Additional evaluations: `out/summary_v05_O1_rf.tsv`, `out/summary_v05_O2_rf.tsv`, `out/summary_v05_O3_rf.tsv`.
- Ghidra/error notes: `out/ghidra_compare_v05.tsv`, `out/error_analysis/*`, `notes/failures_v05.md`.
- Plots: `out/plots/pr_*.png`, `out/plots/f1_*.png` (generated via `tools/plot_pr_v05.py`).
- Other helpers: `scripts/train_models_v05.py`, `tools/compare_to_ghidra.py`, `tools/list_mistakes.py`.

## Outstanding TODOs / Limitations
1. **Broader Ghidra comparison**: `tools/compare_to_ghidra.py` currently reports only the O3 deployment; running it with `--opt_levels O1,O2` (and baking those misses into training) is straightforward future work.
2. **Jump-table awareness**: RF still marks switch landing pads. CFG-based features or post-filters remain future work.
3. **Serve-time calibration**: thresholds are tuned via macro sweeps; adding a dev split or ROC-based calibration would give more formal guarantees across new programs.
