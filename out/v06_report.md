# v0.6 Function-Start Detector (RosettaCode-Expanded)

## Dataset
- 1,625 real C sources from RosettaCode (C solutions only), built at O0–O3.
- Program-level split: 1137 train / 244 val / 244 test (`splits/v06.json`).
- Inventory: `out/dataset_inventory_v06.tsv` (4091 binaries built; failures logged inline during build).

## Models & Training
- Train opts: O0/O1/O2 on train programs (`scripts/train_models_v06.py`).
- Models: `models/start_detector_v06_{logreg,rf,xgb}.joblib`.
- Log-loss (train set, deduped vectors): LogReg 0.2524, RF 0.0296, XGB 0.1020.

## Threshold Selection (val O3)
- Best thresholds (macro F1 on val): LogReg 0.20 (MacroF1 0.580 / MacroR 0.812), RF 0.40 (0.669 / 0.909), XGB 0.40 (0.668 / 0.826) — see `out/macro_v06_val.tsv`.
- Plots: PR/F1 curves per model in `out/plots_v06/` (from `out/macro_v06_plot.tsv`).

## Test O3 (244 programs)
- LogReg @0.20: Macro P/R/F1 = 0.455 / 0.823 / 0.566 (`out/summary_thr_v06_logreg_O3_0.20.tsv`).
- RF @0.40: Macro P/R/F1 = 0.525 / 0.900 / 0.638 (`out/summary_thr_v06_rf_O3_0.40.tsv`).
- XGB @0.40: Macro P/R/F1 = 0.564 / 0.818 / 0.645 (`out/summary_thr_v06_xgb_O3_0.40.tsv`).
- RF chosen: highest macro F1 while keeping recall at 0.90.

## Cross-Opt (test programs, RF @0.40)
- O1: Macro 0.673 / 0.904 / 0.762 (`out/summary_thr_v06_rf_O1_0.40.tsv`).
- O2: Macro 0.575 / 0.901 / 0.688 (`out/summary_thr_v06_rf_O2_0.40.tsv`).

## Ghidra Agreement (test O3 subset)
- Exports under `out/ghidra/O3_<stem>_stripped.csv` (headless via `_JAVA_OPTIONS=-Duser.home=...`).
- Comparison table: `out/ghidra_compare_v06_O3.tsv` (45 binaries processed; O1/O2 tables in `out/ghidra_compare_v06_O{1,2}.tsv`).
- FN/FP feature dumps: `out/error_analysis_v06/` via `tools/list_mistakes_v06.py`.

## Threshold Rationale
- Macro sweeps per model/threshold: `out/macro_v06.tsv` (test) and `out/macro_v06_val.tsv` (val). RF shows a plateau 0.35–0.40 with recall ≥0.90; we pick 0.40 to balance precision.
- PR/F1 plots (`out/plots_v06/pr_*`, `f1_*`) visualize the trade-offs; LogReg stays lower-left, XGB lifts recall but drops precision on diverse O3.

## Artifacts
- Models: `models/start_detector_v06_{logreg,rf,xgb}.joblib`
- Summaries: `out/summary_thr_v06_{logreg,rf,xgb}_O3_<thr>.tsv`, RF O1/O2 summaries
- Macros: `out/macro_v06.tsv`, `out/macro_v06_val.tsv`, `out/macro_v06_plot.tsv`
- Plots: `out/plots_v06/`
- Ghidra: `out/ghidra_compare_v06_O{1,2,3}.tsv`, CSV exports under `out/ghidra/`
- Error dumps: `out/error_analysis_v06/`

## Next Steps
1. Finish Ghidra compare across all test binaries (O1/O2) and fold agree/miss/extra into model selection.
2. Run ablations (post-filter off, cond-branch candidates off, no hard negatives) using the fast scorer; log to an ablation TSV.
3. Consider class-weighting or threshold tuning per program family to reduce RF FPs on complex tasks.
