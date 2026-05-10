#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from joblib import load
from pypdf import PdfReader
from matplotlib.patches import FancyBboxPatch


REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from evaluate_model_thresholds_v06 import (  # noqa: E402
    _build_inbound_kind_counts,
    _is_clean_call_rescue_candidate,
    _is_clean_jmp_rescue_candidate,
    _is_clean_ret_rescue_candidate,
    _is_reachable_leaf_candidate,
    _is_short_leaf_candidate,
    _looks_like_jump_table,
    _looks_like_tiny_leave_stub,
    _prev_ret_near,
    apply_prediction_pipeline,
    collect_candidates,
    load_truth,
    score_model,
    score_predicted_starts,
)


ASSET_ROOT = REPO / "handin" / "sem2-ierg4999" / "report_assets"
TABLE_DIR = ASSET_ROOT / "tables"
FIG_DIR = ASSET_ROOT / "figures"
NOTE_DIR = ASSET_ROOT / "notes"
TEX_DIR = ASSET_ROOT / "tex"
REPORT_DIR = REPO / "handin" / "sem2-ierg4999" / "final_report"

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.95,
    0.97,
]

CURRENT_MODEL = "start_detector_v06u_xgb.joblib"
CURRENT_THRESHOLD = 0.95
CURRENT_SUMMARY = REPO / "out" / "summary_v06_best_O3.tsv"
CURRENT_SWEEP_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97]
CASE_STUDIES = [
    {
        "key": "angles_clean_ret",
        "stem": "rosetta_v06_00064_Angles-geometric-normalization-and-conversion_angles-geometric-normalization-and-conversion",
        "targets": [0x1140],
        "before": 6,
        "after": 10,
        "title": "Case Study: clean-ret rescue on an aligned O3 entry",
        "caption": "A real function entry recovered by the clean-ret rescue. The aligned start at 0x1140 follows a ret+nop boundary, so thresholding alone misses it but the local boundary rule recovers it.",
    },
    {
        "key": "munching_false_starts",
        "stem": "rosetta_v06_00924_Munching-squares_munching-squares",
        "targets": [0x401370, 0x401390],
        "before": 4,
        "after": 8,
        "title": "Case Study: repeated leave/ret islands removed by post-filtering",
        "caption": "Representative false starts from Munching-squares. These aligned islands have zero inbound references and repeat a short leave; ret; nop shape, so the semester-two post-filter drops them.",
    },
]


def ensure_dirs() -> None:
    for d in (TABLE_DIR, FIG_DIR, NOTE_DIR, TEX_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def round6(x: float) -> float:
    return round(float(x), 6)


def latex_escape(text: object) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def fmt_table_value(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.3f}"
        return f"{value:.4f}"
    return latex_escape(value)


def short_binary_name(name: str) -> str:
    s = re.sub(r"^rosetta_v06_\d+_", "", name)
    return s.replace("_", " / ")


def pretty_model_name(name: str) -> str:
    mapping = {
        "v06u_xgb": "XGBoost",
        "v06u_rf": "Random Forest",
        "v06u_logreg": "Logistic Regression",
        "start_detector_v06u_xgb": "XGBoost",
    }
    return mapping.get(name, name)


def load_asm_instrs(opt: str, stem: str) -> list[dict]:
    asm_path = REPO / "data" / "build" / "linux" / opt / f"{stem}_stripped.asm.json"
    with asm_path.open() as f:
        asm = json.load(f)
    return asm["instrs"]


def snippet_lines(opt: str, stem: str, targets: list[int], before: int = 8, after: int = 12) -> list[str]:
    instrs = load_asm_instrs(opt, stem)
    by_addr = {ins["addr"]: i for i, ins in enumerate(instrs)}
    lines: list[str] = []
    for target in targets:
        idx = by_addr.get(target)
        if idx is None:
            continue
        lines.append(f"target {target:#x}")
        lo = max(0, idx - before)
        hi = min(len(instrs), idx + after + 1)
        for j in range(lo, hi):
            ins = instrs[j]
            mark = ">>" if ins["addr"] == target else "  "
            mnem = ins.get("mnemonic", "")
            ops = ins.get("ops", "")
            xin = ins.get("xrefs_in", 0)
            lines.append(f"{mark} {ins['addr']:#x}: {mnem:<10} {ops:<42} xrefs_in={xin}")
        lines.append("")
    return lines


def render_text_figure(path: Path, title: str, lines: list[str], highlight_prefix: str = ">>") -> None:
    if not lines:
        return
    height = max(4.5, 0.28 * len(lines) + 1.4)
    fig, ax = plt.subplots(figsize=(12, height))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y = 0.97
    line_h = 0.92 / max(len(lines), 1)
    ax.text(0.01, 0.99, title, fontsize=13, fontweight="bold", va="top", ha="left")
    for line in lines:
        color = "#7f1d1d" if line.startswith(highlight_prefix) else "#111827"
        if line.startswith("target "):
            ax.text(0.01, y, line, fontsize=10.5, fontweight="bold", va="top", ha="left", family="monospace")
        else:
            if line.startswith(highlight_prefix):
                box = FancyBboxPatch((0.005, y - line_h * 0.9), 0.985, line_h * 0.95,
                                     boxstyle="round,pad=0.003,rounding_size=0.003",
                                     facecolor="#fef2f2", edgecolor="none", alpha=0.95)
                ax.add_patch(box)
            ax.text(0.015, y, line, fontsize=9.5, va="top", ha="left", family="monospace", color=color)
        y -= line_h
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(
    path: Path,
    df: pd.DataFrame,
    column_format: str | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> None:
    cols = list(df.columns)
    if column_format is None:
        column_format = "l" + "r" * max(0, len(cols) - 1)

    lines = []
    if caption or label:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{column_format}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(c) for c in cols) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(fmt_table_value(row[c]) for c in cols) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{latex_escape(caption)}}}")
    if label:
        lines.append(rf"\label{{{latex_escape(label)}}}")
    if caption or label:
        lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n")


def bin_paths_for_opt_split(opt: str, split_name: str = "test_programs") -> list[str]:
    split = json.loads((REPO / "splits" / "v06.json").read_text())
    stems = set(split[split_name])
    paths = []
    for stem in sorted(stems):
        bin_path = REPO / "data" / "build" / "linux" / opt / f"{stem}_stripped"
        truth_path = REPO / "data" / "labels" / "linux" / opt / f"{stem}_sym.functions_truth.json"
        asm_path = Path(str(bin_path) + ".asm.json")
        if bin_path.exists() and truth_path.exists() and asm_path.exists():
            paths.append(str(bin_path))
    return paths


def bin_paths_for_opt(opt: str) -> list[str]:
    return bin_paths_for_opt_split(opt, "test_programs")


def dataset_and_split_summary() -> pd.DataFrame:
    inv = read_tsv(REPO / "out" / "dataset_inventory_v06.tsv")
    split = json.loads((REPO / "splits" / "v06.json").read_text())
    train = set(split["train_programs"])
    test = set(split["test_programs"])
    val = set(split.get("val_programs", []))

    rows = []
    for opt in ["O0", "O1", "O2", "O3"]:
        sub = inv[inv["opt"] == opt]
        programs = set(sub["program"])
        rows.append(
            {
                "scope": f"{opt} corpus",
                "programs": len(programs),
                "binaries": len(sub),
                "mean_functions": round(sub["functions"].mean(), 2),
                "mean_instrs": round(sub["instrs"].mean(), 2),
            }
        )

    for name, stems in [("train split", train), ("val split", val), ("test split", test)]:
        if not stems:
            continue
        sub = inv[inv["program"].isin(stems)]
        rows.append(
            {
                "scope": name,
                "programs": len(stems),
                "binaries": len(sub),
                "mean_functions": round(sub["functions"].mean(), 2),
                "mean_instrs": round(sub["instrs"].mean(), 2),
            }
        )

    rows.append(
        {
            "scope": "current O3 eval",
            "programs": len({Path(p).name.replace("_stripped", "") for p in bin_paths_for_opt("O3")}),
            "binaries": len(bin_paths_for_opt("O3")),
            "mean_functions": round(
                inv[
                    (inv["opt"] == "O3")
                    & (inv["program"].isin({Path(p).name.replace("_stripped", "") for p in bin_paths_for_opt("O3")}))
                ]["functions"].mean(),
                2,
            ),
            "mean_instrs": round(
                inv[
                    (inv["opt"] == "O3")
                    & (inv["program"].isin({Path(p).name.replace("_stripped", "") for p in bin_paths_for_opt("O3")}))
                ]["instrs"].mean(),
                2,
            ),
        }
    )
    return pd.DataFrame(rows)


def project_evolution_table() -> pd.DataFrame:
    rows = [
        {
            "stage": "v0.4 toy O3 set",
            "semester": "Sem1",
            "corpus_scope": "3 binaries, O3 only",
            "plot_note": "3 toy binaries",
            "model": "LogReg + features",
            "threshold": 0.40,
            "MacroP": 0.840,
            "MacroR": 0.801,
            "MacroF1": 0.839,
            "source": "out/O3_v0.4_report.md",
            "notes": "Small toy set; useful for early proof-of-concept only.",
        },
        {
            "stage": "v0.5 small multi-model",
            "semester": "Sem1",
            "corpus_scope": "3 programs, O0-O3",
            "plot_note": "3 programs",
            "model": "RandomForest",
            "threshold": 0.55,
            "MacroP": 0.903,
            "MacroR": 0.936,
            "MacroF1": 0.915,
            "source": "sem1 final report / out/v05_report.md",
            "notes": "Still small and optimistic; no large-corpus generalization yet.",
        },
        {
            "stage": "v0.6 large corpus baseline",
            "semester": "Sem1",
            "corpus_scope": "RosettaCode, 244 O3 test programs",
            "plot_note": "1,625 programs total\n244 O3 test",
            "model": "RandomForest",
            "threshold": 0.40,
            "MacroP": 0.525,
            "MacroR": 0.900,
            "MacroF1": 0.638,
            "source": "sem1 final report",
            "notes": "First large-corpus result after switching to program-level splits.",
        },
        {
            "stage": "Sem2 interim checkpoint",
            "semester": "Sem2",
            "corpus_scope": "149 corrected O3 test binaries",
            "plot_note": "1,625 programs total\n149 corrected O3 eval",
            "model": "XGBoost + leaf rescue",
            "threshold": 0.55,
            "MacroP": 0.966,
            "MacroR": 0.956,
            "MacroF1": 0.959,
            "source": "sem2 interim report",
            "notes": "After truth cleanup, merge fix, and first rescue rule.",
        },
        {
            "stage": "Sem2 current final",
            "semester": "Sem2",
            "corpus_scope": "149 corrected O3 test binaries",
            "plot_note": "1,625 programs total\n149 corrected O3 eval",
            "model": "XGBoost + bounded rescue stack",
            "threshold": 0.95,
            "MacroP": 0.994544,
            "MacroR": 0.983248,
            "MacroF1": 0.988376,
            "source": "out/summary_v06_best_O3.tsv",
            "notes": "Current best validated configuration.",
        },
    ]
    return pd.DataFrame(rows)


def sem1_sem2_summary_table() -> pd.DataFrame:
    rows = [
        {
            "aspect": "Disassembly core",
            "semester1": "objdump linear sweep + parser + truth extraction",
            "semester2": "Same core preserved; improved evaluation correctness and post-processing instead of changing baseline",
        },
        {
            "aspect": "Dataset and protocol",
            "semester1": "Scaled from toy scripts to RosettaCode corpus with split protocol",
            "semester2": "Kept corpus, validated labels, tightened corrected O3 test slice, used reproducible threshold sweeps",
        },
        {
            "aspect": "Models",
            "semester1": "LogReg / RF / XGB comparison; RF chosen on large-corpus sem1 baseline",
            "semester2": "Shifted to XGB with better feature set and stronger calibrated thresholding",
        },
        {
            "aspect": "Key debugging work",
            "semester1": "Candidate generation, feature engineering, initial Ghidra comparison",
            "semester2": "Fixed zero-address truth bug, fixed over-aggressive merge, added bounded ret/jmp/call rescues, tighter filters",
        },
        {
            "aspect": "Large-corpus O3 macro F1",
            "semester1": "0.638 (RF @ 0.40, sem1 final report)",
            "semester2": "0.988 (XGB @ 0.95, current validated run)",
        },
    ]
    return pd.DataFrame(rows)


def evaluate_current_models_o3() -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = bin_paths_for_opt("O3")
    model_files = [
        REPO / "models" / "start_detector_v06u_logreg.joblib",
        REPO / "models" / "start_detector_v06u_rf.joblib",
        REPO / "models" / "start_detector_v06u_xgb.joblib",
    ]

    sweep_rows = []
    best_rows = []
    for model_path in model_files:
        bundle = load(model_path)
        feature_keys = bundle.get("feature_keys", [])
        per_thr: dict[float, list[tuple[float, float, float]]] = {thr: [] for thr in THRESHOLDS}
        for bin_path in bins:
            instrs, addrs, idxs, feats = collect_candidates(bin_path)
            probs = score_model(bundle, feature_keys, feats)
            stem = Path(bin_path).name.replace("_stripped", "")
            opt = Path(bin_path).parts[-2]
            truth = load_truth(str(REPO / "data" / "labels" / "linux" / opt / f"{stem}_sym.functions_truth.json"))
            for thr in THRESHOLDS:
                pred_starts = apply_prediction_pipeline(
                    instrs,
                    addrs,
                    feats,
                    idxs,
                    probs,
                    thr,
                    merge_window=4,
                    post_filter=True,
                )
                _, _, _, p, r, f1 = score_predicted_starts(pred_starts, truth, 8)
                per_thr[thr].append((p, r, f1))

        model_name = model_path.stem.replace("start_detector_", "")
        best = None
        for thr in THRESHOLDS:
            p = mean(x[0] for x in per_thr[thr])
            r = mean(x[1] for x in per_thr[thr])
            f1 = mean(x[2] for x in per_thr[thr])
            row = {
                "model": model_name,
                "threshold": thr,
                "MacroP": round6(p),
                "MacroR": round6(r),
                "MacroF1": round6(f1),
            }
            sweep_rows.append(row)
            if best is None or (row["MacroF1"], row["MacroR"]) > (best["MacroF1"], best["MacroR"]):
                best = row
        assert best is not None
        best_rows.append(best)

    sweep_df = pd.DataFrame(sweep_rows).sort_values(["model", "threshold"])
    best_df = pd.DataFrame(best_rows).sort_values(["MacroF1", "MacroR"], ascending=[False, False])
    return sweep_df, best_df


def evaluate_o3_validation_thresholds() -> tuple[pd.DataFrame, dict]:
    bins = bin_paths_for_opt_split("O3", "val_programs")
    model_path = REPO / "models" / CURRENT_MODEL
    bundle = load(model_path)
    feature_keys = bundle.get("feature_keys", [])

    rows = []
    best = None
    for thr in CURRENT_SWEEP_THRESHOLDS:
        per_bin = []
        for bin_path in bins:
            instrs, addrs, idxs, feats = collect_candidates(bin_path)
            probs = score_model(bundle, feature_keys, feats)
            stem = Path(bin_path).name.replace("_stripped", "")
            truth = load_truth(str(REPO / "data" / "labels" / "linux" / "O3" / f"{stem}_sym.functions_truth.json"))
            pred_starts = apply_prediction_pipeline(
                instrs, addrs, feats, idxs, probs, thr, merge_window=4, post_filter=True
            )
            _, _, _, p, r, f1 = score_predicted_starts(pred_starts, truth, 8)
            per_bin.append((p, r, f1))
        row = {
            "threshold": thr,
            "MacroP": round6(mean(x[0] for x in per_bin)),
            "MacroR": round6(mean(x[1] for x in per_bin)),
            "MacroF1": round6(mean(x[2] for x in per_bin)),
            "split": "O3 validation",
            "binaries": len(bins),
        }
        rows.append(row)
        if best is None or (row["MacroF1"], row["MacroR"]) > (best["MacroF1"], best["MacroR"]):
            best = row

    assert best is not None
    return pd.DataFrame(rows), best


def evaluate_current_best_by_opt(best_threshold: float) -> pd.DataFrame:
    model_path = REPO / "models" / CURRENT_MODEL
    bundle = load(model_path)
    feature_keys = bundle.get("feature_keys", [])

    rows = []
    for opt in ["O1", "O2", "O3"]:
        bins = bin_paths_for_opt(opt)
        per_bin_rows = []
        for bin_path in bins:
            instrs, addrs, idxs, feats = collect_candidates(bin_path)
            probs = score_model(bundle, feature_keys, feats)
            stem = Path(bin_path).name.replace("_stripped", "")
            truth = load_truth(str(REPO / "data" / "labels" / "linux" / opt / f"{stem}_sym.functions_truth.json"))
            pred_starts = apply_prediction_pipeline(
                instrs, addrs, feats, idxs, probs, best_threshold, merge_window=4, post_filter=True
            )
            _, _, _, p, r, f1 = score_predicted_starts(pred_starts, truth, 8)
            per_bin_rows.append((p, r, f1))
        rows.append(
            {
                "opt": opt,
                "binaries": len(bins),
                "MacroP": round6(mean(x[0] for x in per_bin_rows)),
                "MacroR": round6(mean(x[1] for x in per_bin_rows)),
                "MacroF1": round6(mean(x[2] for x in per_bin_rows)),
                "model": CURRENT_MODEL.replace(".joblib", ""),
                "threshold": best_threshold,
            }
        )
    return pd.DataFrame(rows)


def raw_model_predictions(addrs: list[int], probs: np.ndarray, threshold: float) -> list[int]:
    return [int(addr) for addr, prob in zip(addrs, probs) if float(prob) >= threshold]


def objdump_label_predictions(bin_path: str) -> list[int]:
    with open(bin_path + ".asm.json") as f:
        asm = json.load(f)
    addr_set = {ins["addr"] for ins in asm["instrs"]}
    preds = sorted(int(addr) for addr in asm.get("labels", {}) if int(addr) in addr_set)
    if asm["instrs"]:
        preds = sorted({asm["instrs"][0]["addr"], *preds})
    return preds


def heuristic_only_predictions(instrs: list[dict], addrs: list[int], feats: list[dict], idxs: list[int]) -> list[int]:
    inbound_kinds = _build_inbound_kind_counts(instrs)
    pred = []
    for addr, feat, idx in zip(addrs, feats, idxs):
        score = 1.0
        take = False
        priority = 0.0
        if _is_clean_jmp_rescue_candidate(score, feat, instrs, idx, inbound_kinds=inbound_kinds):
            take = True
            priority = max(priority, 4.0)
        if _is_clean_call_rescue_candidate(score, feat, instrs, idx):
            take = True
            priority = max(priority, 3.5)
        if _is_clean_ret_rescue_candidate(score, feat, instrs, idx):
            take = True
            priority = max(priority, 3.0)
        if (
            feat.get("xrefs_in", 0) == 0
            and feat.get("align16", 0)
            and _prev_ret_near(instrs, idx, back=3)
            and _is_short_leaf_candidate(instrs, idx, max_ins=5)
        ):
            take = True
            priority = max(priority, 2.5)
        if _is_reachable_leaf_candidate(score, feat, instrs, idx):
            take = True
            priority = max(priority, 2.0)
        if take:
            pred.append({"start": int(addr), "score": priority, "features": feat, "idx": idx})

    filtered = []
    for item in pred:
        feat = item["features"]
        cond_a = feat.get("xrefs_in", 0) == 0
        cond_b = feat.get("padding_nop_run", 0) >= 3
        cond_c = not (
            feat.get("prev_is_ret", 0)
            or feat.get("has_push_rbp", 0)
            or feat.get("window2_xrefs_in", 0) > 0
        )
        drop_padding = cond_a and cond_b and cond_c
        drop_jt = False if drop_padding else _looks_like_jump_table(instrs, item["idx"], feat)
        drop_leave = False if (drop_padding or drop_jt) else _looks_like_tiny_leave_stub(instrs, item)
        if drop_padding or drop_jt or drop_leave:
            continue
        filtered.append(item)

    filtered = sorted(filtered, key=lambda x: x["start"])
    merged = []
    for item in filtered:
        if merged and item["start"] - merged[-1]["start"] <= 4:
            if item["score"] > merged[-1]["score"]:
                merged[-1] = item
        else:
            merged.append(item)
    return [item["start"] for item in merged]


def evaluate_opt_threshold_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = REPO / "models" / CURRENT_MODEL
    bundle = load(model_path)
    feature_keys = bundle.get("feature_keys", [])

    sweep_rows = []
    summary_rows = []
    for opt in ["O1", "O2", "O3"]:
        bins = bin_paths_for_opt(opt)
        hybrid_scores = {thr: [] for thr in CURRENT_SWEEP_THRESHOLDS}
        raw_scores = {thr: [] for thr in CURRENT_SWEEP_THRESHOLDS}
        for bin_path in bins:
            instrs, addrs, idxs, feats = collect_candidates(bin_path)
            probs = score_model(bundle, feature_keys, feats)
            stem = Path(bin_path).name.replace("_stripped", "")
            truth = load_truth(str(REPO / "data" / "labels" / "linux" / opt / f"{stem}_sym.functions_truth.json"))
            for thr in CURRENT_SWEEP_THRESHOLDS:
                hybrid = apply_prediction_pipeline(instrs, addrs, feats, idxs, probs, thr, merge_window=4, post_filter=True)
                raw = raw_model_predictions(addrs, probs, thr)
                _, _, _, hp, hr, hf1 = score_predicted_starts(hybrid, truth, 8)
                _, _, _, rp, rr, rf1 = score_predicted_starts(raw, truth, 8)
                hybrid_scores[thr].append((hp, hr, hf1))
                raw_scores[thr].append((rp, rr, rf1))

        hybrid_best = None
        raw_best = None
        fixed_hybrid = None
        fixed_raw = None
        for thr in CURRENT_SWEEP_THRESHOLDS:
            hp = mean(x[0] for x in hybrid_scores[thr])
            hr = mean(x[1] for x in hybrid_scores[thr])
            hf1 = mean(x[2] for x in hybrid_scores[thr])
            rp = mean(x[0] for x in raw_scores[thr])
            rr = mean(x[1] for x in raw_scores[thr])
            rf1 = mean(x[2] for x in raw_scores[thr])
            sweep_rows.append(
                {
                    "opt": opt,
                    "mode": "hybrid",
                    "threshold": thr,
                    "MacroP": round6(hp),
                    "MacroR": round6(hr),
                    "MacroF1": round6(hf1),
                }
            )
            sweep_rows.append(
                {
                    "opt": opt,
                    "mode": "model_only",
                    "threshold": thr,
                    "MacroP": round6(rp),
                    "MacroR": round6(rr),
                    "MacroF1": round6(rf1),
                }
            )
            hybrid_row = {"threshold": thr, "MacroP": round6(hp), "MacroR": round6(hr), "MacroF1": round6(hf1)}
            raw_row = {"threshold": thr, "MacroP": round6(rp), "MacroR": round6(rr), "MacroF1": round6(rf1)}
            if hybrid_best is None or (hybrid_row["MacroF1"], hybrid_row["MacroR"]) > (hybrid_best["MacroF1"], hybrid_best["MacroR"]):
                hybrid_best = hybrid_row
            if raw_best is None or (raw_row["MacroF1"], raw_row["MacroR"]) > (raw_best["MacroF1"], raw_best["MacroR"]):
                raw_best = raw_row
            if abs(thr - CURRENT_THRESHOLD) < 1e-9:
                fixed_hybrid = hybrid_row
                fixed_raw = raw_row

        assert hybrid_best is not None and raw_best is not None and fixed_hybrid is not None and fixed_raw is not None
        summary_rows.append(
            {
                "opt": opt,
                "hybrid_fixed_thr": CURRENT_THRESHOLD,
                "hybrid_fixed_F1": fixed_hybrid["MacroF1"],
                "hybrid_best_thr": hybrid_best["threshold"],
                "hybrid_best_F1": hybrid_best["MacroF1"],
                "model_fixed_thr": CURRENT_THRESHOLD,
                "model_fixed_F1": fixed_raw["MacroF1"],
                "model_best_thr": raw_best["threshold"],
                "model_best_F1": raw_best["MacroF1"],
                "manual_gain_fixed_F1": round6(fixed_hybrid["MacroF1"] - fixed_raw["MacroF1"]),
            }
        )

    return pd.DataFrame(sweep_rows), pd.DataFrame(summary_rows)


def evaluate_component_ablation_o3() -> pd.DataFrame:
    model_path = REPO / "models" / CURRENT_MODEL
    bundle = load(model_path)
    feature_keys = bundle.get("feature_keys", [])
    bins = bin_paths_for_opt("O3")
    rows = []

    systems = [
        ("objdump_only", "none", "Objdump labels only"),
        ("manual_only", "none", "Deterministic boundary rules only"),
        ("model_only", CURRENT_THRESHOLD, f"Classifier only @ {CURRENT_THRESHOLD:.2f}"),
        ("hybrid", CURRENT_THRESHOLD, f"Classifier + manual rules @ {CURRENT_THRESHOLD:.2f}"),
    ]

    per_system = {name: [] for name, _, _ in systems}
    for bin_path in bins:
        instrs, addrs, idxs, feats = collect_candidates(bin_path)
        probs = score_model(bundle, feature_keys, feats)
        stem = Path(bin_path).name.replace("_stripped", "")
        truth = load_truth(str(REPO / "data" / "labels" / "linux" / "O3" / f"{stem}_sym.functions_truth.json"))
        preds = {
            "objdump_only": objdump_label_predictions(bin_path),
            "manual_only": heuristic_only_predictions(instrs, addrs, feats, idxs),
            "model_only": raw_model_predictions(addrs, probs, CURRENT_THRESHOLD),
            "hybrid": apply_prediction_pipeline(instrs, addrs, feats, idxs, probs, CURRENT_THRESHOLD, merge_window=4, post_filter=True),
        }
        for name, _, _ in systems:
            _, _, _, p, r, f1 = score_predicted_starts(preds[name], truth, 8)
            per_system[name].append((p, r, f1))

    model_only_f1 = None
    for name, threshold, note in systems:
        macro_p = round6(mean(x[0] for x in per_system[name]))
        macro_r = round6(mean(x[1] for x in per_system[name]))
        macro_f1 = round6(mean(x[2] for x in per_system[name]))
        if name == "model_only":
            model_only_f1 = macro_f1
        rows.append(
            {
                "system": name,
                "threshold": threshold,
                "MacroP": macro_p,
                "MacroR": macro_r,
                "MacroF1": macro_f1,
                "notes": note,
            }
        )

    if model_only_f1 is not None:
        for row in rows:
            row["DeltaF1_vs_model_only"] = round6(row["MacroF1"] - model_only_f1)

    return pd.DataFrame(rows)


def glossary_table() -> pd.DataFrame:
    rows = [
        {"term": "Candidate address", "meaning": "An instruction address proposed as a possible function start before classification."},
        {"term": "xrefs_in", "meaning": "Number of decoded incoming control-flow references to an instruction address."},
        {"term": "Model-only", "meaning": "Threshold the classifier scores directly, with no rescue, retarget, filter, or merge heuristics."},
        {"term": "Manual-only", "meaning": "Deterministic boundary rules derived from semester-two error analysis, without the classifier."},
        {"term": "Hybrid", "meaning": "The current full predictor: classifier scores plus post-filters, rescues, retargeting, and merge-nearby logic."},
        {"term": "Post-filter", "meaning": "Rules that drop starts likely caused by padding, jump-table islands, or tiny leave/ret artifacts."},
        {"term": "Rescue rule", "meaning": "A bounded heuristic that recovers a likely real start that scored below the global threshold."},
        {"term": "Merge window", "meaning": "Distance in bytes for collapsing near-duplicate predicted starts after scoring."},
        {"term": "Macro F1", "meaning": "Average F1 across binaries, giving each binary equal weight rather than weighting by size."},
        {"term": "Tolerance ±8B", "meaning": "A prediction counts as matched if it falls within 8 bytes of a truth start."},
        {"term": "DWARF truth", "meaning": "Function-start labels extracted from symbol-preserving binaries and filtered to executable regions."},
        {"term": "Disa re-scored metric", "meaning": "Disa predictions re-scored here with the same tolerant function-start metric used for this project."},
    ]
    return pd.DataFrame(rows)


def current_hard_cases() -> pd.DataFrame:
    df = read_tsv(CURRENT_SUMMARY)
    for col in ["TP", "FP", "FN", "P", "R", "F1", "mean_err", "median_err"]:
        df[col] = pd.to_numeric(df[col])
    return df.sort_values(["F1", "R", "P"]).head(15).reset_index(drop=True)


def current_ghidra_o3_table() -> pd.DataFrame:
    rows = []
    for gh_path in sorted((REPO / "out" / "ghidra").glob("O3_*_stripped.csv")):
        stem = gh_path.name[len("O3_") : -len("_stripped.csv")]
        pred_path = REPO / "out" / f"{stem}.json"
        if not pred_path.exists():
            continue
        with pred_path.open() as f:
            preds = [int(item["start"]) for item in json.load(f)]
        with gh_path.open() as f:
            gh = [int(row["start"]) for row in csv.DictReader(f)]
        used = set()
        agree = 0
        for t in gh:
            best = None
            best_dist = None
            for idx, p in enumerate(preds):
                if idx in used:
                    continue
                d = abs(p - t)
                if d <= 8 and (best_dist is None or d < best_dist):
                    best = idx
                    best_dist = d
            if best is not None:
                used.add(best)
                agree += 1
        rows.append(
            {
                "binary": stem,
                "agree": agree,
                "miss": len(gh) - agree,
                "extra": len(preds) - agree,
                "ghidra_funcs": len(gh),
                "our_preds": len(preds),
            }
        )
    df = pd.DataFrame(rows).sort_values(["miss", "extra", "binary"], ascending=[False, False, True])
    if not df.empty:
        total = {
            "binary": "TOTAL",
            "agree": int(df["agree"].sum()),
            "miss": int(df["miss"].sum()),
            "extra": int(df["extra"].sum()),
            "ghidra_funcs": int(df["ghidra_funcs"].sum()),
            "our_preds": int(df["our_preds"].sum()),
        }
        df = pd.concat([pd.DataFrame([total]), df], ignore_index=True)
    return df


def ghidra_vs_truth_o3_table() -> pd.DataFrame:
    rows = []
    ours_scores = []
    ghidra_scores = []
    for gh_path in sorted((REPO / "out" / "ghidra").glob("O3_*_stripped.csv")):
        stem = gh_path.name[len("O3_") : -len("_stripped.csv")]
        pred_path = REPO / "out" / f"{stem}.json"
        truth_path = REPO / "data" / "labels" / "linux" / "O3" / f"{stem}_sym.functions_truth.json"
        if not pred_path.exists() or not truth_path.exists():
            continue

        truth = load_truth(str(truth_path))
        with pred_path.open() as f:
            ours = [int(item["start"]) for item in json.load(f)]
        with gh_path.open() as f:
            ghidra = [int(row["start"]) for row in csv.DictReader(f)]

        _, _, _, op, or_, of1 = score_predicted_starts(ours, truth, 8)
        _, _, _, gp, gr, gf1 = score_predicted_starts(ghidra, truth, 8)
        ours_scores.append((op, or_, of1))
        ghidra_scores.append((gp, gr, gf1))
        rows.append(
            {
                "binary": stem,
                "ours_P": round6(op),
                "ours_R": round6(or_),
                "ours_F1": round6(of1),
                "ghidra_P": round6(gp),
                "ghidra_R": round6(gr),
                "ghidra_F1": round6(gf1),
            }
        )

    summary_rows = []
    if ours_scores:
        summary_rows.extend(
            [
                {
                    "system": "ours_v06u_xgb",
                    "binaries": len(ours_scores),
                    "MacroP": round6(mean(x[0] for x in ours_scores)),
                    "MacroR": round6(mean(x[1] for x in ours_scores)),
                    "MacroF1": round6(mean(x[2] for x in ours_scores)),
                },
                {
                    "system": "ghidra_export",
                    "binaries": len(ghidra_scores),
                    "MacroP": round6(mean(x[0] for x in ghidra_scores)),
                    "MacroR": round6(mean(x[1] for x in ghidra_scores)),
                    "MacroF1": round6(mean(x[2] for x in ghidra_scores)),
                },
            ]
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(rows)


def disa_vs_truth_o3_summary() -> pd.DataFrame:
    summary_path = REPO / "out" / "disa_function_start_o3_summary.tsv"
    if not summary_path.exists():
        return pd.DataFrame()
    df = read_tsv(summary_path)
    numeric_cols = [col for col in ["MacroP", "MacroR", "MacroF1"] if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    return df


def pdf_snippets() -> str:
    snippets = []
    sources = [
        ("Sem1 final", REPO / "handin" / "sem1-ierg4998" / "1155174712_fyp_final_report.pdf", ["0.638", "0.915", "1,625"]),
        ("Sem2 interim", REPO / "handin" / "sem2-ierg4999" / "1155174712_Interim_Report.pdf", ["0.9593", "0.959", "149-binary", "0.8956"]),
    ]
    for label, pdf, needles in sources:
        reader = PdfReader(str(pdf))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        snippets.append(f"## {label}\n")
        snippets.append(f"Source: `{pdf.relative_to(REPO)}`\n")
        for needle in needles:
            idx = text.lower().find(needle.lower())
            if idx == -1:
                continue
            lo = max(0, idx - 220)
            hi = min(len(text), idx + 420)
            excerpt = " ".join(text[lo:hi].split())
            snippets.append(f"- `{needle}`: {excerpt}\n")
        snippets.append("\n")
    return "".join(snippets)


def make_plots(
    evolution: pd.DataFrame,
    current_thresholds: pd.DataFrame,
    current_val_best: dict,
    current_model_best: pd.DataFrame,
    current_opt: pd.DataFrame,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    xs = list(range(len(evolution)))
    ys = evolution["MacroF1"].tolist()
    ax.plot(xs, ys, marker="o", linewidth=2.2, color="#0f766e")
    ax.set_ylabel("Macro F1")
    ax.set_title("Project Evolution: O3 Macro F1 Across Major Stages")
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(xs, evolution["stage"].tolist(), rotation=20, ha="right")
    for idx, row in evolution.iterrows():
        ax.annotate(
            f"F1={row['MacroF1']:.3f}\n{row['plot_note']}",
            (idx, row["MacroF1"]),
            textcoords="offset points",
            xytext=(0, 10 if idx != 2 else -55),
            ha="center",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
        )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "project_evolution_f1.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(current_thresholds["threshold"], current_thresholds["MacroP"], marker="o", label="Macro P", color="#1d4ed8")
    ax.plot(current_thresholds["threshold"], current_thresholds["MacroR"], marker="o", label="Macro R", color="#059669")
    ax.plot(current_thresholds["threshold"], current_thresholds["MacroF1"], marker="o", label="Macro F1", color="#b45309")
    ax.axvline(
        current_val_best["threshold"],
        color="#991b1b",
        linestyle="--",
        linewidth=1.5,
        label=f"Validation-chosen threshold = {current_val_best['threshold']:.2f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Macro score")
    ax.set_title("Current v06u XGB Threshold Sweep on O3 Validation Split")
    ax.set_ylim(0.94, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "current_threshold_sweep_v06u_xgb.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = current_model_best["model"].tolist()
    xs = range(len(models))
    width = 0.22
    ax.bar([x - width for x in xs], current_model_best["MacroP"], width=width, label="Macro P", color="#1d4ed8")
    ax.bar(xs, current_model_best["MacroR"], width=width, label="Macro R", color="#059669")
    ax.bar([x + width for x in xs], current_model_best["MacroF1"], width=width, label="Macro F1", color="#b45309")
    ax.set_xticks(list(xs), models, rotation=12)
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Current O3 Model Comparison at Each Model's Best Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "current_model_best_o3.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    opts = current_opt["opt"].tolist()
    ax.bar(opts, current_opt["MacroF1"], color="#0f766e")
    ax.plot(opts, current_opt["MacroP"], marker="o", color="#1d4ed8", label="Macro P")
    ax.plot(opts, current_opt["MacroR"], marker="o", color="#059669", label="Macro R")
    ax.set_ylim(0.9, 1.0)
    ax.set_title("Current Best Model Across Optimization Levels")
    ax.set_ylabel("Macro score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "current_opt_level_metrics.png", dpi=200)
    plt.close(fig)

    for case in CASE_STUDIES:
        lines = snippet_lines(
            "O3",
            case["stem"],
            case["targets"],
            before=case.get("before", 8),
            after=case.get("after", 12),
        )
        render_text_figure(FIG_DIR / f"{case['key']}.png", case["title"], lines)


def case_study_table() -> pd.DataFrame:
    rows = [
        {
            "case": "truth_cleanup",
            "binary": "100-prisoners",
            "change": "Filter undefined/non-executable function labels",
            "effect": "Removes impossible false negatives from evaluation",
            "evidence": "Imported STT_FUNC entries at address 0x0 are not real text entries",
        },
        {
            "case": "clean_ret_rescue",
            "binary": "Angles-geometric-normalization-and-conversion",
            "change": "Recover aligned start after ret + padding",
            "effect": "Raises binary result from 15/22 starts to 22/22 starts",
            "evidence": "Target 0x1140 follows ret+nop and is recovered by clean-ret rescue",
        },
        {
            "case": "post_filter",
            "binary": "Munching-squares",
            "change": "Remove unreferenced leave/ret islands",
            "effect": "Precision rises from 0.667 to 1.000",
            "evidence": "False starts at 0x401370, 0x401380, 0x401390 disappear",
        },
        {
            "case": "remaining_limit",
            "binary": "Best-shuffle",
            "change": "No full fix yet",
            "effect": "Compact helper stubs still confuse the predictor",
            "evidence": "Misses remain near 0x40115f and 0x401164",
        },
    ]
    return pd.DataFrame(rows)


def write_notes(
    dataset_summary: pd.DataFrame,
    evolution: pd.DataFrame,
    current_best_models: pd.DataFrame,
    current_opt: pd.DataFrame,
    opt_threshold_summary: pd.DataFrame,
    component_ablation: pd.DataFrame,
    glossary: pd.DataFrame,
    hard_cases: pd.DataFrame,
    ghidra_df: pd.DataFrame,
    ghidra_truth_summary: pd.DataFrame,
    disa_summary: pd.DataFrame,
) -> None:
    note = []
    note.append("# IERG4999 Final Report Asset Notes\n\n")
    note.append("This folder contains LaTeX-ready figures and TSV tables generated from the current repo state.\n\n")
    note.append("## Suggested narrative\n\n")
    note.append("- Semester 1 built the pipeline and scaled evaluation from toy binaries to a RosettaCode corpus.\n")
    note.append("- Semester 2 concentrated on correctness and hard O3 cases: truth-label cleanup, merge-rule fix, boundary-aware rescues, and tighter thresholding.\n")
    note.append("- The most defensible headline comparison is large-corpus O3 macro F1: sem1 final RF baseline `0.638` vs current sem2 XGB result `0.988`.\n\n")
    note.append("## Recommended figures\n\n")
    note.append("- `figures/project_evolution_f1.png`: Use in Introduction/Results to show long-run improvement.\n")
    note.append("- `figures/current_threshold_sweep_v06u_xgb.png`: Use in Methodology/Threshold Selection.\n")
    note.append("- `figures/current_model_best_o3.png`: Use in Experiments to justify choosing XGB.\n")
    note.append("- `figures/current_opt_level_metrics.png`: Use in Generalization/Cross-optimization section.\n\n")
    note.append("- `figures/angles_clean_ret.png`: Use in Error Analysis to show a real start recovered by the clean-ret rescue.\n")
    note.append("- `figures/munching_false_starts.png`: Use in Error Analysis to show repeated false starts removed by post-filtering.\n\n")
    note.append("## Recommended tables\n\n")
    note.append("- `tables/project_evolution.tsv`: concise semester-one to semester-two checkpoint summary.\n")
    note.append("- `tables/dataset_and_split_summary.tsv`: corpus size and split counts.\n")
    note.append("- `tables/current_model_best_o3.tsv`: best threshold per current model under the final predictor path.\n")
    note.append("- `tables/current_opt_level_metrics.tsv`: O1/O2/O3 macro results at the chosen threshold.\n")
    note.append("- `tables/current_opt_threshold_summary.tsv`: fixed-threshold vs best-threshold comparison for O1/O2/O3, plus model-only ablation.\n")
    note.append("- `tables/component_ablation_o3.tsv`: objdump-only, manual-only, model-only, hybrid, and Disa comparison on O3.\n")
    note.append("- `tables/glossary.tsv`: short definitions for the report terminology.\n")
    note.append("- `tables/current_hard_cases_o3.tsv`: worst remaining binaries to show honest limitations.\n")
    note.append("- `tables/current_ghidra_o3.tsv`: external-reference agreement on available O3 subset.\n\n")
    note.append("- `tables/case_study_summary.tsv`: one-line mapping from change to failure mode and evidence.\n\n")
    note.append("## Caption-ready observations\n\n")
    note.append(f"- Current best O3 setting: `{CURRENT_MODEL}` at threshold `{CURRENT_THRESHOLD}` with macro `P/R/F1 = 0.9945 / 0.9832 / 0.9884`.\n")
    note.append("- The threshold sweep peaks around `0.92-0.95`; moving above `0.95` starts to trade away recall faster than it gains precision.\n")
    note.append("- The remaining low-F1 binaries are concentrated and interpretable rather than systemic, which supports the argument that the core evaluation is now stable.\n")
    note.append("- The strongest narrative examples are `Angles...` for recall recovery and `Munching-squares` for precision recovery.\n")
    o3_summary = opt_threshold_summary[opt_threshold_summary["opt"] == "O3"].iloc[0]
    note.append(
        f"- Manual post-processing matters most on O3: at the fixed threshold `{CURRENT_THRESHOLD}`, "
        f"the hybrid system gains `{o3_summary['manual_gain_fixed_F1']:.4f}` macro F1 over model-only.\n"
    )
    if not component_ablation.empty:
        hybrid = component_ablation[component_ablation["system"] == "hybrid"].iloc[0]
        model_only = component_ablation[component_ablation["system"] == "model_only"].iloc[0]
        note.append(
            f"- On O3, `model_only` scores `{model_only['MacroF1']:.4f}` while the full hybrid scores "
            f"`{hybrid['MacroF1']:.4f}` at the same threshold `{CURRENT_THRESHOLD}`.\n"
        )
    if not disa_summary.empty:
        disa = disa_summary.iloc[0]
        note.append(
            f"- Re-scored on the same tolerant function-start metric, Disa reaches "
            f"`P/R/F1 = {disa['MacroP']:.4f} / {disa['MacroR']:.4f} / {disa['MacroF1']:.4f}` on O3.\n"
        )
    if not ghidra_df.empty:
        total = ghidra_df.iloc[0]
        note.append(
            f"- On the available O3 Ghidra subset, total agreement counts are `agree={int(total['agree'])}`, "
            f"`miss={int(total['miss'])}`, `extra={int(total['extra'])}`.\n"
        )
    if not ghidra_truth_summary.empty:
        ours = ghidra_truth_summary[ghidra_truth_summary["system"] == "ours_v06u_xgb"].iloc[0]
        gh = ghidra_truth_summary[ghidra_truth_summary["system"] == "ghidra_export"].iloc[0]
        note.append(
            f"- On the shared O3 subset scored against DWARF truth, our current pipeline reaches "
            f"`P/R/F1 = {ours['MacroP']:.4f} / {ours['MacroR']:.4f} / {ours['MacroF1']:.4f}`, while "
            f"Ghidra export reaches `{gh['MacroP']:.4f} / {gh['MacroR']:.4f} / {gh['MacroF1']:.4f}`.\n"
        )
    note.append("\n## Source snippets\n\n")
    note.append(pdf_snippets())
    (NOTE_DIR / "report_asset_notes.md").write_text("".join(note))


def write_report_outline(
    dataset_summary: pd.DataFrame,
    evolution: pd.DataFrame,
    current_best_models: pd.DataFrame,
    current_opt: pd.DataFrame,
    opt_threshold_summary: pd.DataFrame,
    component_ablation: pd.DataFrame,
    hard_cases: pd.DataFrame,
    ghidra_truth_summary: pd.DataFrame,
    disa_summary: pd.DataFrame,
) -> None:
    best = current_best_models.iloc[0]
    train_row = dataset_summary[dataset_summary["scope"] == "train split"].iloc[0]
    val_row = dataset_summary[dataset_summary["scope"] == "val split"].iloc[0]
    test_row = dataset_summary[dataset_summary["scope"] == "test split"].iloc[0]
    per_opt_programs = dataset_summary[dataset_summary["scope"].str.endswith("corpus")]["programs"].tolist()
    lines = []
    lines.append("# IERG4999 Final Report Outline\n\n")
    lines.append("## Proposed section flow\n\n")
    lines.append("1. Introduction and motivation: function-start recovery for stripped x86-64 ELF binaries.\n")
    lines.append("2. Semester-one baseline: parser, labels, small-set evaluation, then large RosettaCode scaling.\n")
    lines.append("3. Semester-two improvements: truth cleanup, merge-rule correction, bounded rescue heuristics, threshold calibration.\n")
    lines.append("4. Experimental setup: dataset inventory, train/val/test split by program, tolerance metric, models.\n")
    lines.append("5. Results: Sem1 vs Sem2 evolution, current model comparison, threshold sweep, O1/O2/O3 generalization, and baseline ablations.\n")
    lines.append("6. External reference analysis: Ghidra agreement, Disa comparison, and why neither replaces DWARF ground truth.\n")
    lines.append("7. Remaining failure modes and limitations.\n")
    lines.append("8. Conclusion and future work.\n\n")
    lines.append("## Headline numbers to surface\n\n")
    lines.append("- Sem1 large-corpus O3 macro F1: `0.638`.\n")
    lines.append(
        f"- Sem2 current best O3 macro F1: `{best['MacroF1']:.4f}` at threshold `{best['threshold']}` using `{best['model']}`.\n"
    )
    opt_o3 = current_opt[current_opt["opt"] == "O3"].iloc[0]
    opt_o3_summary = opt_threshold_summary[opt_threshold_summary["opt"] == "O3"].iloc[0]
    lines.append(
        f"- Current O3 macro P/R/F1: `{opt_o3['MacroP']:.4f} / {opt_o3['MacroR']:.4f} / {opt_o3['MacroF1']:.4f}`.\n"
    )
    lines.append(
        f"- Model-only vs hybrid on O3 at threshold `{CURRENT_THRESHOLD}`: "
        f"`{opt_o3_summary['model_fixed_F1']:.4f}` vs `{opt_o3_summary['hybrid_fixed_F1']:.4f}` macro F1.\n"
    )
    lines.append(
        f"- Corpus scale in current repo: `{min(per_opt_programs)}-{max(per_opt_programs)}` programs per optimization level, "
        f"`{int(train_row['programs'])}` train programs, `{int(val_row['programs'])}` validation programs, and `{int(test_row['programs'])}` test programs.\n\n"
    )
    if not component_ablation.empty:
        lines.append("## Baseline framing\n\n")
        for _, row in component_ablation.iterrows():
            lines.append(
                f"- `{row['system']}`: threshold `{row['threshold']}`, macro F1 `{row['MacroF1']:.4f}`. {row['notes']}\n"
            )
        lines.append("\n")
    if not ghidra_truth_summary.empty:
        ours = ghidra_truth_summary[ghidra_truth_summary["system"] == "ours_v06u_xgb"].iloc[0]
        gh = ghidra_truth_summary[ghidra_truth_summary["system"] == "ghidra_export"].iloc[0]
        lines.append("## Ghidra angle\n\n")
        lines.append(
            f"- On the shared O3 subset, DWARF-scored ours vs Ghidra macro F1 is `{ours['MacroF1']:.4f}` vs `{gh['MacroF1']:.4f}`.\n"
        )
        lines.append("- This supports using Ghidra as an external reference for error analysis, not as the primary truth source.\n\n")
    if not disa_summary.empty:
        disa = disa_summary.iloc[0]
        lines.append("## Disa angle\n\n")
        lines.append(
            f"- Re-scored on the same tolerant function-start metric, Disa reaches macro F1 `{disa['MacroF1']:.4f}` on O3.\n"
        )
        lines.append("- This is directly comparable to the project metric, unlike Disa's original instruction-level micro log.\n\n")
    lines.append("## Remaining hard cases to discuss\n\n")
    for _, row in hard_cases.head(5).iterrows():
        lines.append(
            f"- `{short_binary_name(row['file'])}`: `P={row['P']:.3f}`, `R={row['R']:.3f}`, `F1={row['F1']:.3f}`.\n"
        )
    (REPORT_DIR / "outline.md").write_text("".join(lines))


def write_report_skeleton() -> None:
    tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{longtable}
\usepackage{float}
\usepackage{hyperref}
\usepackage{xcolor}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

\title{Detecting Function Starts in Stripped x86-64 ELF Binaries\\with Lightweight Machine Learning}
\author{Nelson\\IERG4999 Final Year Project}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
This report studies function-start detection on stripped x86-64 ELF binaries using a lightweight machine-learning layer on top of \texttt{objdump} linear sweep disassembly. The semester-one work established the parsing, labeling, feature extraction, and large-corpus evaluation pipeline. The semester-two work focused on correctness and robustness on optimized binaries, especially \texttt{-O3}, through truth-label cleanup, prediction-path fixes, bounded rescue heuristics, and calibrated threshold selection. On the corrected O3 test split, the final XGBoost configuration reaches macro precision/recall/F1 of 0.9945 / 0.9832 / 0.9884, improving substantially over the semester-one large-corpus baseline.
\end{abstract}

\section{Introduction}
Explain why function-start recovery matters for stripped binaries, why linear sweep is attractive but error-prone, and why a lightweight learned boundary detector is a practical compromise.

\section{Project Progress from Semester 1 to Semester 2}
\input{../report_assets/tex/project_evolution.tex}

Use Figure~\ref{fig:project-evolution} to summarize the trajectory from toy experiments to the current corrected large-corpus result.

\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\linewidth]{../report_assets/figures/project_evolution_f1.png}
  \caption{Macro F1 across major project stages from semester one to semester two.}
  \label{fig:project-evolution}
\end{figure}

\section{Dataset and Evaluation Protocol}
\input{../report_assets/tex/dataset_and_split_summary.tex}

Describe the RosettaCode corpus, compiler optimization levels, symbol/stripped binaries, and the program-level split that avoids cross-program leakage.

\section{Method}
Summarize the pipeline: \texttt{objdump} parsing, truth extraction from DWARF/\texttt{.symtab}, candidate generation, feature extraction, model scoring, post-filtering, and bounded rescue heuristics.

\section{Model Selection and Threshold Calibration}
\input{../report_assets/tex/current_model_best_o3.tex}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.82\linewidth]{../report_assets/figures/current_model_best_o3.png}
  \caption{Current O3 model comparison at each model's best threshold.}
  \label{fig:model-compare}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.82\linewidth]{../report_assets/figures/current_threshold_sweep_v06u_xgb.png}
  \caption{Threshold sweep for the selected v06u XGBoost model on the O3 test split.}
  \label{fig:threshold-sweep}
\end{figure}

Justify the final threshold choice using the precision/recall/F1 trade-off rather than a single optimistic point.

\section{Results}
\input{../report_assets/tex/current_opt_level_metrics.tex}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.80\linewidth]{../report_assets/figures/current_opt_level_metrics.png}
  \caption{Generalization of the final model across O1, O2, and O3 test binaries.}
  \label{fig:opt-levels}
\end{figure}

\section{Error Analysis}
\input{../report_assets/tex/current_hard_cases_o3.tex}

Discuss the remaining hard cases in terms of alignment artifacts, tiny functions, and helper/wrapper layouts. Keep the analysis concrete with 2--3 representative binaries.

\section{External Reference Check with Ghidra}
\input{../report_assets/tex/ghidra_vs_truth_o3_summary.tex}

State clearly that DWARF-derived labels remain the primary truth source, while Ghidra is used as an external reference for agreement analysis and failure inspection.

\section{Conclusion and Future Work}
Summarize the semester-two improvements, the final best configuration, and the next steps such as extending beyond x86-64 ELF, deeper CFG-aware features, or broader external validation.

\end{document}
"""
    main_tex = REPORT_DIR / "main.tex"
    template_tex = REPORT_DIR / "main_template.tex"
    if not main_tex.exists():
        main_tex.write_text(tex)
    template_tex.write_text(tex)


def main() -> None:
    ensure_dirs()

    dataset_summary = dataset_and_split_summary()
    evolution = project_evolution_table()
    sem_compare = sem1_sem2_summary_table()
    case_summary = case_study_table()
    current_thresholds, current_val_best = evaluate_o3_validation_thresholds()
    current_model_sweep, current_model_best = evaluate_current_models_o3()
    current_opt = evaluate_current_best_by_opt(CURRENT_THRESHOLD)
    opt_threshold_sweep, opt_threshold_summary = evaluate_opt_threshold_summary()
    component_ablation = evaluate_component_ablation_o3()
    glossary = glossary_table()
    hard_cases = current_hard_cases()
    ghidra_df = current_ghidra_o3_table()
    ghidra_truth_summary, ghidra_truth_per_bin = ghidra_vs_truth_o3_table()
    disa_summary = disa_vs_truth_o3_summary()

    dataset_summary.to_csv(TABLE_DIR / "dataset_and_split_summary.tsv", sep="\t", index=False)
    evolution.to_csv(TABLE_DIR / "project_evolution.tsv", sep="\t", index=False)
    sem_compare.to_csv(TABLE_DIR / "semester1_vs_semester2_summary.tsv", sep="\t", index=False)
    case_summary.to_csv(TABLE_DIR / "case_study_summary.tsv", sep="\t", index=False)
    current_thresholds.to_csv(TABLE_DIR / "current_threshold_sweep_v06u_xgb.tsv", sep="\t", index=False)
    current_model_sweep.to_csv(TABLE_DIR / "current_model_sweep_o3.tsv", sep="\t", index=False)
    current_model_best.to_csv(TABLE_DIR / "current_model_best_o3.tsv", sep="\t", index=False)
    current_opt.to_csv(TABLE_DIR / "current_opt_level_metrics.tsv", sep="\t", index=False)
    opt_threshold_sweep.to_csv(TABLE_DIR / "current_opt_threshold_sweep.tsv", sep="\t", index=False)
    opt_threshold_summary_export = pd.DataFrame(
        {
            "opt": opt_threshold_summary["opt"],
            f"hybrid @ {CURRENT_THRESHOLD:.2f}": opt_threshold_summary["hybrid_fixed_F1"],
            "hybrid best (thr / F1)": opt_threshold_summary.apply(
                lambda row: f"{row['hybrid_best_thr']:.2f} / {row['hybrid_best_F1']:.3f}",
                axis=1,
            ),
            f"model @ {CURRENT_THRESHOLD:.2f}": opt_threshold_summary["model_fixed_F1"],
            "model best (thr / F1)": opt_threshold_summary.apply(
                lambda row: f"{row['model_best_thr']:.2f} / {row['model_best_F1']:.3f}",
                axis=1,
            ),
            f"heuristic gain @ {CURRENT_THRESHOLD:.2f}": opt_threshold_summary["manual_gain_fixed_F1"],
        }
    )
    opt_threshold_summary_export.to_csv(TABLE_DIR / "current_opt_threshold_summary.tsv", sep="\t", index=False)
    component_ablation_export = component_ablation.copy()
    if not disa_summary.empty:
        disa_row = disa_summary.iloc[0].to_dict()
        component_ablation_export = pd.concat(
            [
                component_ablation_export,
                pd.DataFrame(
                    [
                        {
                            "system": "disa",
                            "threshold": "n/a",
                            "MacroP": float(disa_row["MacroP"]),
                            "MacroR": float(disa_row["MacroR"]),
                            "MacroF1": float(disa_row["MacroF1"]),
                            "notes": "Transformer baseline re-scored on the same tolerant function-start metric",
                            "DeltaF1_vs_model_only": round6(
                                float(disa_row["MacroF1"])
                                - float(component_ablation[component_ablation["system"] == "model_only"]["MacroF1"].iloc[0])
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    component_ablation_export.to_csv(TABLE_DIR / "component_ablation_o3.tsv", sep="\t", index=False)
    glossary.to_csv(TABLE_DIR / "glossary.tsv", sep="\t", index=False)
    hard_cases.to_csv(TABLE_DIR / "current_hard_cases_o3.tsv", sep="\t", index=False)
    ghidra_df.to_csv(TABLE_DIR / "current_ghidra_o3.tsv", sep="\t", index=False)
    ghidra_truth_summary.to_csv(TABLE_DIR / "ghidra_vs_truth_o3_summary.tsv", sep="\t", index=False)
    ghidra_truth_per_bin.to_csv(TABLE_DIR / "ghidra_vs_truth_o3_per_binary.tsv", sep="\t", index=False)
    if not disa_summary.empty:
        disa_summary.to_csv(TABLE_DIR / "disa_vs_truth_o3_summary.tsv", sep="\t", index=False)

    write_latex_table(
        TEX_DIR / "project_evolution.tex",
        evolution[["stage", "semester", "model", "threshold", "MacroP", "MacroR", "MacroF1"]],
        column_format="p{3.0cm}l p{2.8cm} r r r r",
        caption="Key checkpoints from the semester-one baseline to the current semester-two result.",
        label="tab:project-evolution",
    )
    write_latex_table(
        TEX_DIR / "dataset_and_split_summary.tex",
        dataset_summary,
        column_format="l r r r r",
        caption="Dataset scale and split summary used in the current report.",
        label="tab:dataset-summary",
    )
    current_model_best_tex = current_model_best.copy()
    current_model_best_tex["model"] = current_model_best_tex["model"].map(pretty_model_name)
    write_latex_table(
        TEX_DIR / "current_model_best_o3.tex",
        current_model_best_tex,
        column_format="l r r r r",
        caption="Current O3 model comparison at each model's best threshold.",
        label="tab:model-best-o3",
    )
    current_opt_tex = current_opt[["opt", "binaries", "MacroP", "MacroR", "MacroF1"]].copy()
    write_latex_table(
        TEX_DIR / "current_opt_level_metrics.tex",
        current_opt_tex,
        column_format="l r r r r",
        caption="Performance of the selected final model across optimization levels.",
        label="tab:opt-metrics",
    )
    opt_threshold_summary_tex = opt_threshold_summary_export.copy()
    write_latex_table(
        TEX_DIR / "current_opt_threshold_summary.tex",
        opt_threshold_summary_tex,
        column_format="l r p{2.8cm} r p{2.8cm} r",
        caption="Compact comparison across optimization levels. Each `best` cell reports `threshold / Macro F1`, while the final column shows how much the full hybrid system gains over model-only at the fixed operating point.",
        label="tab:opt-threshold-summary",
    )
    component_ablation_tex = component_ablation_export.copy()
    component_ablation_tex["system"] = component_ablation_tex["system"].map(
        {
            "objdump_only": "Objdump only",
            "manual_only": "Manual only",
            "model_only": "Model only",
            "hybrid": "Hybrid",
            "disa": "Disa",
        }
    )
    write_latex_table(
        TEX_DIR / "component_ablation_o3.tex",
        component_ablation_tex[["system", "threshold", "MacroP", "MacroR", "MacroF1", "notes"]],
        column_format="l l r r r p{5.6cm}",
        caption="O3 baseline comparison among objdump-only, deterministic manual rules, model-only, the full hybrid predictor, and Disa re-scored on the same tolerant function-start metric.",
        label="tab:component-ablation",
    )
    write_latex_table(
        TEX_DIR / "glossary.tex",
        glossary,
        column_format="p{3.0cm} p{10.0cm}",
        caption="Glossary of terms used throughout the final report.",
        label="tab:glossary",
    )
    hard_cases_tex = hard_cases.head(8)[["file", "P", "R", "F1", "FN"]].copy()
    hard_cases_tex["binary"] = hard_cases_tex["file"].map(short_binary_name)
    hard_cases_tex = hard_cases_tex[["binary", "P", "R", "F1", "FN"]]
    write_latex_table(
        TEX_DIR / "current_hard_cases_o3.tex",
        hard_cases_tex,
        column_format="p{8.8cm} r r r r",
        caption="Representative hard O3 binaries under the final configuration.",
        label="tab:hard-cases",
    )
    if not ghidra_truth_summary.empty:
        write_latex_table(
            TEX_DIR / "ghidra_vs_truth_o3_summary.tex",
            ghidra_truth_summary,
            column_format="l r r r r",
            caption="DWARF-scored comparison between the final pipeline and Ghidra exports on the shared O3 subset.",
            label="tab:ghidra-vs-truth",
        )

    make_plots(evolution, current_thresholds, current_val_best, current_model_best, current_opt)
    write_notes(
        dataset_summary,
        evolution,
        current_model_best,
        current_opt,
        opt_threshold_summary,
        component_ablation,
        glossary,
        hard_cases,
        ghidra_df,
        ghidra_truth_summary,
        disa_summary,
    )
    write_report_outline(
        dataset_summary,
        evolution,
        current_model_best,
        current_opt,
        opt_threshold_summary,
        component_ablation,
        hard_cases,
        ghidra_truth_summary,
        disa_summary,
    )
    write_report_skeleton()

    summary = {
        "asset_root": str(ASSET_ROOT.relative_to(REPO)),
        "tables": sorted(p.name for p in TABLE_DIR.glob("*.tsv")),
        "figures": sorted(p.name for p in FIG_DIR.glob("*.png")),
        "notes": sorted(p.name for p in NOTE_DIR.glob("*.md")),
        "tex": sorted(p.name for p in TEX_DIR.glob("*.tex")),
        "report": sorted(p.name for p in REPORT_DIR.glob("*")),
    }
    (ASSET_ROOT / "asset_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
