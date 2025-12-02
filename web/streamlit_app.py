#!/usr/bin/env python3
"""
Streamlit UI for the function-start detector (best model: v06b RF, macro F1~0.64 on O3).

Run:
  export PYTHONPATH=src
  source .venv/bin/activate
  streamlit run web/streamlit_app.py --server.headless true --server.port 8501
"""
import json
import pathlib
import subprocess
from functools import lru_cache

import numpy as np
import streamlit as st
from joblib import load

from features import candidate_addresses, featurize_point
from predict_starts import _looks_like_jump_table

try:
    import pandas as pd
    import altair as alt
except Exception:  # pragma: no cover - optional viz deps are installed with streamlit
    pd = None
    alt = None

MODEL_PATH = pathlib.Path("models/start_detector_v06b_rf.joblib")
DEFAULT_THRESHOLD = 0.25  # best macro F1 ~0.639 on O3 (v06b RF)
TOLERANCE = 8
BIN_LIST = pathlib.Path("out/test_bins_O3.txt")  # O3 test split list


@lru_cache(maxsize=1)
def load_model():
    bundle = load(MODEL_PATH)
    return bundle["model"], bundle.get("feature_keys", [])


def load_bins():
    if not BIN_LIST.exists():
        return []
    return [
        line.strip()
        for line in BIN_LIST.read_text().splitlines()
        if line.strip()
    ]


def ensure_asm(bin_path: pathlib.Path) -> pathlib.Path:
    asm_path = pathlib.Path(f"{bin_path}.asm.json")
    if not asm_path.exists():
        asm_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["python", "src/parse_objdump.py", "--bin", str(bin_path), "--out", str(asm_path)],
            check=True,
        )
    return asm_path


def merge_nearby(preds, window=8):
    if not preds:
        return []
    preds = sorted(preds, key=lambda x: x["start"])
    merged = [preds[0]]
    for item in preds[1:]:
        if item["start"] - merged[-1]["start"] <= window:
            if item["score"] > merged[-1]["score"]:
                merged[-1] = item
        else:
            merged.append(item)
    return merged


def apply_post_filter(preds, instrs):
    filtered = []
    removed_padding = 0
    removed_jt = 0
    for item in preds:
        feats = item["feats"]
        cond_a = feats.get("xrefs_in", 0) == 0
        cond_b = feats.get("padding_nop_run", 0) >= 3
        cond_c = not (
            feats.get("prev_is_ret", 0)
            or feats.get("has_push_rbp", 0)
            or feats.get("window2_xrefs_in", 0) > 0
        )
        drop_padding = cond_a and cond_b and cond_c
        drop_jt = False
        if not drop_padding:
            drop_jt = _looks_like_jump_table(instrs, item["idx"], feats)
        if drop_padding:
            removed_padding += 1
            continue
        if drop_jt:
            removed_jt += 1
            continue
        filtered.append(item)
    return filtered, removed_padding, removed_jt


def eval_metrics(preds, truth, tol=TOLERANCE):
    used = set()
    tp = 0
    for t in truth:
        best = None
        best_dist = None
        for idx, p in enumerate(preds):
            if idx in used:
                continue
            d = abs(p - t)
            if d <= tol and (best_dist is None or d < best_dist):
                best = idx
                best_dist = d
        if best is not None:
            used.add(best)
            tp += 1
    fp = len(preds) - len(used)
    fn = len(truth) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, fp, fn, prec, rec, f1


def run_prediction(bin_path: str, threshold: float, post_filter: bool):
    bin_path = pathlib.Path(bin_path)
    asm_path = ensure_asm(bin_path)
    asm = json.loads(asm_path.read_text())
    instrs = asm["instrs"]
    addr_to_idx = {ins["addr"]: i for i, ins in enumerate(instrs)}
    cands = candidate_addresses(asm)

    clf, keys = load_model()
    feats_list = []
    addrs = []
    idxs = []
    for addr in cands:
        idx = addr_to_idx.get(addr)
        if idx is None:
            continue
        feats = featurize_point(instrs, idx)
        vec = [feats.get(k, 0) for k in keys]
        feats_list.append(vec)
        addrs.append(addr)
        idxs.append(idx)

    if not feats_list:
        return {"error": "No candidates found."}

    X = np.array(feats_list, dtype=np.float32)
    probs = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)

    preds = []
    for addr, prob, idx in zip(addrs, probs, idxs):
        if prob >= threshold:
            preds.append(
                {"start": int(addr), "score": float(prob), "idx": idx, "feats": featurize_point(instrs, idx)}
            )

    removed_padding = removed_jt = 0
    if post_filter:
        preds, removed_padding, removed_jt = apply_post_filter(preds, instrs)

    preds = merge_nearby(preds)
    for p in preds:
        p.pop("feats", None)
        p.pop("idx", None)

    # Save
    out_dir = pathlib.Path("out/web_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{bin_path.name}_pred.json"
    out_path.write_text(json.dumps(preds, indent=2))

    # Try to load truth if present
    metrics = None
    opt = bin_path.parts[-2] if len(bin_path.parts) >= 2 else ""
    truth_path = pathlib.Path(f"data/labels/linux/{opt}/{bin_path.name.replace('_stripped','')}_sym.functions_truth.json")
    if truth_path.exists():
        truth = [int(e["start"]) for e in json.loads(truth_path.read_text())]
        metrics = eval_metrics([p["start"] for p in preds], truth, TOLERANCE)

    return {
        "preds": preds,
        "pred_path": str(out_path),
        "removed_padding": removed_padding,
        "removed_jt": removed_jt,
        "metrics": metrics,
        "probs": probs.tolist(),
        "addrs": addrs,
        "truth": truth if truth_path.exists() else None,
        "cands": cands,
    }


def main():
    st.set_page_config(page_title="Function Start Detector Demo", layout="wide")
    st.title("Function Start Detector Demo (RF v06b, macro F1≈0.64 on O3)")
    st.caption("DWARF ground truth; post-filter trims padding/jump-table artifacts.")

    st.sidebar.header("Demo flow")
    st.sidebar.markdown(
        "1) Pick a binary (O3 test list) or paste a path\n"
        "2) Adjust threshold (default 0.25)\n"
        "3) Run prediction (post-filter ON)\n"
        "4) Inspect metrics, histogram, PR/F1 vs threshold"
    )

    bins = load_bins()
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Pick an O3 test binary", [""] + bins, index=0)
        manual = st.text_input("Or enter a binary path", value="")
    with col2:
        threshold = st.slider("Threshold", 0.05, 0.80, DEFAULT_THRESHOLD, 0.01)
        post_filter = st.checkbox("Post-filter (padding + JT)", value=True)

    run_btn = st.button("Run prediction", type="primary")
    if not run_btn:
        st.info("Select a binary and click Run to see predictions, metrics, and plots.")
        return

    bin_path = manual.strip() or selected
    if not bin_path:
        st.error("Please select or enter a binary path.")
        return

    with st.spinner("Scoring..."):
        try:
            result = run_prediction(bin_path, threshold, post_filter)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            return

    if "error" in result:
        st.error(result["error"])
        return

    st.success(f"Predicted {len(result['preds'])} functions. Saved: {result['pred_path']}")
    st.write(f"Post-filter removed padding={result['removed_padding']}, jump-table={result['removed_jt']}")

    tabs = st.tabs([
        "Metrics",
        "Scores",
        "PR/F1 vs threshold",
        "Candidate map",
        "FP/FN details",
        "Feature importances",
        "Pipeline",
    ])

    with tabs[0]:
        if result["metrics"]:
            tp, fp, fn, p, r, f1 = result["metrics"]
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Precision", f"{p:.3f}")
            mcol2.metric("Recall", f"{r:.3f}")
            mcol3.metric("F1", f"{f1:.3f}")
            mcol4.metric("TP/FP/FN", f"{tp}/{fp}/{fn}")
            st.caption(f"Tolerance: ±{TOLERANCE} bytes")
        else:
            st.info("No ground truth found for this binary; showing predictions only.")

        if result["preds"]:
            if pd is not None:
                df = pd.DataFrame(
                    [{"rank": i + 1, "start_hex": hex(p["start"]), "score": p["score"]} for i, p in enumerate(result["preds"])]
                )
                st.dataframe(df.head(50), use_container_width=True, hide_index=True)
            else:
                st.write(result["preds"][:30])

    with tabs[1]:
        if result["preds"] and pd is not None and alt is not None:
            df_scores = pd.DataFrame([p["score"] for p in result["preds"]], columns=["score"])
            chart = (
                alt.Chart(df_scores)
                .mark_bar()
                .encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=25)), y="count()")
                .properties(width=500, height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Score distribution of kept predictions.")
        else:
            st.info("No scores to plot. Run a prediction first.")

    with tabs[2]:
        if pd is not None and alt is not None:
            macro_path = pathlib.Path("out/macro_v06b.tsv")
            if macro_path.exists():
                df_macro = pd.read_csv(macro_path, sep="\t")
                df_rf = df_macro[df_macro["name"].str.contains("summary_thr_v06b_rf_O3_")].copy()
                if not df_rf.empty:
                    df_rf["threshold"] = df_rf["threshold"].astype(float)
                    base = alt.Chart(df_rf).transform_fold(
                        ["MacroP", "MacroR", "MacroF1"], as_=["metric", "value"]
                    )
                    line = base.mark_line(point=True).encode(
                        x=alt.X("threshold:Q", title="Threshold"),
                        y=alt.Y("value:Q", title="Macro metric"),
                        color="metric:N",
                    )
                    vline = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(color="red").encode(x="threshold:Q")
                    st.altair_chart(line + vline, use_container_width=True)
                    st.caption("Macro P/R/F1 on O3 vs threshold (red = current).")
                else:
                    st.info("Macro file found but no RF rows parsed.")
            else:
                st.info("Macro metrics file not found (expected: out/macro_v06b.tsv).")
        else:
            st.info("Altair/pandas not available for plotting.")

    with tabs[3]:
        if pd is not None and alt is not None and result["truth"] and result["preds"]:
            df_map = pd.DataFrame(
                [{"addr": a, "kind": "candidate", "score": s} for a, s in zip(result["addrs"], result["probs"])]
            )
            df_truth = pd.DataFrame([{"addr": t, "kind": "truth", "score": 1.0} for t in result["truth"]])
            df_preds = pd.DataFrame(
                [{"addr": p["start"], "kind": "pred", "score": p["score"]} for p in result["preds"]]
            )
            df_all = pd.concat([df_map, df_truth, df_preds], ignore_index=True)
            chart = (
                alt.Chart(df_all)
                .mark_circle(size=60)
                .encode(
                    x=alt.X("addr:Q", title="Address"),
                    y=alt.Y("score:Q", title="Score/proxy"),
                    color=alt.Color("kind:N", scale=alt.Scale(domain=["truth", "pred", "candidate"], range=["green", "red", "steelblue"])),
                    tooltip=["kind", "addr", "score"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            st.caption("Scatter of candidates vs. predictions vs. truth along address space.")
        else:
            st.info("Need candidates/preds/truth to show the map.")

    with tabs[4]:
        if pd is not None and result["preds"]:
            preds = result["preds"]
            truth = set(result["truth"] or [])
            fn_addrs = [t for t in truth if all(abs(t - p["start"]) > TOLERANCE for p in preds)]
            fp_preds = [p for p in preds if all(abs(p["start"] - t) > TOLERANCE for t in truth)]

            st.subheader("False Positives (top 20 by score)")
            fp_rows = []
            for p in sorted(fp_preds, key=lambda x: x["score"], reverse=True)[:20]:
                fp_rows.append({"start_hex": hex(p["start"]), "score": p["score"]})
            if fp_rows:
                st.dataframe(pd.DataFrame(fp_rows), use_container_width=True, hide_index=True)
            else:
                st.write("No FPs at this threshold.")

            st.subheader("False Negatives (missed truth, top 20)")
            fn_rows = [{"start_hex": hex(a)} for a in fn_addrs[:20]]
            if fn_rows:
                st.dataframe(pd.DataFrame(fn_rows), use_container_width=True, hide_index=True)
            else:
                st.write("No FNs at this threshold.")

            if alt is not None and fp_rows:
                fp_scores = pd.DataFrame([r["score"] for r in fp_rows], columns=["score"])
                chart_fp = (
                    alt.Chart(fp_scores)
                    .mark_bar()
                    .encode(x=alt.X("score:Q", bin=alt.Bin(maxbins=15)), y="count()")
                    .properties(height=200)
                )
                st.altair_chart(chart_fp, use_container_width=True)
                st.caption("FP score distribution (top 20).")
        else:
            st.info("Run a prediction with ground truth available to inspect FP/FN details.")

    with tabs[5]:
        if pd is not None and alt is not None:
            try:
                bundle = load(MODEL_PATH)
                model = bundle["model"]
                keys = bundle.get("feature_keys", [])
                importances = None
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_[0])
                if importances is not None and len(importances) == len(keys):
                    df_imp = pd.DataFrame({"feature": keys, "importance": importances})
                    df_imp = df_imp.sort_values("importance", ascending=False).head(20)
                    chart_imp = (
                        alt.Chart(df_imp)
                        .mark_bar()
                        .encode(
                            x=alt.X("importance:Q", title="Importance / |coef|"),
                            y=alt.Y("feature:N", sort="-x", title="Feature"),
                            tooltip=["feature", "importance"],
                        )
                        .properties(height=400)
                    )
                    st.altair_chart(chart_imp, use_container_width=True)
                    st.caption("Top 20 features by importance (RF: Gini; LR: |coef|).")
                else:
                    st.info("Could not extract feature importances from this model.")
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
        else:
            st.info("Altair/pandas not available for plotting.")

    with tabs[6]:
        st.subheader("Pipeline flow (source files)")
        st.markdown(
            "- `src/parse_objdump.py`: disassembles binary to `*.asm.json` (instrs, xrefs, bytes).\n"
            "- `src/features.py`: candidate selection + feature vector per instruction.\n"
            "- `scripts/train_models_v06_tuned.py`: trains RF/LogReg/XGB (model used here: RF v06b).\n"
            "- `src/predict_starts.py`: inference + post-filter (padding/jump-table) + merge-nearby.\n"
            "- `src/eval_starts.py`: computes P/R/F1 with tolerance against DWARF truth.\n"
            "- Streamlit (this app): orchestrates parse → featurize → score → filter → visualize."
        )
        st.caption("All steps run locally; DWARF labels are the reference ground truth.")


if __name__ == "__main__":
    main()
