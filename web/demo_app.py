#!/usr/bin/env python3
"""
Lightweight Flask demo app to run the best-performing model (v06b RF, ~0.64
macro F1 on O3) against sample binaries and show precision/recall instantly.
Launch with:
  PYTHONPATH=src FLASK_APP=web.demo_app FLASK_RUN_PORT=5000 FLASK_RUN_HOST=0.0.0.0 .venv/bin/flask run
"""
import json
import pathlib
import subprocess
from typing import List, Tuple

import numpy as np
from flask import Flask, request, render_template_string
from joblib import load

from features import candidate_addresses, featurize_point
from predict_starts import _looks_like_jump_table


app = Flask(__name__)

MODEL_PATH = pathlib.Path("models/start_detector_v06b_rf.joblib")
DEFAULT_THRESHOLD = 0.25  # best macro F1 ≈ 0.639 on O3
TOLERANCE = 8
BIN_LIST = pathlib.Path("out/test_bins_O3.txt")  # program-level O3 test split


def load_bins(max_items: int = 80) -> List[str]:
    if not BIN_LIST.exists():
        return []
    bins = [
        line.strip()
        for line in BIN_LIST.read_text().splitlines()
        if line.strip()
    ]
    return bins[:max_items]


def ensure_asm(bin_path: pathlib.Path) -> pathlib.Path:
    asm_path = pathlib.Path(f"{bin_path}.asm.json")
    if not asm_path.exists():
        asm_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["python", "src/parse_objdump.py", "--bin", str(bin_path), "--out", str(asm_path)],
            check=True,
        )
    return asm_path


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")
    bundle = load(MODEL_PATH)
    clf = bundle["model"]
    keys = bundle.get("feature_keys", [])
    return clf, keys


def merge_nearby(preds: List[dict], window: int = 8) -> List[dict]:
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


def apply_post_filter(preds: List[dict], instrs: list) -> Tuple[List[dict], int, int]:
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


def eval_metrics(preds: List[int], truth: List[int], tol: int = TOLERANCE):
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
    return {"tp": tp, "fp": fp, "fn": fn, "prec": prec, "rec": rec, "f1": f1}


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
        return {"error": "No candidates found"}, None

    X = np.array(feats_list, dtype=np.float32)
    probs = (
        clf.predict_proba(X)[:, 1]
        if hasattr(clf, "predict_proba")
        else clf.decision_function(X)
    )

    preds = []
    for addr, prob, idx in zip(addrs, probs, idxs):
        if prob >= threshold:
            preds.append({"start": int(addr), "score": float(prob), "idx": idx, "feats": featurize_point(instrs, idx)})

    removed_padding = removed_jt = 0
    if post_filter:
        preds, removed_padding, removed_jt = apply_post_filter(preds, instrs)

    preds = merge_nearby(preds)
    for p in preds:
        p.pop("feats", None)
        p.pop("idx", None)

    # Persist demo output
    stem = bin_path.name
    out_dir = pathlib.Path("out/web_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_pred.json"
    out_path.write_text(json.dumps(preds, indent=2))

    # Attempt evaluation if ground truth exists
    metrics = None
    parts = bin_path.parts
    opt = parts[-2] if len(parts) >= 2 else ""
    truth_path = pathlib.Path(f"data/labels/linux/{opt}/{stem.replace('_stripped','')}_sym.functions_truth.json")
    if truth_path.exists():
        truth = [int(entry["start"]) for entry in json.loads(truth_path.read_text())]
        metrics = eval_metrics([p["start"] for p in preds], truth, TOLERANCE)

    summary = {
        "pred_path": str(out_path),
        "pred_count": len(preds),
        "removed_padding": removed_padding,
        "removed_jt": removed_jt,
        "preds": preds,
        "metrics": metrics,
    }
    return summary, None


TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>Function Start Demo (RF v06b ~0.64 F1)</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 24px auto; }
    .metrics { background: #f4f6f8; padding: 12px; border-radius: 6px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 6px; text-align: left; font-size: 14px; }
    th { background: #eee; }
  </style>
</head>
<body>
  <h2>Function Start Detector Demo (RF v06b, macro F1≈0.64 on O3)</h2>
  <p>Select a prepared O3 stripped binary or enter a path, choose a threshold (default 0.25), and run prediction. Post-filter is ON by default.</p>
  <form method="POST">
    <label>Pick a sample (O3 test subset):</label><br/>
    <select name="bin_select">
      <option value="">-- choose --</option>
      {% for b in bins %}
        <option value="{{ b }}" {% if selected_bin==b %}selected{% endif %}>{{ b }}</option>
      {% endfor %}
    </select><br/><br/>
    <label>Or enter binary path:</label><br/>
    <input type="text" name="bin_path" size="80" value="{{ manual_bin or '' }}"><br/><br/>
    <label>Threshold:</label>
    <input type="text" name="threshold" value="{{ threshold }}"><br/>
    <label><input type="checkbox" name="post_filter" value="on" {% if post_filter %}checked{% endif %}> Post-filter (padding + jump-table)</label><br/><br/>
    <button type="submit">Run prediction</button>
  </form>
  {% if error %}
    <p style="color:red;">Error: {{ error }}</p>
  {% endif %}
  {% if result %}
    <div class="metrics">
      <p><strong>Output:</strong> {{ result.pred_count }} functions → saved to {{ result.pred_path }}</p>
      <p>Post-filter removed padding={{ result.removed_padding }} jump-table={{ result.removed_jt }}</p>
      {% if result.metrics %}
        <p><strong>Metrics (tol=8 bytes):</strong>
          TP={{ result.metrics.tp }}, FP={{ result.metrics.fp }}, FN={{ result.metrics.fn }},
          P={{ "%.3f"|format(result.metrics.prec) }},
          R={{ "%.3f"|format(result.metrics.rec) }},
          F1={{ "%.3f"|format(result.metrics.f1) }}
        </p>
      {% else %}
        <p><em>No ground truth found for this binary; showing predictions only.</em></p>
      {% endif %}
    </div>
    <h3>Top predictions</h3>
    <table>
      <tr><th>#</th><th>Start (hex)</th><th>Score</th></tr>
      {% for p in result.preds[:30] %}
        <tr><td>{{ loop.index }}</td><td>{{ "0x%x"|format(p.start) }}</td><td>{{ "%.3f"|format(p.score) }}</td></tr>
      {% endfor %}
    </table>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    bins = load_bins()
    selected_bin = None
    manual_bin = None
    threshold = DEFAULT_THRESHOLD
    post_filter = True
    result = None
    error = None
    if request.method == "POST":
        selected_bin = request.form.get("bin_select") or None
        manual_bin = request.form.get("bin_path") or None
        bin_path = selected_bin or manual_bin
        if not bin_path:
            error = "Please select or enter a binary path."
        else:
            try:
                threshold = float(request.form.get("threshold") or DEFAULT_THRESHOLD)
            except ValueError:
                error = "Threshold must be a number."
                threshold = DEFAULT_THRESHOLD
            post_filter = request.form.get("post_filter") == "on"
            if not error:
                try:
                    result, err = run_prediction(bin_path, threshold, post_filter)
                    if err:
                        error = err
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
    return render_template_string(
        TEMPLATE,
        bins=bins,
        selected_bin=selected_bin,
        manual_bin=manual_bin,
        threshold=threshold,
        post_filter=post_filter,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
