#!/usr/bin/env python3
import argparse, json, subprocess, os, pathlib
from joblib import load
import numpy as np

from features import candidate_addresses, featurize_point

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model_path", default="models/start_detector.joblib")
    ap.add_argument("--out", default="functions_pred.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    # Ensure asm json exists
    asm_json = f"{args.bin}.asm.json"
    if not os.path.exists(asm_json):
        subprocess.check_call(["python", "src/parse_objdump.py", "--bin", args.bin, "--out", asm_json])

    with open(asm_json) as f:
        asm = json.load(f)

    instrs = asm["instrs"]
    addr_to_idx = {ins["addr"]: i for i, ins in enumerate(instrs)}
    cands = candidate_addresses(asm)
    cand_idxs = [addr_to_idx[a] for a in cands if a in addr_to_idx]

    bundle = load(args.model_path)
    clf = bundle["model"]
    keys = bundle["feature_keys"]

    X = []
    addrs = []
    for idx in cand_idxs:
        feats = featurize_point(instrs, idx)
        vec = [feats[k] if k in feats else 0 for k in keys]
        X.append(vec)
        addrs.append(instrs[idx]["addr"])
    X = np.array(X, dtype=np.float32)
    probs = clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(X)

    pred = []
    for addr, p in zip(addrs, probs):
        if p >= args.threshold:
            pred.append({"start": int(addr), "score": float(p)})
    # naive end stitching: next predicted start or end of list
    pred_sorted = sorted(pred, key=lambda x: x["start"])
    for i in range(len(pred_sorted)):
        if i < len(pred_sorted) - 1:
            pred_sorted[i]["end"] = pred_sorted[i+1]["start"]
        else:
            pred_sorted[i]["end"] = pred_sorted[i]["start"] + 64  # placeholder
    with open(args.out, "w") as f:
        json.dump(pred_sorted, f, indent=2)
    print(f"Wrote predictions -> {args.out} ({len(pred_sorted)} functions).")

if __name__ == "__main__":
    main()
