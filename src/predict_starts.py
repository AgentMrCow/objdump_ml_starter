#!/usr/bin/env python3
import argparse, json, subprocess, os, pathlib, re
from joblib import load
import numpy as np

from features import candidate_addresses, featurize_point

JUMP_TABLE_BYTE_TARGET = 48
JUMP_TABLE_INSTR_TARGET = 12
LARGE_IMM_THRESHOLD = 0x1000
IMM_RE = re.compile(r"0x([0-9a-fA-F]+)")


def _looks_like_jump_table(instrs, start_idx, feats):
    if feats.get("xrefs_in", 0) > 0:
        return False
    if feats.get("window2_xrefs_in", 0) > 0:
        return False

    total = 0
    flagged = 0
    byte_budget = 0
    idx = start_idx + 1
    while idx < len(instrs) and total < JUMP_TABLE_INSTR_TARGET and byte_budget < JUMP_TABLE_BYTE_TARGET:
        ins = instrs[idx]
        total += 1
        byte_budget += len(ins.get("bytes", []))
        if _is_jump_table_like_ins(ins):
            flagged += 1
        idx += 1
    if total == 0:
        return False
    return (flagged / total) >= 0.5


def _is_jump_table_like_ins(ins):
    mnem = ins.get("mnemonic", "")
    ops = ins.get("ops", "")
    if not mnem:
        return False
    if mnem.startswith("jmp"):
        return True
    lowered = mnem.lower()
    if lowered in {"db", "dd", "dq", ".byte", ".quad", ".long"}:
        return True
    if lowered.startswith("mov") and _has_large_immediate(ops):
        return True
    return False


def _has_large_immediate(ops):
    if not ops:
        return False
    match = IMM_RE.search(ops)
    if not match:
        return False
    try:
        val = int(match.group(1), 16)
    except ValueError:
        return False
    return val >= LARGE_IMM_THRESHOLD

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model_path", default="models/start_detector.joblib")
    ap.add_argument("--out", default="functions_pred.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--post_filter", choices=["on", "off"], default="on")
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
    feats_list = []
    for idx in cand_idxs:
        feats = featurize_point(instrs, idx)
        vec = [feats[k] if k in feats else 0 for k in keys]
        X.append(vec)
        addrs.append(instrs[idx]["addr"])
        feats_list.append((feats, idx))

    if not X:
        with open(args.out, "w") as f:
            json.dump([], f, indent=2)
        print(f"Wrote predictions -> {args.out} (0 functions).")
        return

    X = np.array(X, dtype=np.float32)
    probs = clf.predict_proba(X)[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(X)

    pred = []
    for addr, p, (feats, idx) in zip(addrs, probs, feats_list):
        if p >= args.threshold:
            pred.append({"start": int(addr), "score": float(p), "features": feats, "idx": idx})

    removed_padding = 0
    removed_jt = 0
    if args.post_filter == "on":
        filtered = []
        for item in pred:
            feats = item["features"]
            cond_a = feats.get("xrefs_in", 0) == 0
            cond_b = feats.get("padding_nop_run", 0) >= 3
            cond_c = not (
                feats.get("prev_is_ret", 0) or
                feats.get("has_push_rbp", 0) or
                feats.get("window2_xrefs_in", 0) > 0
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
        pred = filtered
        print(f"Post-filter removed {removed_padding} candidate(s).")
        print(f"jt-filter removed {removed_jt} candidate(s).")
    else:
        print("Post-filter disabled (0 candidates removed).")
        print("jt-filter skipped (post_filter=off).")

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

    pred = merge_nearby(pred)
    # naive end stitching: next predicted start or end of list
    pred_sorted = sorted(pred, key=lambda x: x["start"])
    for i in range(len(pred_sorted)):
        if i < len(pred_sorted) - 1:
            pred_sorted[i]["end"] = pred_sorted[i+1]["start"]
        else:
            pred_sorted[i]["end"] = pred_sorted[i]["start"] + 64  # placeholder
        pred_sorted[i].pop("features", None)
        pred_sorted[i].pop("idx", None)
    with open(args.out, "w") as f:
        json.dump(pred_sorted, f, indent=2)
    print(f"Wrote predictions -> {args.out} ({len(pred_sorted)} functions).")

if __name__ == "__main__":
    main()
