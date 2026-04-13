#!/usr/bin/env python3
import argparse, json, subprocess, os, pathlib, re
from joblib import load
import numpy as np

from features import candidate_addresses, featurize_point

JUMP_TABLE_BYTE_TARGET = 48
JUMP_TABLE_INSTR_TARGET = 12
LARGE_IMM_THRESHOLD = 0x1000
RESCUE_SCORE_FLOOR = 0.15
BORDERLINE_ALIGN_RESCUE_SCORE = 0.50
ALIGN_SUCCESSOR_MAX_BYTES = 24
WRAPPER_SCAN_INSTRS = 6
RET_STUB_SCAN_INSTRS = 5
REACHABLE_LEAF_RESCUE_FLOOR = 0.05
RET_STUB_STRONG_SUCCESSOR_SCORE = 0.75
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


def _prev_ret_near(instrs, start_idx, back=3):
    lo = max(0, start_idx - back)
    for idx in range(lo, start_idx):
        if instrs[idx].get("mnemonic", "").startswith("ret"):
            return True
    return False


def _is_short_leaf_candidate(instrs, start_idx, max_ins=5):
    hi = min(len(instrs), start_idx + max_ins)
    for idx in range(start_idx, hi):
        mnem = instrs[idx].get("mnemonic", "")
        if mnem.startswith("call") or mnem.startswith("jmp"):
            return False
        if mnem.startswith("ret"):
            return True
    return False


def _find_aligned_successor(instrs, start_idx, max_bytes=ALIGN_SUCCESSOR_MAX_BYTES):
    base_addr = instrs[start_idx]["addr"]
    for idx in range(start_idx + 1, len(instrs)):
        addr = instrs[idx]["addr"]
        if addr - base_addr > max_bytes:
            break
        if addr % 16 == 0:
            return idx
    return None


def _maybe_shift_wrapper(item, info, instrs):
    feats = item["features"]
    idx = item["idx"]
    if feats.get("xrefs_in", 0) <= 0:
        return item, False

    succ_idx = _find_aligned_successor(instrs, idx, max_bytes=20)
    if succ_idx is None:
        return item, False

    succ_addr = instrs[succ_idx]["addr"]
    succ_info = info.get(succ_addr)
    if succ_info is None:
        return item, False
    succ_score, succ_feats, _ = succ_info
    if succ_feats.get("xrefs_in", 0) != 0:
        return item, False

    body = instrs[idx:succ_idx]
    if not body or len(body) > WRAPPER_SCAN_INSTRS or succ_addr - item["start"] > 16:
        return item, False

    first = body[0]
    second = body[1] if len(body) > 1 else None
    if not (
        first.get("mnemonic", "").startswith("push") and
        "rbp" in first.get("ops", "") and
        second is not None and
        second.get("mnemonic", "").startswith("mov") and
        "rbp,rsp" in second.get("ops", "").replace(" ", "")
    ):
        return item, False

    allowed_prefixes = ("push", "mov", "call", "nop", "xchg")
    if not all(ins.get("mnemonic", "").startswith(allowed_prefixes) for ins in body):
        return item, False

    if not any(ins.get("mnemonic", "").startswith("call") for ins in body[:4]):
        return item, False

    shifted = dict(item)
    shifted["start"] = succ_addr
    shifted["score"] = max(float(item["score"]), float(succ_score) + 1e-6)
    shifted["features"] = succ_feats
    shifted["idx"] = succ_idx
    return shifted, True


def _block_until_ret(instrs, start_idx, limit=RET_STUB_SCAN_INSTRS):
    seq = []
    idx = start_idx
    while idx < len(instrs) and len(seq) < limit:
        ins = instrs[idx]
        seq.append(ins)
        if ins.get("mnemonic", "").startswith("ret"):
            break
        idx += 1
    return seq


def _maybe_retarget_ret_stub(item, info, instrs):
    feats = item["features"]
    idx = item["idx"]
    if feats.get("xrefs_in", 0) <= 0:
        return item, False

    seq = _block_until_ret(instrs, idx, limit=RET_STUB_SCAN_INSTRS)
    if not seq or not seq[-1].get("mnemonic", "").startswith("ret"):
        return item, False
    if any(ins.get("mnemonic", "").startswith("call") for ins in seq):
        return item, False

    succ_idx = _find_aligned_successor(instrs, idx, max_bytes=24)
    if succ_idx is None:
        return item, False

    succ_addr = instrs[succ_idx]["addr"]
    succ_info = info.get(succ_addr)
    if succ_info is None:
        return item, False
    succ_score, succ_feats, _ = succ_info
    if succ_addr - seq[-1]["addr"] > 16:
        return item, False

    if succ_feats.get("xrefs_in", 0) != 0:
        if float(succ_score) < RET_STUB_STRONG_SUCCESSOR_SCORE:
            return item, False
    elif float(succ_score) < 0.20:
        return None, True

    shifted = dict(item)
    shifted["start"] = succ_addr
    shifted["score"] = max(float(item["score"]), float(succ_score) + 1e-6)
    shifted["features"] = succ_feats
    shifted["idx"] = succ_idx
    return shifted, True


def _has_clean_predecessor(instrs, start_idx):
    if start_idx <= 0:
        return True
    prev = instrs[start_idx - 1].get("mnemonic", "").lower()
    if prev.startswith("ret"):
        return True
    if prev in {"ud2", "hlt", "endbr64", "xchg"}:
        return True
    return "nop" in prev


def _is_reachable_leaf_candidate(score, feats, instrs, start_idx, max_ins=8):
    if score >= REACHABLE_LEAF_RESCUE_FLOOR:
        pass
    else:
        return False
    if feats.get("xrefs_in", 0) != 0:
        return False
    if not feats.get("align16", 0):
        return False
    if not feats.get("reachable", 0):
        return False
    if not _has_clean_predecessor(instrs, start_idx):
        return False

    branches = 0
    hi = min(len(instrs), start_idx + max_ins)
    for idx in range(start_idx, hi):
        mnem = instrs[idx].get("mnemonic", "")
        if mnem.startswith("call") or mnem.startswith("jmp"):
            return False
        if mnem.startswith("j"):
            branches += 1
            if branches > 1:
                return False
        if mnem.startswith("ret"):
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model_path", default="models/start_detector.joblib")
    ap.add_argument("--out", default="functions_pred.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--post_filter", choices=["on", "off"], default="on")
    ap.add_argument("--merge_window", type=int, default=4)
    args = ap.parse_args()

    # Ensure asm json exists
    asm_json = f"{args.bin}.asm.json"
    if not os.path.exists(asm_json):
        subprocess.check_call(["python", "src/parse_objdump.py", "--bin", args.bin, "--out", asm_json])

    with open(asm_json) as f:
        asm = json.load(f)

    instrs = asm["instrs"]
    reachable_set = set(asm.get("reachable_addrs", []))
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
        feats = featurize_point(instrs, idx, reachable_set=reachable_set)
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
    rescued = 0
    aligned_rescued = 0
    reachable_rescued = 0
    for addr, p, (feats, idx) in zip(addrs, probs, feats_list):
        if p >= args.threshold:
            pred.append({"start": int(addr), "score": float(p), "features": feats, "idx": idx})
            continue
        if p < RESCUE_SCORE_FLOOR:
            if _is_reachable_leaf_candidate(float(p), feats, instrs, idx):
                pred.append({"start": int(addr), "score": float(p), "features": feats, "idx": idx})
                reachable_rescued += 1
            continue
        if feats.get("xrefs_in", 0) != 0:
            continue
        if not feats.get("align16", 0):
            continue
        if _prev_ret_near(instrs, idx, back=3) and _is_short_leaf_candidate(instrs, idx, max_ins=5):
            pred.append({"start": int(addr), "score": float(p), "features": feats, "idx": idx})
            rescued += 1
            continue
        if float(p) >= BORDERLINE_ALIGN_RESCUE_SCORE and not _looks_like_jump_table(instrs, idx, feats):
            pred.append({"start": int(addr), "score": float(p), "features": feats, "idx": idx})
            aligned_rescued += 1
            continue
        if _is_reachable_leaf_candidate(float(p), feats, instrs, idx):
            pred.append({"start": int(addr), "score": float(p), "features": feats, "idx": idx})
            reachable_rescued += 1

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
    print(f"leaf-rescue added {rescued} candidate(s).")
    print(f"aligned-rescue added {aligned_rescued} candidate(s).")
    print(f"reachable-leaf-rescue added {reachable_rescued} candidate(s).")

    info = {addr: (float(p), feats, idx) for addr, p, (feats, idx) in zip(addrs, probs, feats_list)}
    wrapper_shifted = 0
    shifted_pred = []
    for item in pred:
        shifted_item, changed = _maybe_shift_wrapper(item, info, instrs)
        shifted_pred.append(shifted_item)
        if changed:
            wrapper_shifted += 1
    pred = shifted_pred

    ret_stub_adjusted = 0
    adjusted_pred = []
    for item in pred:
        shifted_item, changed = _maybe_retarget_ret_stub(item, info, instrs)
        if changed:
            ret_stub_adjusted += 1
        if shifted_item is not None:
            adjusted_pred.append(shifted_item)
    pred = adjusted_pred
    print(f"wrapper-shift adjusted {wrapper_shifted} candidate(s).")
    print(f"ret-stub adjusted {ret_stub_adjusted} candidate(s).")

    def merge_nearby(preds, window):
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

    pred = merge_nearby(pred, args.merge_window)
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
