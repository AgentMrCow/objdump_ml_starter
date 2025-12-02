#!/usr/bin/env python3
import json, argparse, pathlib

CONTEXT_RADIUS = 6
SHORT_WINDOW_RADII = (2, 6)
NGRAM_RADIUS = 3
PADDING_LOOKAHEAD = 8

BASE_MNEMONICS = ["push", "mov", "sub", "add", "call", "jmp", "lea", "xor", "nop"]
NGRAM_PATTERNS = [
    ("push", "mov"),
    ("push", "mov", "sub"),
    ("mov", "call"),
    ("sub", "call"),
    ("xor", "xor"),
    ("leave", "ret"),
]

BASE_FEATURE_KEYS = [
    "align16",
    "align32",
    "align64",
    "xrefs_in",
    "xrefs_out_count",
    "prev_is_ret",
    "has_push_rbp",
    "window2_xrefs_in",
    "window2_xrefs_out",
    "window6_xrefs_in",
    "window6_xrefs_out",
    "padding_nop_run",
]

MNEMONIC_FEATURE_KEYS = [f"m_{m}" for m in BASE_MNEMONICS]
NGRAM_FEATURE_KEYS = ["ng_" + "_".join(pat) for pat in NGRAM_PATTERNS]

CFG_FEATURE_KEYS = [
    "bb_start",
    "byte_sum",
    "byte_len",
    "rsp_touch",
    "bb_preds",
    "bb_succs",
    "bb_len",
    "byte_entropy",
]

FEATURE_KEYS = BASE_FEATURE_KEYS + MNEMONIC_FEATURE_KEYS + NGRAM_FEATURE_KEYS + CFG_FEATURE_KEYS


def get_feature_keys():
    return list(FEATURE_KEYS)

def load_asm_json(path):
    with open(path) as f:
        return json.load(f)

def build_index(instrs):
    return {ins["addr"]: i for i, ins in enumerate(instrs)}


def _is_branch(ins):
    mnem = ins.get("mnemonic", "")
    if not mnem:
        return False
    if mnem.startswith("j"):
        return True
    return mnem in {"loop", "loope", "loopne", "jecxz", "jrcxz"}


def _is_conditional_branch(ins):
    mnem = ins.get("mnemonic", "")
    if not mnem:
        return False
    if mnem.startswith("j") and not mnem.startswith("jmp"):
        return True
    return False


def candidate_addresses(data):
    instrs = data["instrs"]
    if not instrs:
        return []

    addr_set = {ins["addr"] for ins in instrs}
    addrs = set()

    # section start as candidate
    addrs.add(instrs[0]["addr"])

    # target of direct calls/jumps as candidate
    for ins in instrs:
        for t in ins.get("xrefs_out", []):
            if isinstance(t, int):
                addrs.add(t)
        if _is_conditional_branch(ins):
            for t in ins.get("xrefs_out", []):
                if isinstance(t, int) and t in addr_set:
                    addrs.add(t)

    # address following a 'ret' or branch as candidate when instruction follows
    for i, ins in enumerate(instrs[:-1]):
        if ins["mnemonic"].startswith("ret") or _is_branch(ins):
            addrs.add(instrs[i + 1]["addr"])

    # alignment-derived candidates
    for ins in instrs:
        addr = ins["addr"]
        if addr % 16 == 0 or addr % 32 == 0 or addr % 64 == 0:
            addrs.add(addr)

    # objdump label anchors
    for addr_str in data.get("labels", {}):
        try:
            addr = int(addr_str)
        except ValueError:
            continue
        if addr in addr_set:
            addrs.add(addr)

    return sorted(addr for addr in addrs if addr in addr_set)


def window(instrs, idx, k=CONTEXT_RADIUS):
    start = max(0, idx - k)
    end = min(len(instrs), idx + k + 1)
    return instrs[start:end], idx - start

def _contains_pattern(mnemonics, pattern):
    if len(mnemonics) < len(pattern):
        return False
    for i in range(len(mnemonics) - len(pattern) + 1):
        for j, token in enumerate(pattern):
            if not mnemonics[i + j].startswith(token):
                break
        else:
            return True
    return False


def featurize_point(instrs, idx):
    ins = instrs[idx]
    features = {key: 0 for key in FEATURE_KEYS}

    addr = ins["addr"]
    features["align16"] = 1 if addr % 16 == 0 else 0
    features["align32"] = 1 if addr % 32 == 0 else 0
    features["align64"] = 1 if addr % 64 == 0 else 0

    features["xrefs_in"] = ins.get("xrefs_in", 0)
    features["xrefs_out_count"] = len(ins.get("xrefs_out", []))

    if idx > 0 and instrs[idx - 1]["mnemonic"].startswith("ret"):
        features["prev_is_ret"] = 1

    context_instrs, context_pos = window(instrs, idx, CONTEXT_RADIUS)
    before_context = context_instrs[:context_pos]

    for prev_ins in reversed(before_context):
        if prev_ins["mnemonic"].startswith("push") and "rbp" in prev_ins.get("ops", ""):
            features["has_push_rbp"] = 1
            break

    for radius in SHORT_WINDOW_RADII:
        win_instrs, _ = window(instrs, idx, radius)
        in_total = sum(w.get("xrefs_in", 0) for w in win_instrs)
        out_total = sum(len(w.get("xrefs_out", [])) for w in win_instrs)
        features[f"window{radius}_xrefs_in"] = in_total
        features[f"window{radius}_xrefs_out"] = out_total

    padding_run = 0
    for j in range(idx + 1, min(len(instrs), idx + 1 + PADDING_LOOKAHEAD)):
        next_mnemonic = instrs[j]["mnemonic"]
        if "nop" in next_mnemonic:
            padding_run += 1
        else:
            break
    features["padding_nop_run"] = padding_run

    m = ins["mnemonic"]
    for key in BASE_MNEMONICS:
        features[f"m_{key}"] = 1 if m.startswith(key) else 0

    ngram_window, _ = window(instrs, idx, NGRAM_RADIUS)
    mnemonics = [item["mnemonic"] for item in ngram_window]
    for pattern, key in zip(NGRAM_PATTERNS, NGRAM_FEATURE_KEYS):
        if _contains_pattern(mnemonics, pattern):
            features[key] = 1

    # CFG / byte-level helpers (defaults to zero when absent)
    features["bb_start"] = ins.get("bb_start", 0)
    features["byte_sum"] = ins.get("byte_sum", sum(ins.get("bytes", [])))
    features["byte_len"] = ins.get("byte_len", len(ins.get("bytes", [])))
    features["rsp_touch"] = ins.get("rsp_touch", 0)
    features["bb_preds"] = ins.get("bb_preds", 0)
    features["bb_succs"] = ins.get("bb_succs", 0)
    features["bb_len"] = ins.get("bb_len", 0)
    if "byte_entropy" in ins:
        features["byte_entropy"] = ins["byte_entropy"]
    else:
        b = ins.get("bytes", [])
        features["byte_entropy"] = (sum(b) / len(b)) if b else 0

    return features

def to_vector(feature_dict):
    return [feature_dict[k] for k in FEATURE_KEYS], FEATURE_KEYS

def build_matrix(instrs, cand_idxs):
    X = []
    meta = []
    last_keys = None
    for idx in cand_idxs:
        f = featurize_point(instrs, idx)
        vec, keys = to_vector(f)
        last_keys = keys
        X.append(vec)
        meta.append(instrs[idx]["addr"])
    return X, meta, last_keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm_json", required=True)
    ap.add_argument("--out_features", required=True)
    args = ap.parse_args()
    data = load_asm_json(args.asm_json)
    instrs = data["instrs"]
    addr_to_idx = {ins["addr"]: i for i, ins in enumerate(instrs)}
    cands = candidate_addresses(data)
    cand_idxs = [addr_to_idx[a] for a in cands if a in addr_to_idx]
    X, meta, keys = build_matrix(instrs, cand_idxs)
    out = {
        "X": X,
        "keys": keys,
        "addresses": meta
    }
    pathlib.Path(args.out_features).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_features, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote features for {len(meta)} candidate addresses.")

if __name__ == "__main__":
    main()
