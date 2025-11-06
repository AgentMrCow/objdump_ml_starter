#!/usr/bin/env python3
import json, argparse, math, pathlib

def load_asm_json(path):
    with open(path) as f:
        return json.load(f)

def build_index(instrs):
    addr_to_idx = {ins["addr"]: i for i, ins in enumerate(instrs)}
    return addr_to_idx

def candidate_addresses(data):
    instrs = data["instrs"]
    addrs = set()
    if not instrs:
        return []
    # section start as candidate
    addrs.add(instrs[0]["addr"])
    # target of direct calls/jumps as candidate
    for ins in instrs:
        for t in ins.get("xrefs_out", []):
            if isinstance(t, int):
                addrs.add(t)
    # address following a 'ret' as candidate
    for i, ins in enumerate(instrs[:-1]):
        if ins["mnemonic"].startswith("ret"):
            addrs.add(instrs[i+1]["addr"])
    return sorted(a for a in addrs if a in {ins["addr"] for ins in instrs})

def window(instrs, idx, k=2):
    start = max(0, idx-k)
    end = min(len(instrs), idx+k+1)
    return instrs[start:end], idx-start

def featurize_point(instrs, idx):
    ins = instrs[idx]
    prevs, pos = window(instrs, idx, k=2)
    # simple numeric features
    features = {}
    # alignment
    features["align16"] = 1 if (ins["addr"] % 16 == 0) else 0
    # xref count to this addr
    features["xrefs_in"] = instrs[idx].get("xrefs_in", 0)
    # prev is ret?
    features["prev_is_ret"] = 1 if idx > 0 and instrs[idx-1]["mnemonic"].startswith("ret") else 0
    # prologue-like sequence nearby
    seq = " ".join(p["mnemonic"] for p in prevs)
    features["has_push_rbp"] = 1 if "push" in seq and "rbp" in " ".join(p["ops"] for p in prevs) else 0
    # current mnemonic one-hot (top few only)
    m = ins["mnemonic"]
    for key in ["push","mov","sub","add","call","jmp","lea","xor","nop"]:
        features[f"m_{key}"] = 1 if m.startswith(key) else 0
    return features

def to_vector(feature_dict):
    keys = sorted(feature_dict.keys())
    return [feature_dict[k] for k in keys], keys

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
