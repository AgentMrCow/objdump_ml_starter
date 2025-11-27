#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from features import candidate_addresses, featurize_point


def load_ghidra_starts(csv_path):
    starts = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                starts.append(int(row["start"]))
            except (KeyError, ValueError):
                continue
    return sorted(set(starts))


def nearest_distance(addr, starts):
    if not starts:
        return None
    # binary search could be faster but list is tiny; simple min works.
    return min(abs(addr - s) for s in starts)


def process_binary(bin_path, ghidra_csv, out_dir, tolerance):
    asm_json = f"{bin_path}.asm.json"
    if not os.path.exists(asm_json):
        raise FileNotFoundError(f"Missing asm json for {bin_path}")
    if not os.path.exists(ghidra_csv):
        raise FileNotFoundError(f"Missing Ghidra CSV for {bin_path}")

    with open(asm_json) as f:
        asm = json.load(f)
    instrs = asm["instrs"]
    addr_to_idx = {ins["addr"]: i for i, ins in enumerate(instrs)}
    cands = candidate_addresses(asm)
    ghidra_starts = load_ghidra_starts(ghidra_csv)

    hard_negs = []
    for addr in cands:
        dist = nearest_distance(addr, ghidra_starts)
        if dist is not None and dist <= tolerance:
            continue
        idx = addr_to_idx.get(addr)
        if idx is None:
            continue
        feats = featurize_point(instrs, idx)
        hard_negs.append({"addr": addr, "features": feats})

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = os.path.basename(bin_path)
    out_path = os.path.join(out_dir, f"{stem}_hardnegs.json")
    with open(out_path, "w") as f:
        json.dump(hard_negs, f, indent=2)
    return out_path, len(hard_negs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins_glob", default="data/build/linux/O3/*_stripped")
    ap.add_argument("--ghidra_dir", default="out/ghidra")
    ap.add_argument("--out_dir", default="out/mining")
    ap.add_argument("--tolerance", type=int, default=8)
    args = ap.parse_args()

    bins = sorted(pathlib.Path().glob(args.bins_glob))
    if not bins:
        print(f"No binaries matched {args.bins_glob}")
        return

    totals = []
    for bin_path in bins:
        ghidra_csv = os.path.join(args.ghidra_dir, f"{bin_path.name}.csv")
        try:
            out_path, count = process_binary(str(bin_path), ghidra_csv, args.out_dir, args.tolerance)
            print(f"{bin_path}: wrote {count} hard negatives -> {out_path}")
            totals.append((bin_path.name, count))
        except FileNotFoundError as e:
            print(f"[WARN] {e}")

    if totals:
        print("\nSummary:")
        for name, count in totals:
            print(f"  {name}: {count} hard negatives")

if __name__ == "__main__":
    main()
