#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib

def load_preds(path):
    with open(path) as f:
        data = json.load(f)
    return [int(entry['start']) for entry in data]


def load_ghidra(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return [int(row['start']) for row in reader]


def compare(preds, truth, tol):
    used_preds = set()
    agree = 0
    for t in truth:
        best = None
        best_dist = None
        for idx, p in enumerate(preds):
            if idx in used_preds:
                continue
            dist = abs(p - t)
            if dist <= tol and (best_dist is None or dist < best_dist):
                best = idx
                best_dist = dist
        if best is not None:
            used_preds.add(best)
            agree += 1
    miss = len(truth) - agree
    extra = len(preds) - agree
    return agree, miss, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_dir', default='out')
    ap.add_argument('--ghidra_dir', default='out/ghidra')
    ap.add_argument('--stems', nargs='+', default=['hello', 'mathlib', 'sort'])
    ap.add_argument('--opt_levels', default='O3')
    ap.add_argument('--suffix', default='_stripped')
    ap.add_argument('--tol', type=int, default=8)
    ap.add_argument('--out', default='out/ghidra_compare_v05.tsv')
    args = ap.parse_args()

    rows = []
    opt_levels = args.opt_levels.split(',')
    for opt in opt_levels:
        for stem in args.stems:
            pred_path = pathlib.Path(args.pred_dir) / f"{stem}.json"
            ghidra_name = f"{opt}_{stem}{args.suffix}.csv"
            ghidra_path = pathlib.Path(args.ghidra_dir) / ghidra_name
            if not pred_path.exists() or not ghidra_path.exists():
                continue
            preds = load_preds(pred_path)
            truth = load_ghidra(ghidra_path)
            agree, miss, extra = compare(preds, truth, args.tol)
            rows.append((f"{opt}_{stem}", agree, miss, extra))

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('file\tagree\tmiss\textra\n')
        for stem, agree, miss, extra in rows:
            f.write(f"{stem}{args.suffix}\t{agree}\t{miss}\t{extra}\n")

    print(f"Wrote {args.out} for {len(rows)} binaries")

if __name__ == '__main__':
    main()
