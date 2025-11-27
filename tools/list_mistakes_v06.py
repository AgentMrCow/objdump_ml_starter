#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from features import featurize_point

def load_instrs(asm_path):
    with open(asm_path) as f:
        data = json.load(f)
    instrs = data['instrs']
    addr_to_idx = {ins['addr']: i for i, ins in enumerate(instrs)}
    return instrs, addr_to_idx


def within_tolerance(addr, others, tol):
    return any(abs(addr - other) <= tol for other in others)

KEYS = [
    'xrefs_in','xrefs_out_count','prev_is_ret','has_push_rbp','padding_nop_run',
    'window2_xrefs_in','window2_xrefs_out','window6_xrefs_in','window6_xrefs_out',
    'ng_push_mov','ng_push_mov_sub','align16'
]

def emit(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('\t'.join(['addr_hex'] + KEYS) + '\n')
        for addr_hex, feats in rows:
            f.write('\t'.join([addr_hex] + [str(feats.get(k, '')) for k in KEYS]) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bins_list', required=True)
    ap.add_argument('--opt', default='O3')
    ap.add_argument('--tol', type=int, default=8)
    ap.add_argument('--pred_dir', default='out')
    ap. add_argument('--out_dir', default='out/error_analysis_v06')
    args = ap.parse_args()

    bins = [line.strip() for line in Path(args.bins_list).read_text().splitlines() if line.strip()]
    for bin_path in bins:
        stem = Path(bin_path).name.replace('_stripped','')
        pred_path = Path(args.pred_dir) / f"{stem}.json"
        if not pred_path.exists():
            continue
        with open(pred_path) as f:
            preds = [int(item['start']) for item in json.load(f)]
        truth_path = Path(f"data/labels/linux/{args.opt}/{stem}_sym.functions_truth.json")
        asm_path = Path(f"data/build/linux/{args.opt}/{stem}_stripped.asm.json")
        if not truth_path.exists() or not asm_path.exists():
            continue
        with open(truth_path) as f:
            truth = [int(item['start']) for item in json.load(f)]
        instrs, idx_map = load_instrs(asm_path)
        fn_rows = []
        for addr in truth:
            if not within_tolerance(addr, preds, args.tol):
                idx = idx_map.get(addr)
                feats = featurize_point(instrs, idx) if idx is not None else {}
                fn_rows.append((hex(addr), feats))
        fp_rows = []
        for addr in preds:
            if not within_tolerance(addr, truth, args.tol):
                idx = idx_map.get(addr)
                feats = featurize_point(instrs, idx) if idx is not None else {}
                fp_rows.append((hex(addr), feats))
        emit(fn_rows, os.path.join(args.out_dir, f"{stem}_fn.tsv"))
        emit(fp_rows, os.path.join(args.out_dir, f"{stem}_fp.tsv"))

if __name__ == '__main__':
    main()
