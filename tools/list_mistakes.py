#!/usr/bin/env python3
import argparse
import json
import os
import pathlib

from features import featurize_point

def load_instrs(asm_path):
    with open(asm_path) as f:
        data = json.load(f)
    instrs = data["instrs"]
    addr_to_idx = {ins["addr"]: idx for idx, ins in enumerate(instrs)}
    return instrs, addr_to_idx


def within_tolerance(addr, others, tol):
    return any(abs(addr - other) <= tol for other in others)


KEYS = [
    'xrefs_in',
    'xrefs_out_count',
    'prev_is_ret',
    'has_push_rbp',
    'padding_nop_run',
    'window2_xrefs_in',
    'window2_xrefs_out',
    'window6_xrefs_in',
    'window6_xrefs_out',
    'ng_push_mov',
    'ng_push_mov_sub',
    'align16',
]


def emit(rows, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        if not rows:
            f.write('addr_hex\n')
            return
        headers = ['addr_hex'] + KEYS
        f.write('\t'.join(headers) + '\n')
        for addr_hex, feats in rows:
            f.write('\t'.join([addr_hex] + [str(feats.get(h, '')) for h in headers[1:]]) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--opt', default='O3')
    ap.add_argument('--stems', nargs='+', default=['hello', 'mathlib', 'sort'])
    ap.add_argument('--pred_dir', default='out')
    ap.add_argument('--tol', type=int, default=8)
    ap.add_argument('--out_dir', default='out/error_analysis')
    args = ap.parse_args()

    for stem in args.stems:
        pred_path = pathlib.Path(args.pred_dir) / f"{stem}.json"
        if not pred_path.exists():
            continue
        with open(pred_path) as f:
            preds = [int(item['start']) for item in json.load(f)]
        truth_path = pathlib.Path(f"data/labels/linux/{args.opt}/{stem}_sym.functions_truth.json")
        asm_path = pathlib.Path(f"data/build/linux/{args.opt}/{stem}_stripped.asm.json")
        if not truth_path.exists() or not asm_path.exists():
            continue
        with open(truth_path) as f:
            truth = [int(item['start']) for item in json.load(f)]
        instrs, addr_to_idx = load_instrs(asm_path)

        fn_rows = []
        for addr in truth:
            if not within_tolerance(addr, preds, args.tol):
                idx = addr_to_idx.get(addr)
                feats = featurize_point(instrs, idx) if idx is not None else {}
                fn_rows.append((hex(addr), feats))
        fp_rows = []
        for addr in preds:
            if not within_tolerance(addr, truth, args.tol):
                idx = addr_to_idx.get(addr)
                feats = featurize_point(instrs, idx) if idx is not None else {}
                fp_rows.append((hex(addr), feats))

        emit(fn_rows, os.path.join(args.out_dir, f"{stem}_fn.tsv"))
        emit(fp_rows, os.path.join(args.out_dir, f"{stem}_fp.tsv"))

if __name__ == '__main__':
    main()
