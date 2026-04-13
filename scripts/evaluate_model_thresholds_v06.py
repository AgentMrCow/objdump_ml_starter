#!/usr/bin/env python3
import argparse
import json
import pathlib
import numpy as np
from joblib import load

from features import candidate_addresses, featurize_point
from predict_starts import (
    RESCUE_SCORE_FLOOR,
    _is_short_leaf_candidate,
    _looks_like_jump_table,
    _prev_ret_near,
)


def load_truth(label_glob):
    import glob
    paths = glob.glob(label_glob)
    truth = set()
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for entry in data:
            start = int(entry['start'])
            if start > 0:
                truth.add(start)
    return truth


def collect_candidates(bin_path):
    asm_path = bin_path + '.asm.json'
    with open(asm_path) as f:
        asm = json.load(f)
    instrs = asm['instrs']
    reachable_set = set(asm.get("reachable_addrs", []))
    addr_to_idx = {ins['addr']: i for i, ins in enumerate(instrs)}
    cands = candidate_addresses(asm)
    feats = []
    addrs = []
    idxs = []
    for addr in cands:
        idx = addr_to_idx.get(addr)
        if idx is None:
            continue
        fvec = featurize_point(instrs, idx, reachable_set=reachable_set)
        feats.append(fvec)
        addrs.append(addr)
        idxs.append(idx)
    return instrs, addrs, idxs, feats


def score_model(bundle, feature_keys, feats):
    clf = bundle['model']
    keys = bundle['feature_keys'] if 'feature_keys' in bundle else feature_keys
    X = np.array([[f.get(k, 0) for k in keys] for f in feats], dtype=np.float32)
    probs = clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X)
    return probs


def apply_prediction_pipeline(instrs, addrs, feats, idxs, probs, threshold, merge_window, post_filter):
    pred = []
    for addr, prob, feat, idx in zip(addrs, probs, feats, idxs):
        keep = False
        if prob >= threshold:
            keep = True
        elif (
            prob >= RESCUE_SCORE_FLOOR and
            feat.get('xrefs_in', 0) == 0 and
            feat.get('align16', 0) and
            _prev_ret_near(instrs, idx, back=3) and
            _is_short_leaf_candidate(instrs, idx, max_ins=5)
        ):
            keep = True
        if keep:
            pred.append({'start': addr, 'score': float(prob), 'features': feat, 'idx': idx})

    if post_filter:
        filtered = []
        for item in pred:
            feat = item['features']
            cond_a = feat.get('xrefs_in', 0) == 0
            cond_b = feat.get('padding_nop_run', 0) >= 3
            cond_c = not (
                feat.get('prev_is_ret', 0) or
                feat.get('has_push_rbp', 0) or
                feat.get('window2_xrefs_in', 0) > 0
            )
            drop_padding = cond_a and cond_b and cond_c
            drop_jt = False if drop_padding else _looks_like_jump_table(instrs, item['idx'], feat)
            if drop_padding or drop_jt:
                continue
            filtered.append(item)
        pred = filtered

    pred = sorted(pred, key=lambda x: x['start'])
    if merge_window > 0:
        merged = []
        for item in pred:
            if merged and item['start'] - merged[-1]['start'] <= merge_window:
                if item['score'] > merged[-1]['score']:
                    merged[-1] = item
            else:
                merged.append(item)
        pred = merged

    return [item['start'] for item in pred]


def score_predicted_starts(pred_starts, truth, tol):
    used = set()
    tp = 0
    for t in truth:
        best = None
        best_dist = None
        for i, p in enumerate(pred_starts):
            if i in used:
                continue
            d = abs(p - t)
            if d <= tol and (best_dist is None or d < best_dist):
                best = i
                best_dist = d
        if best is not None:
            tp += 1
            used.add(best)
    fp = len(pred_starts) - len(used)
    fn = len(truth) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, fp, fn, prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bins_list', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--tolerance', type=int, default=8)
    ap.add_argument('--thresholds', default='0.20,0.25,0.30,0.33,0.35,0.40,0.45,0.50,0.55')
    ap.add_argument('--merge_window', type=int, default=4)
    ap.add_argument('--post_filter', choices=['on', 'off'], default='on')
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',') if x]
    bins = [line.strip() for line in pathlib.Path(args.bins_list).read_text().splitlines() if line.strip()]
    bundle = load(args.model)
    feature_keys = bundle.get('feature_keys', [])

    all_rows = {thr: [] for thr in thresholds}
    for bin_path in bins:
        instrs, addrs, idxs, feats = collect_candidates(bin_path)
        probs = score_model(bundle, feature_keys, feats)
        stem = pathlib.Path(bin_path).name.replace('_stripped','')
        opt_level = pathlib.Path(bin_path).parts[-2]
        truth = load_truth(f"data/labels/linux/{opt_level}/{stem}_sym.functions_truth.json")
        for thr in thresholds:
            pred_starts = apply_prediction_pipeline(
                instrs, addrs, feats, idxs, probs, thr,
                merge_window=args.merge_window,
                post_filter=(args.post_filter == 'on'),
            )
            tp, fp, fn, p, r, f1 = score_predicted_starts(pred_starts, truth, args.tolerance)
            all_rows[thr].append((stem, tp, fp, fn, p, r, f1))

    for thr in thresholds:
        out_path = pathlib.Path(f"{args.out_prefix}_{thr:.2f}.tsv")
        with out_path.open('w') as f:
            f.write('file\tTP\tFP\tFN\tP\tR\tF1\n')
            for stem, tp, fp, fn, p, r, f1 in all_rows[thr]:
                f.write(f"{stem}\t{tp}\t{fp}\t{fn}\t{p:.3f}\t{r:.3f}\t{f1:.3f}\n")
    print(f"Wrote summaries for {len(thresholds)} thresholds")

if __name__ == '__main__':
    main()
