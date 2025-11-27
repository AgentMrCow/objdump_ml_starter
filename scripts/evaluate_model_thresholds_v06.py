#!/usr/bin/env python3
import argparse
import json
import pathlib
import numpy as np
from joblib import load

from features import candidate_addresses, featurize_point


def load_truth(label_glob):
    import glob
    paths = glob.glob(label_glob)
    truth = set()
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for entry in data:
            truth.add(int(entry['start']))
    return truth


def collect_candidates(bin_path):
    asm_path = bin_path + '.asm.json'
    with open(asm_path) as f:
        asm = json.load(f)
    instrs = asm['instrs']
    addr_to_idx = {ins['addr']: i for i, ins in enumerate(instrs)}
    cands = candidate_addresses(asm)
    feats = []
    addrs = []
    for addr in cands:
        idx = addr_to_idx.get(addr)
        if idx is None:
            continue
        fvec = featurize_point(instrs, idx)
        feats.append(fvec)
        addrs.append(addr)
    return addrs, feats


def score_model(model_path, feature_keys, feats):
    bundle = load(model_path)
    clf = bundle['model']
    keys = bundle['feature_keys'] if 'feature_keys' in bundle else feature_keys
    X = np.array([[f.get(k, 0) for k in keys] for f in feats], dtype=np.float32)
    probs = clf.predict_proba(X)[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X)
    return probs


def eval_thresholds(addrs, probs, truth, thresholds, tol):
    preds_sorted = sorted(zip(addrs, probs), key=lambda x: x[1], reverse=True)
    results = {}
    for thr in thresholds:
        pred_starts = [a for a, p in preds_sorted if p >= thr]
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
                    best = i; best_dist = d
            if best is not None:
                tp += 1; used.add(best)
        fp = len(pred_starts) - len(used)
        fn = len(truth) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        results[thr] = (tp, fp, fn, prec, rec, f1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bins_list', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--tolerance', type=int, default=8)
    ap.add_argument('--thresholds', default='0.20,0.25,0.30,0.33,0.35,0.40,0.45,0.50,0.55')
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',') if x]
    bins = [line.strip() for line in pathlib.Path(args.bins_list).read_text().splitlines() if line.strip()]
    bundle = load(args.model)
    feature_keys = bundle.get('feature_keys', [])

    all_rows = {thr: [] for thr in thresholds}
    for bin_path in bins:
        addrs, feats = collect_candidates(bin_path)
        probs = score_model(args.model, feature_keys, feats)
        stem = pathlib.Path(bin_path).name.replace('_stripped','')
        opt_level = pathlib.Path(bin_path).parts[-2]
        truth = load_truth(f"data/labels/linux/{opt_level}/{stem}_sym.functions_truth.json")
        results = eval_thresholds(addrs, probs, truth, thresholds, args.tolerance)
        for thr, (tp, fp, fn, p, r, f1) in results.items():
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
