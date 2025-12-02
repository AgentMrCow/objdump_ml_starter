#!/usr/bin/env python3
import argparse, json, pathlib
import numpy as np
from joblib import load
from features import candidate_addresses, featurize_point, get_feature_keys

def load_truth(label_glob):
    import glob
    starts = set()
    for path in glob.glob(label_glob):
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            starts.add(int(entry['start']))
    return starts

def collect_candidates(asm_path, cond_targets=True):
    with open(asm_path) as f:
        asm = json.load(f)
    instrs = asm['instrs']
    addr_to_idx = {ins['addr']: i for i, ins in enumerate(instrs)}
    if cond_targets:
        cands = candidate_addresses(asm)
    else:
        # rebuild candidates without conditional targets
        # naive: filter out branch-derived candidates
        cands = []
        addr_set = {ins['addr'] for ins in instrs}
        cands.append(instrs[0]['addr'])
        for ins in instrs:
            for t in ins.get('xrefs_out', []):
                if isinstance(t, int):
                    cands.append(t)
        for i, ins in enumerate(instrs[:-1]):
            if ins['mnemonic'].startswith('ret'):
                cands.append(instrs[i+1]['addr'])
        cands = sorted({c for c in cands if c in addr_set})
    feats = []
    addrs = []
    for addr in cands:
        idx = addr_to_idx.get(addr)
        if idx is None:
            continue
        feats.append(featurize_point(instrs, idx))
        addrs.append(addr)
    return addrs, feats


def score(bundle, feats):
    keys = bundle['feature_keys']
    clf = bundle['model']
    X = np.array([[f.get(k,0) for k in keys] for f in feats], dtype=np.float32)
    return clf.predict_proba(X)[:,1] if hasattr(clf,'predict_proba') else clf.decision_function(X)


def eval_threshold(addrs, probs, truth, thr, tol):
    preds = [a for a,p in zip(addrs, probs) if p >= thr]
    used = set()
    tp = 0
    for t in truth:
        best=None; bestd=None
        for i,p in enumerate(preds):
            if i in used: continue
            d=abs(p - t)
            if d <= tol and (bestd is None or d<bestd):
                best=i; bestd=d
        if best is not None:
            used.add(best); tp+=1
    fp = len(preds)-len(used)
    fn = len(truth)-tp
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
    return tp,fp,fn,prec,rec,f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bins_list', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--threshold', type=float, default=0.40)
    ap.add_argument('--tol', type=int, default=8)
    ap.add_argument('--cond_targets', action='store_true')
    ap.add_argument('--post_filter', choices=['on','off'], default='on')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    bundle = load(args.model)
    rows=[]
    bins=[line.strip() for line in pathlib.Path(args.bins_list).read_text().splitlines() if line.strip()]
    for bin_path in bins:
        asm_path = bin_path + '.asm.json'
        opt_level = pathlib.Path(bin_path).parts[-2]
        stem = pathlib.Path(bin_path).name.replace('_stripped','')
        truth = load_truth(f"data/labels/linux/{opt_level}/{stem}_sym.functions_truth.json")
        addrs, feats = collect_candidates(asm_path, cond_targets=args.cond_targets)
        probs = score(bundle, feats)
        # simple post-filter on padding
        if args.post_filter == 'on':
            filtered = []
            filtered_addrs = []
            for addr, fvec, p in zip(addrs, feats, probs):
                if fvec.get('xrefs_in',0)==0 and fvec.get('padding_nop_run',0)>=3 and fvec.get('window2_xrefs_in',0)==0:
                    continue
                filtered.append(p)
                filtered_addrs.append(addr)
            addrs = filtered_addrs
            probs = np.array(filtered)
        tp, fp, fn, prec, rec, f1 = eval_threshold(addrs, probs, truth, args.threshold, args.tol)
        rows.append((stem,opt_level,tp,fp,fn,prec,rec,f1))

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,'w') as f:
        f.write('file\topt\tTP\tFP\tFN\tP\tR\tF1\n')
        for r in rows:
            f.write('\t'.join([str(x) if not isinstance(x,float) else f"{x:.3f}" for x in r])+'\n')
    print(f"Wrote {args.out} ({len(rows)} bins)")

if __name__=='__main__':
    main()
