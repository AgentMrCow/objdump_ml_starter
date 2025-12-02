#!/usr/bin/env python3
import argparse, json, os, pathlib
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from features import candidate_addresses, featurize_point, get_feature_keys

def collect_files(programs, opts):
    pairs=[]
    for opt in opts:
        for prog in programs:
            asm=f"data/build/linux/{opt}/{prog}_sym.asm.json"
            label=f"data/labels/linux/{opt}/{prog}_sym.functions_truth.json"
            if os.path.exists(asm) and os.path.exists(label):
                pairs.append((asm,label))
    return pairs

def build_dataset(pairs, keys):
    samples={}
    for asm_path,label_path in pairs:
        asm=json.load(open(asm_path))
        truth=json.load(open(label_path))
        truth_starts={e['start'] for e in truth}
        instrs=asm['instrs']
        idx_map={ins['addr']: i for i, ins in enumerate(instrs)}
        cands=candidate_addresses(asm)
        for addr in cands:
            idx=idx_map.get(addr)
            if idx is None: continue
            feats=featurize_point(instrs, idx)
            vec=tuple(feats.get(k,0) for k in keys)
            label=1 if addr in truth_starts else 0
            if label==1 or vec not in samples:
                samples[vec]=label
    X=np.array([list(k) for k in samples.keys()], dtype=np.float32)
    y=np.array(list(samples.values()), dtype=np.int32)
    pos=int(y.sum()); neg=len(y)-pos
    return X,y,pos,neg

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split', default='splits/v06.json')
    ap.add_argument('--opts', default='O0,O1,O2,O3')
    ap.add_argument('--out_dir', default='models')
    ap.add_argument('--tag', default='v06e', help="suffix tag for model filenames")
    args=ap.parse_args()

    split=json.load(open(args.split))
    train_programs=split['train_programs']
    opts=[o.strip() for o in args.opts.split(',') if o.strip()]
    keys=get_feature_keys()

    pairs=collect_files(train_programs, opts)
    X,y,pos,neg=build_dataset(pairs, keys)
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Train (cfg) total={len(y)} pos={pos} neg={neg}")

    rf=RandomForestClassifier(n_estimators=700,max_depth=None,n_jobs=-1,random_state=42,class_weight='balanced_subsample')
    rf.fit(X,y)
    rf_loss=log_loss(y, rf.predict_proba(X))
    dump({'model': rf, 'feature_keys': keys}, os.path.join(args.out_dir,f'start_detector_{args.tag}_rf.joblib'))
    print(f"RF cfg log-loss={rf_loss:.4f}")

    xgb=XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='hist',
        random_state=42,
        n_jobs=4,
        scale_pos_weight=max(1.0, neg/pos)
    )
    xgb.fit(X,y)
    xgb_loss=log_loss(y, xgb.predict_proba(X))
    dump({'model': xgb, 'feature_keys': keys}, os.path.join(args.out_dir,f'start_detector_{args.tag}_xgb.joblib'))
    print(f"XGB cfg log-loss={xgb_loss:.4f}")

if __name__=='__main__':
    main()
