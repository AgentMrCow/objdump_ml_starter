#!/usr/bin/env python3
import argparse
import glob
import json
import os
import pathlib

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from features import candidate_addresses, featurize_point, get_feature_keys


def load_manifest(path):
    with open(path) as f:
        return {entry['name']: entry for entry in json.load(f)}


def collect_files(programs, opt_levels):
    pairs = []
    for opt in opt_levels:
        for prog in programs:
            base = f"data/build/linux/{opt}/{prog}_sym"
            asm = base + ".asm.json"
            label = f"data/labels/linux/{opt}/{prog}_sym.functions_truth.json"
            if os.path.exists(asm) and os.path.exists(label):
                pairs.append((asm, label))
    return pairs


def build_dataset(file_pairs, feature_keys):
    samples = {}
    for asm_path, label_path in file_pairs:
        with open(asm_path) as f:
            asm = json.load(f)
        with open(label_path) as f:
            truth = json.load(f)
        truth_starts = {entry['start'] for entry in truth}
        instrs = asm['instrs']
        addr_to_idx = {ins['addr']: idx for idx, ins in enumerate(instrs)}
        cands = candidate_addresses(asm)
        for addr in cands:
            idx = addr_to_idx.get(addr)
            if idx is None:
                continue
            feats = featurize_point(instrs, idx)
            vec = tuple(feats[k] for k in feature_keys)
            label = 1 if addr in truth_starts else 0
            existing = samples.get(vec)
            if existing is None or label > existing:
                samples[vec] = label
    X = np.array([list(k) for k in samples.keys()], dtype=np.float32)
    y = np.array(list(samples.values()), dtype=np.int32)
    pos = int(y.sum())
    neg = len(y) - pos
    return X, y, pos, neg


def train_models(train_programs, train_opts, out_dir):
    feature_keys = get_feature_keys()
    pairs = collect_files(train_programs, train_opts)
    if not pairs:
        raise SystemExit('No training files found; ensure dataset built and splits correct.')
    X, y, pos, neg = build_dataset(pairs, feature_keys)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f"Train set: total={len(y)} pos={pos} neg={neg}")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    lr_loss = log_loss(y, lr.predict_proba(X))
    dump({'model': lr, 'feature_keys': feature_keys}, os.path.join(out_dir, 'start_detector_v06_logreg.joblib'))
    print(f"LogReg log-loss={lr_loss:.4f}")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    rf_loss = log_loss(y, rf.predict_proba(X))
    print(f"RF log-loss={rf_loss:.4f}")
    dump({'model': rf, 'feature_keys': feature_keys}, os.path.join(out_dir, 'start_detector_v06_rf.joblib'))

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=42,
        n_jobs=4,
    )
    xgb.fit(X, y)
    xgb_loss = log_loss(y, xgb.predict_proba(X))
    print(f"XGB log-loss={xgb_loss:.4f}")
    dump({'model': xgb, 'feature_keys': feature_keys}, os.path.join(out_dir, 'start_detector_v06_xgb.joblib'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='splits/v06.json')
    ap.add_argument('--train_opts', default='O0,O1,O2')
    ap.add_argument('--out_dir', default='models')
    args = ap.parse_args()

    with open(args.split) as f:
        split = json.load(f)
    train_programs = split['train_programs']
    train_opts = [o.strip() for o in args.train_opts.split(',') if o.strip()]

    train_models(train_programs, train_opts, args.out_dir)

if __name__ == '__main__':
    main()
