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

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from features import candidate_addresses, featurize_point, get_feature_keys

def load_hard_negative_vectors(patterns, feature_keys):
    vectors = []
    if not patterns:
        return vectors
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for entry in data:
                feats = entry.get("features")
                if isinstance(feats, dict):
                    vectors.append([feats.get(k, 0) for k in feature_keys])
    return vectors

def collect_program_files(programs, opt_levels):
    paths = []
    for opt in opt_levels:
        for prog in programs:
            base = f"data/build/linux/{opt}/{prog}_sym"
            asm = base + ".asm.json"
            label = f"data/labels/linux/{opt}/{prog}_sym.functions_truth.json"
            if os.path.exists(asm) and os.path.exists(label):
                paths.append((asm, label))
    return paths


def build_dataset(file_pairs, feature_keys):
    samples = {}
    for asm_path, label_path in file_pairs:
        with open(asm_path) as f:
            asm = json.load(f)
        with open(label_path) as f:
            truth = json.load(f)
        truth_starts = {entry["start"] for entry in truth}
        instrs = asm["instrs"]
        addr_to_idx = {ins["addr"]: idx for idx, ins in enumerate(instrs)}
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
    return samples


def add_vectors(samples, vectors):
    for vec in vectors:
        samples.setdefault(tuple(vec), 0)


def finalize_arrays(samples):
    X = np.array([list(k) for k in samples.keys()], dtype=np.float32)
    y = np.array(list(samples.values()), dtype=np.int32)
    pos = int(y.sum())
    neg = len(y) - pos
    return X, y, pos, neg


def train_models(train_programs, opt_levels, hn_patterns, out_dir):
    feature_keys = get_feature_keys()
    files = collect_program_files(train_programs, opt_levels)
    if not files:
        raise SystemExit("No training files found; ensure dataset is built.")
    samples = build_dataset(files, feature_keys)
    hn_vectors = load_hard_negative_vectors(hn_patterns, feature_keys)
    add_vectors(samples, hn_vectors)
    X, y, pos, neg = finalize_arrays(samples)
    print(f"Training dataset: total={len(y)} pos={pos} neg={neg}")

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X, y)
    lr_loss = log_loss(y, logreg.predict_proba(X))
    print(f"LogReg log-loss={lr_loss:.4f}")
    dump({"model": logreg, "feature_keys": feature_keys}, os.path.join(out_dir, "start_detector_v05_logreg.joblib"))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    rf_probs = rf.predict_proba(X)
    rf_loss = log_loss(y, rf_probs)
    importances = rf.feature_importances_
    print(f"RandomForest log-loss={rf_loss:.4f}; top features:")
    top_idx = np.argsort(importances)[-5:][::-1]
    for idx in top_idx:
        print(f"  {feature_keys[idx]}: {importances[idx]:.4f}")
    dump({"model": rf, "feature_keys": feature_keys}, os.path.join(out_dir, "start_detector_v05_rf.joblib"))

    # XGBoost (optional)
    if XGBClassifier is None:
        print("XGBoost unavailable (module import failed); skipping.")
    else:
        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=42,
        )
        xgb.fit(X, y)
        xgb_loss = log_loss(y, xgb.predict_proba(X))
        print(f"XGBoost log-loss={xgb_loss:.4f}")
        dump({"model": xgb, "feature_keys": feature_keys}, os.path.join(out_dir, "start_detector_v05_xgb.joblib"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits/v05.json")
    ap.add_argument("--opt_levels", default="O0,O1,O2")
    ap.add_argument("--hn_glob", action="append", default=["out/mining/*_hardnegs.json"])
    ap.add_argument("--out_dir", default="models")
    args = ap.parse_args()

    with open(args.splits) as f:
        split = json.load(f)
    train_programs = split.get("train_programs", [])
    if not train_programs:
        raise SystemExit("split file missing train_programs")
    opt_levels = args.opt_levels.split(",")

    train_models(train_programs, opt_levels, args.hn_glob, args.out_dir)

if __name__ == "__main__":
    main()
