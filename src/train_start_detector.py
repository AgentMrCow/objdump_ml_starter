#!/usr/bin/env python3
import argparse, glob, json, os, subprocess, pathlib
from joblib import dump
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from features import candidate_addresses, featurize_point, get_feature_keys

def run_objdump(bin_path, out_json):
    from parse_objdump import run_objdump, parse
    text = run_objdump(bin_path)
    data = parse(text)
    pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    return data

def truth_from_file(path):
    with open(path) as f:
        return json.load(f)

def collect_train_items(train_globs):
    # Expect train_globs to match *_sym binaries
    bins = []
    for pattern in train_globs:
        bins.extend(glob.glob(pattern))
    bins = sorted(set(bins))
    items = []
    for b in bins:
        # derive label path alongside
        # Create labels via elf_labels.py if missing
        label_dir = f"data/labels/linux"
        opt = "O0" if "/O0/" in b else "O3"
        pathlib.Path(f"{label_dir}/{opt}").mkdir(parents=True, exist_ok=True)
        truth_path = f"{label_dir}/{opt}/{os.path.basename(b)}.functions_truth.json"
        if not os.path.exists(truth_path):
            # generate
            subprocess.check_call(["python", "src/elf_labels.py", "--bin", b, "--out", truth_path])
        # parse asm
        asm_json = f"{os.path.dirname(b)}/{os.path.basename(b)}.asm.json"
        if not os.path.exists(asm_json):
            subprocess.check_call(["python", "src/parse_objdump.py", "--bin", b, "--out", asm_json])
        items.append((b, asm_json, truth_path))
    return items

def add_sample(store, vec, label):
    key = tuple(vec)
    existing = store.get(key)
    if existing is None or label > existing:
        store[key] = label


def load_hard_negative_vectors(hn_globs, feature_keys):
    vectors = []
    if not hn_globs:
        return vectors
    for pattern in hn_globs:
        for path in glob.glob(pattern):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            for entry in data:
                feats = entry.get("features")
                if not isinstance(feats, dict):
                    continue
                vectors.append([feats.get(k, 0) for k in feature_keys])
    return vectors

def train(train_globs, hn_globs, model_path):
    items = collect_train_items(train_globs)
    feature_keys = get_feature_keys()
    samples = {}
    raw_pos = 0
    raw_neg = 0
    for b, asm_json, truth_json in items:
        with open(asm_json) as f:
            asm = json.load(f)
        instrs = asm["instrs"]
        addr_to_idx = {ins["addr"]: i for i, ins in enumerate(instrs)}
        cands = candidate_addresses(asm)
        cand_idxs = [addr_to_idx[a] for a in cands if a in addr_to_idx]
        truth = truth_from_file(truth_json)
        truth_starts = {f["start"] for f in truth}
        X = []
        for idx in cand_idxs:
            feats = featurize_point(instrs, idx)
            vec = [feats[k] for k in feature_keys]
            addr = instrs[idx]["addr"]
            label = 1 if addr in truth_starts else 0
            raw_pos += label
            raw_neg += (1 - label)
            add_sample(samples, vec, label)

    hn_vectors = load_hard_negative_vectors(hn_globs, feature_keys)
    for vec in hn_vectors:
        raw_neg += 1
        add_sample(samples, vec, 0)

    X_all = np.array([list(k) for k in samples.keys()], dtype=np.float32)
    y_all = np.array(list(samples.values()), dtype=np.int32)
    unique_pos = int(y_all.sum())
    unique_neg = len(y_all) - unique_pos
    # Simple classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_all, y_all)
    if hasattr(clf, "predict_proba"):
        loss = log_loss(y_all, clf.predict_proba(X_all))
    else:
        loss = float("nan")
    dump({"model": clf, "feature_keys": feature_keys}, model_path)
    print(f"Raw counts -> pos:{raw_pos} neg:{raw_neg}")
    print(f"Deduped counts -> pos:{unique_pos} neg:{unique_neg} (total {len(y_all)})")
    print(f"Training log-loss={loss:.4f} iterations={clf.n_iter_[0] if hasattr(clf, 'n_iter_') else 'n/a'}")
    print(f"Trained on {len(y_all)} samples. Saved -> {model_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_glob", required=True, action="append", metavar="GLOB",
                    help="Repeat per glob, e.g., --train_glob data/build/*/O0/*_sym --train_glob data/build/*/O3/*_sym")
    ap.add_argument("--hn_json_glob", action="append", metavar="HNGLOB",
                    help="Repeat per glob for hard-negative JSON files")
    ap.add_argument("--model_path", default="models/start_detector.joblib")
    args = ap.parse_args()
    pathlib.Path(os.path.dirname(args.model_path)).mkdir(parents=True, exist_ok=True)
    train(args.train_glob, args.hn_json_glob, args.model_path)

if __name__ == "__main__":
    main()
