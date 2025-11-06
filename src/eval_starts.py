#!/usr/bin/env python3
import argparse, json, glob, math, statistics

def load_json(p):
    with open(p) as f:
        return json.load(f)

def match_with_tolerance(truth_starts, pred_starts, tol):
    tp = 0
    used = set()
    offsets = []
    for t in truth_starts:
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
            offsets.append(best_dist)
    fp = len(pred_starts) - len(used)
    fn = len(truth_starts) - tp
    return tp, fp, fn, offsets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth_glob", required=True)
    ap.add_argument("--tolerance", type=int, default=8)
    args = ap.parse_args()

    pred = load_json(args.pred)
    pred_starts = [int(x["start"]) for x in pred]

    truth_files = glob.glob(args.truth_glob)
    if not truth_files:
        print("No truth files found for", args.truth_glob)
        return
    # Use the first match
    truth = load_json(truth_files[0])
    truth_starts = [int(x["start"]) for x in truth]

    tp, fp, fn, offsets = match_with_tolerance(truth_starts, pred_starts, args.tolerance)
    prec = tp / (tp + fp) if (tp+fp) else 0.0
    rec = tp / (tp + fn) if (tp+fn) else 0.0
    f1 = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0
    mean_err = statistics.mean(offsets) if offsets else 0.0
    median_err = statistics.median(offsets) if offsets else 0.0
    print(
        f"TP={tp} FP={fp} FN={fn}  |  P={prec:.3f} R={rec:.3f} F1={f1:.3f} (tol={args.tolerance}) | "
        f"mean_err={mean_err:.1f} median_err={median_err:.1f}"
    )

if __name__ == "__main__":
    main()
