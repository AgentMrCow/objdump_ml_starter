#!/usr/bin/env python3
import argparse
import csv
import statistics
from pathlib import Path

def compute_macro(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        p_vals, r_vals, f_vals = [], [], []
        for row in reader:
            p_vals.append(float(row['P']))
            r_vals.append(float(row['R']))
            f_vals.append(float(row['F1']))
    return statistics.mean(p_vals), statistics.mean(r_vals), statistics.mean(f_vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patterns', nargs='+', required=True,
                    help='e.g., out/summary_thr_v06_logreg_O3_{thr}.tsv')
    ap.add_argument('--thresholds', nargs='+', required=True)
    ap.add_argument('--out', default='out/macro_v06.tsv')
    args = ap.parse_args()

    rows = []
    best = {}
    for pattern in args.patterns:
        name = Path(pattern).name.replace('{thr}', '').replace('.tsv','')
        for thr in args.thresholds:
            path = pattern.replace('{thr}', thr)
            macroP, macroR, macroF1 = compute_macro(path)
            rows.append((name, thr, macroP, macroR, macroF1))
            key = name
            val = (macroF1, macroR, thr)
            if key not in best or val > best[key]:
                best[key] = val

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        f.write('name\tthreshold\tMacroP\tMacroR\tMacroF1\n')
        for name, thr, p, r, f1 in rows:
            f.write(f"{name}\t{thr}\t{p:.3f}\t{r:.3f}\t{f1:.3f}\n")
    for name, (f1, r, thr) in best.items():
        print(f"Best {name}: thr={thr} MacroF1={f1:.3f} MacroR={r:.3f}")

if __name__ == '__main__':
    main()
