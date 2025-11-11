#!/usr/bin/env python3
import argparse
import csv
import statistics


def compute_macro(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        p_vals, r_vals, f_vals = [], [], []
        for row in reader:
            p_vals.append(float(row['P']))
            r_vals.append(float(row['R']))
            f_vals.append(float(row['F1']))
    return (
        statistics.mean(p_vals),
        statistics.mean(r_vals),
        statistics.mean(f_vals),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thresholds', nargs='+', required=True)
    ap.add_argument('--pattern', action='append', nargs=3, metavar=('VERSION','MODE','GLOB'), required=True,
                    help='Pattern should include {thr} placeholder, e.g., out/summary_thr_{thr}_v04_filtered.tsv')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    rows = []
    best = {}
    for version, mode, pattern in args.pattern:
        for thr in args.thresholds:
            path = pattern.format(thr=thr)
            macroP, macroR, macroF1 = compute_macro(path)
            rows.append((version, mode, thr, macroP, macroR, macroF1))
            key = (version, mode)
            score = best.get(key)
            current = (macroF1, macroR, thr)
            if score is None or current > score:
                best[key] = current

    with open(args.out, 'w') as f:
        f.write('version\tmode\tthreshold\tMacroP\tMacroR\tMacroF1\n')
        for version, mode, thr, macroP, macroR, macroF1 in rows:
            f.write(f"{version}\t{mode}\t{thr}\t{macroP:.3f}\t{macroR:.3f}\t{macroF1:.3f}\n")

    for (version, mode), (macroF1, macroR, thr) in best.items():
        print(f"Best {version} {mode}: thr={thr} MacroF1={macroF1:.3f} MacroR={macroR:.3f}")

if __name__ == '__main__':
    main()
