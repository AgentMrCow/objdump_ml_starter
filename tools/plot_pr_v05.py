#!/usr/bin/env python3
import argparse
import csv
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_macro(path):
    groups = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            key = (row['version'], row['mode'])
            groups.setdefault(key, []).append({
                'threshold': float(row['threshold']),
                'P': float(row['MacroP']),
                'R': float(row['MacroR']),
                'F1': float(row['MacroF1']),
            })
    for key in groups:
        groups[key].sort(key=lambda x: x['threshold'])
    return groups


def plot_group(key, data, out_dir):
    version, mode = key
    thresholds = [d['threshold'] for d in data]
    precisions = [d['P'] for d in data]
    recalls = [d['R'] for d in data]
    f1s = [d['F1'] for d in data]

    plt.figure()
    plt.plot(recalls, precisions, marker='o')
    for r, p, t in zip(recalls, precisions, thresholds):
        plt.annotate(f"{t:.2f}", (r, p))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{version} {mode} P-R curve')
    pr_path = pathlib.Path(out_dir) / f'pr_{version}_{mode}.png'
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()

    plt.figure()
    plt.plot(thresholds, f1s, marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Macro F1')
    plt.title(f'{version} {mode} F1 vs Threshold')
    plt.grid(True)
    plt.tight_layout()
    f1_path = pathlib.Path(out_dir) / f'f1_{version}_{mode}.png'
    plt.savefig(f1_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--macro', default='out/macro_v05.tsv')
    ap.add_argument('--out_dir', default='out/plots')
    args = ap.parse_args()

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    groups = load_macro(args.macro)
    for key, data in groups.items():
        plot_group(key, data, args.out_dir)
    print(f'Wrote plots for {len(groups)} groups -> {args.out_dir}')

if __name__ == '__main__':
    main()
