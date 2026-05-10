#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from elftools.elf.constants import SH_FLAGS
from elftools.elf.elffile import ELFFile


REPO = Path(__file__).resolve().parents[1]
DISA_ROOT = REPO / "tmp" / "external" / "Disa" / "Disa_task1_2"
if str(DISA_ROOT) not in sys.path:
    sys.path.insert(0, str(DISA_ROOT))

from DisaModel import DisaModel  # type: ignore  # noqa: E402
from preprocess import get_one_file_data  # type: ignore  # noqa: E402


SEQ_LEN = 512
BATCH_SIZE = 16
D_MODEL = 384
VOCAB_SIZE = 2308
DROPOUT_RATE = 0.1
NUM_HEADS = 8
NUM_LAYERS = 6


def load_truth(path: Path) -> list[int]:
    with path.open() as f:
        data = json.load(f)
    return [int(entry["start"]) for entry in data if int(entry["start"]) > 0]


def score_predicted_starts(pred_starts: list[int], truth: list[int], tol: int) -> tuple[int, int, int, float, float, float]:
    used = set()
    tp = 0
    for t in truth:
        best = None
        best_dist = None
        for idx, pred in enumerate(pred_starts):
            if idx in used:
                continue
            dist = abs(pred - t)
            if dist <= tol and (best_dist is None or dist < best_dist):
                best = idx
                best_dist = dist
        if best is not None:
            used.add(best)
            tp += 1
    fp = len(pred_starts) - len(used)
    fn = len(truth) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return tp, fp, fn, prec, rec, f1


def build_model(model_path: Path, device: torch.device) -> DisaModel:
    model = DisaModel(
        embed_size=D_MODEL,
        vocab_size=VOCAB_SIZE,
        ins_seq_len=SEQ_LEN,
        dropout=DROPOUT_RATE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    ).to(device)
    map_location = torch.device("cpu") if device.type == "cpu" else None
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.eval()
    return model


def predict_file(model: DisaModel, file_path: Path, device: torch.device) -> list[int]:
    data, _, addrs = get_one_file_data(str(file_path))
    total = len(addrs)
    if total == 0:
        return []

    seq_count = math.ceil(total / SEQ_LEN)
    predicted = []
    with torch.no_grad():
        for seq_base in range(0, seq_count, BATCH_SIZE):
            seq_limit = min(seq_count, seq_base + BATCH_SIZE)
            batch_rows = []
            seq_slices = []
            for seq_idx in range(seq_base, seq_limit):
                lo = seq_idx * SEQ_LEN
                hi = min(total, lo + SEQ_LEN)
                chunk = data[lo:hi]
                seq_slices.append((lo, hi))
                if hi - lo < SEQ_LEN:
                    pad = np.zeros((SEQ_LEN - (hi - lo), data.shape[1]), dtype=data.dtype)
                    chunk = np.vstack([chunk, pad])
                batch_rows.append(chunk)
            batch = torch.LongTensor(np.stack(batch_rows)).to(device)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy().reshape(len(batch_rows), SEQ_LEN)
            for row_idx, (lo, hi) in enumerate(seq_slices):
                seq_preds = preds[row_idx][: hi - lo]
                seq_addrs = addrs[lo:hi]
                predicted.extend(addr for addr, pred in zip(seq_addrs, seq_preds) if int(pred) == 1)
    return sorted(set(int(addr) for addr in predicted))


def infer_stem(txt_name: str) -> str:
    if txt_name.endswith("_sym.txt"):
        return txt_name[: -len("_sym.txt")]
    return txt_name.rsplit(".", 1)[0]


def executable_section_layout(sym_path: Path) -> list[tuple[int, int, int]]:
    layout = []
    concat_offset = 0
    with sym_path.open("rb") as f:
        elf = ELFFile(f)
        for section in elf.iter_sections():
            if not (section["sh_flags"] & SH_FLAGS.SHF_ALLOC) or section.data_size == 0:
                continue
            if not (section["sh_flags"] & SH_FLAGS.SHF_EXECINSTR):
                continue
            layout.append((concat_offset, concat_offset + section.data_size, int(section["sh_addr"])))
            concat_offset += section.data_size
    return layout


def offsets_to_virtual(offsets: list[int], layout: list[tuple[int, int, int]]) -> list[int]:
    virtual = []
    for offset in offsets:
        mapped = None
        for lo, hi, base in layout:
            if lo <= offset < hi:
                mapped = base + (offset - lo)
                break
        if mapped is not None:
            virtual.append(mapped)
    return sorted(set(virtual))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="out/disa_eval/O3_test")
    ap.add_argument("--model_path", default="tmp/external/Disa/T1/models/Disa_ELF_x64.pkl")
    ap.add_argument("--opt", default="O3")
    ap.add_argument("--tolerance", type=int, default=8)
    ap.add_argument("--out_per_bin", default="out/disa_function_start_o3.tsv")
    ap.add_argument("--out_summary", default="out/disa_function_start_o3_summary.tsv")
    args = ap.parse_args()

    data_dir = REPO / args.data_dir
    model_path = REPO / args.model_path
    out_per_bin = REPO / args.out_per_bin
    out_summary = REPO / args.out_summary
    out_per_bin.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_path, device)

    rows = []
    txt_files = sorted(Path(p) for p in glob.glob(str(data_dir / "*.txt")))
    for txt_path in txt_files:
        stem = infer_stem(txt_path.name)
        truth_path = REPO / "data" / "labels" / "linux" / args.opt / f"{stem}_sym.functions_truth.json"
        sym_path = REPO / "data" / "build" / "linux" / args.opt / f"{stem}_sym"
        if not truth_path.exists() or not sym_path.exists():
            continue
        pred_offsets = predict_file(model, txt_path, device)
        pred = offsets_to_virtual(pred_offsets, executable_section_layout(sym_path))
        truth = load_truth(truth_path)
        tp, fp, fn, p, r, f1 = score_predicted_starts(pred, truth, args.tolerance)
        rows.append(
            {
                "file": stem,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "P": p,
                "R": r,
                "F1": f1,
                "predicted": len(pred),
                "truth": len(truth),
            }
        )

    with out_per_bin.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "TP", "FP", "FN", "P", "R", "F1", "predicted", "truth"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    macro_p = statistics.mean(row["P"] for row in rows) if rows else 0.0
    macro_r = statistics.mean(row["R"] for row in rows) if rows else 0.0
    macro_f1 = statistics.mean(row["F1"] for row in rows) if rows else 0.0
    with out_summary.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "binaries", "MacroP", "MacroR", "MacroF1"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "system": "disa_function_start_metric",
                "binaries": len(rows),
                "MacroP": f"{macro_p:.6f}",
                "MacroR": f"{macro_r:.6f}",
                "MacroF1": f"{macro_f1:.6f}",
            }
        )

    print(
        f"Disa tolerant-start macro on {len(rows)} {args.opt} binaries: "
        f"P={macro_p:.4f} R={macro_r:.4f} F1={macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()
