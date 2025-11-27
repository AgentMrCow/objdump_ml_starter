#!/usr/bin/env python3
import os, subprocess, pathlib, sys, json, argparse

DEFAULT_SAMPLES = [
    ("hello", "samples/hello.c", True),
    ("mathlib", "samples/mathlib.c", False),
    ("sort", "samples/sort.c", True),
]

def load_samples():
    manifest_path = pathlib.Path('data/program_manifest_v06.json')
    if manifest_path.exists():
        with manifest_path.open() as f:
            entries = json.load(f)
        samples = []
        for entry in entries:
            samples.append((entry["name"], entry["src"], bool(entry.get("has_main", True))))
        return samples
    return DEFAULT_SAMPLES

SAMPLES = load_samples()

def build_one(name, src, has_main, opt, outdir):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    sym = os.path.join(outdir, f"{name}_sym")
    stripped = os.path.join(outdir, f"{name}_stripped")
    cflags = ["-Wall", f"-{opt}", "-fno-omit-frame-pointer"]
    if has_main:
        cmd_sym = ["gcc", src, "-o", sym] + cflags + ["-lm", "-no-pie"]
    else:
        obj = os.path.join(outdir, f"{name}.o")
        try:
            subprocess.check_call(["gcc", "-c", src, "-o", obj] + cflags + ["-fPIC"])
        except subprocess.CalledProcessError as exc:
            print(f"[build] FAIL compile-object {name} {opt}: {exc}", file=sys.stderr)
            return None, None
        cmd_sym = ["gcc", obj, "-shared", "-o", sym]
    print("BUILD:", " ".join(cmd_sym))
    try:
        subprocess.check_call(cmd_sym)
    except subprocess.CalledProcessError as exc:
        print(f"[build] FAIL link {name} {opt}: {exc}", file=sys.stderr)
        return None, None
    # stripped
    try:
        subprocess.check_call(["strip", "-s", "-o", stripped, sym])
    except subprocess.CalledProcessError as exc:
        print(f"[build] FAIL strip {name} {opt}: {exc}", file=sys.stderr)
        return None, None
    return sym, stripped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--opt_levels', default='O0,O1,O2,O3')
    args = parser.parse_args()

    base = "data/build/linux"
    opt_levels = [opt.strip() for opt in args.opt_levels.split(',') if opt.strip()]
    subset = SAMPLES[args.start:args.end]
    print(f"Building {len(subset)} programs across opts {opt_levels}")
    for opt in opt_levels:
        for (name, src, has_main) in subset:
            result = build_one(name, src, has_main, opt, f"{base}/{opt}")
            if not result or not all(result):
                continue
            sym, stripped = result
            # also prepare labels for the sym build
            label_out = f"data/labels/linux/{opt}/{os.path.basename(sym)}.functions_truth.json"
            pathlib.Path(os.path.dirname(label_out)).mkdir(parents=True, exist_ok=True)
            try:
                subprocess.check_call(["python", "src/elf_labels.py", "--bin", sym, "--out", label_out])
            except subprocess.CalledProcessError as exc:
                print(f"[label] FAIL {name} {opt}: {exc}", file=sys.stderr)
                continue
            # also parse asm for both
            for b in [sym, stripped]:
                asm_out = f"{os.path.dirname(b)}/{os.path.basename(b)}.asm.json"
                try:
                    subprocess.check_call(["python", "src/parse_objdump.py", "--bin", b, "--out", asm_out])
                except subprocess.CalledProcessError as exc:
                    print(f"[asm] FAIL {name} {opt}: {exc}", file=sys.stderr)
                    break
    print("Dataset build complete.")

if __name__ == "__main__":
    main()
