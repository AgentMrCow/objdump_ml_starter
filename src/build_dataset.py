#!/usr/bin/env python3
import os, subprocess, pathlib, sys

SAMPLES = [
    ("hello", "samples/hello.c", True),
    ("mathlib", "samples/mathlib.c", False),
    ("sort", "samples/sort.c", True),
]

def build_one(name, src, has_main, opt, outdir):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    sym = os.path.join(outdir, f"{name}_sym")
    stripped = os.path.join(outdir, f"{name}_stripped")
    cflags = ["-Wall", f"-{opt}", "-fno-omit-frame-pointer"]
    if has_main:
        cmd_sym = ["gcc", src, "-o", sym] + cflags + ["-no-pie"]
    else:
        obj = os.path.join(outdir, f"{name}.o")
        subprocess.check_call(["gcc", "-c", src, "-o", obj] + cflags + ["-fPIC"])
        cmd_sym = ["gcc", obj, "-shared", "-o", sym]
    print("BUILD:", " ".join(cmd_sym))
    subprocess.check_call(cmd_sym)
    # stripped
    subprocess.check_call(["strip", "-s", "-o", stripped, sym])
    return sym, stripped

def main():
    base = "data/build/linux"
    opt_levels = ["O0", "O1", "O2", "O3"]
    for opt in opt_levels:
        for (name, src, has_main) in SAMPLES:
            sym, stripped = build_one(name, src, has_main, opt, f"{base}/{opt}")
            # also prepare labels for the sym build
            label_out = f"data/labels/linux/{opt}/{os.path.basename(sym)}.functions_truth.json"
            pathlib.Path(os.path.dirname(label_out)).mkdir(parents=True, exist_ok=True)
            subprocess.check_call(["python", "src/elf_labels.py", "--bin", sym, "--out", label_out])
            # also parse asm for both
            for b in [sym, stripped]:
                asm_out = f"{os.path.dirname(b)}/{os.path.basename(b)}.asm.json"
                subprocess.check_call(["python", "src/parse_objdump.py", "--bin", b, "--out", asm_out])
    print("Dataset build complete.")

if __name__ == "__main__":
    main()
