# objdump-ML Starter (Function Start Detection)

This is a minimal, **runnable** starter kit that augments `objdump` (linear sweep)
with a tiny ML model to detect **function starts** on x86-64 ELF binaries.

> Goal: show how simple ML can improve a simple baseline without building a new disassembler.

## Quickstart (demo)

```bash
cd objdump_ml_starter

# 1) create venv & install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) build tiny dataset (O0 & O3, with symbols and stripped)
python src/build_dataset.py

# 3) train start detector on O0 symbol binaries
python src/train_start_detector.py --train_glob "data/build/*/O0/*_sym" --model_path models/start_detector.joblib

# 4) run prediction on a stripped O3 binary
python src/predict_starts.py --bin "data/build/linux/O3/hello_stripped" --model_path models/start_detector.joblib --out functions_pred.json

# 5) evaluate against ground truth (from the _sym build of the same source/opt)
#    (tolerance default: 8 bytes)
python src/eval_starts.py --pred functions_pred.json --truth_glob "data/labels/*/O3/hello_sym.functions_truth.json" --tolerance 8
```

## Project layout
```
src/
  build_dataset.py      # builds sample binaries (O0/O3), symbol & stripped
  parse_objdump.py      # parses `objdump -d` to JSON
  elf_labels.py         # extracts function start/end from DWARF/.symtab
  features.py           # feature extraction & candidate generation
  train_start_detector.py
  predict_starts.py
  eval_starts.py
samples/
  hello.c, mathlib.c, sort.c
data/
  build/...             # binaries produced here
  labels/...            # truth JSON files produced here
models/
  start_detector.joblib # saved model (after training)
```

## Notes
- This is **not** a production disassembler. It’s a small educational pipeline.
- Start with the sample programs; then point scripts at your own binaries.
- For PE/ARM etc., extend later—Term 1 scope is x86‑64 ELF.
