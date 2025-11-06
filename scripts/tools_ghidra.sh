#!/bin/sh
set -eu

GHIDRA_HOME=${GHIDRA_HOME:-third_party/ghidra_11.4.2_PUBLIC}
OUT_DIR=${OUT_DIR:-out/ghidra}
LOG_DIR=${LOG_DIR:-out/logs}
PROJECT_ROOT=${PROJECT_ROOT:-out/ghidra/projects}

if [ ! -x "$GHIDRA_HOME/support/analyzeHeadless" ]; then
    echo "analyzeHeadless not found at $GHIDRA_HOME/support/analyzeHeadless" >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR" "$PROJECT_ROOT"
OUT_DIR_ABS=$(cd "$OUT_DIR" && pwd)
PROJECT_ROOT_ABS=$(cd "$PROJECT_ROOT" && pwd)

for bin in data/build/linux/O3/*_stripped; do
    if [ ! -f "$bin" ]; then
        continue
    fi
    stem=$(basename "$bin")
    proj_dir="$PROJECT_ROOT_ABS/$stem"
    rm -rf "$proj_dir"
    mkdir -p "$proj_dir"
    log_file="$LOG_DIR/ghidra_${stem}.log"
    bin_abs=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
    "$GHIDRA_HOME/support/analyzeHeadless" "$proj_dir" "${stem}_proj" \
        -import "$bin_abs" \
        -scriptPath tools \
        -postScript ghidra_export_functions.py "$bin_abs" "$OUT_DIR_ABS" \
        -deleteProject > "$log_file" 2>&1
    echo "Exported functions for $bin -> $OUT_DIR/${stem}.csv"
done
