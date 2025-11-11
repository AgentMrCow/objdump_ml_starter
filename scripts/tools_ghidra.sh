#!/bin/bash
set -eu

GHIDRA_HOME=${GHIDRA_HOME:-third_party/ghidra_11.4.2_PUBLIC}
OUT_DIR=${OUT_DIR:-out/ghidra}
LOG_DIR=${LOG_DIR:-out/logs}
PROJECT_ROOT=${PROJECT_ROOT:-out/ghidra/projects}
BIN_GLOB=${BIN_GLOB:-data/build/linux/O3/*_stripped}
GHIDRA_USER_HOME=${GHIDRA_USER_HOME:-out/ghidra/user_home}

if [ ! -x "$GHIDRA_HOME/support/analyzeHeadless" ]; then
    echo "analyzeHeadless not found at $GHIDRA_HOME/support/analyzeHeadless" >&2
    exit 1
fi

mkdir -p "$OUT_DIR" "$LOG_DIR" "$PROJECT_ROOT" "$GHIDRA_USER_HOME"
OUT_DIR_ABS=$(cd "$OUT_DIR" && pwd)
PROJECT_ROOT_ABS=$(cd "$PROJECT_ROOT" && pwd)
USER_HOME_ABS=$(cd "$GHIDRA_USER_HOME" && pwd)

for bin in $BIN_GLOB; do
    if [ ! -f "$bin" ]; then
        continue
    fi
    stem=$(basename "$bin")
    opt_level=$(basename "$(dirname "$bin")")
    proj_dir="$PROJECT_ROOT_ABS/${opt_level}_${stem}"
    rm -rf "$proj_dir"
    mkdir -p "$proj_dir"
    log_file="$LOG_DIR/ghidra_${opt_level}_${stem}.log"
    bin_abs=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
    csv_name="${opt_level}_${stem}.csv"
    _JAVA_OPTIONS="-Duser.home=$USER_HOME_ABS" "$GHIDRA_HOME/support/analyzeHeadless" "$proj_dir" "${opt_level}_${stem}_proj" \
        -import "$bin_abs" \
        -scriptPath tools \
        -postScript ghidra_export_functions.py "$bin_abs" "$OUT_DIR_ABS/$csv_name" \
        -deleteProject > "$log_file" 2>&1
    echo "Exported functions for $bin -> $OUT_DIR/$csv_name"
done
