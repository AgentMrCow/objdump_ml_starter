#!/bin/sh
set -eu

OUT_DIR="out"
LOG_DIR="$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.tsv"
MODEL_PATH="models/start_detector.joblib"
BIN_GLOB="data/build/linux/O3"
TOL=8

mkdir -p "$OUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_batch_predict.log"
: > "$LOG_FILE"

printf 'file\tTP\tFP\tFN\tP\tR\tF1\n' > "$SUMMARY"

for bin in "$BIN_GLOB"/*_stripped; do
    if [ ! -f "$bin" ]; then
        continue
    fi
    base=$(basename "$bin")
    stem=${base%_stripped}
    pred_out="$OUT_DIR/${stem}.json"

    echo "Predicting for $bin" >> "$LOG_FILE"
    python src/predict_starts.py --bin "$bin" --model_path "$MODEL_PATH" --out "$pred_out" >> "$LOG_FILE" 2>&1

    eval_output=$(python src/eval_starts.py --pred "$pred_out" --truth_glob "data/labels/*/O3/${stem}_sym.functions_truth.json" --tolerance "$TOL")
    echo "$eval_output" >> "$LOG_FILE"

    # Parse eval_output (format: TP=.. FP=.. FN=..  |  P=.. R=.. F1=.. (tol=..))
    set -- $eval_output
    tp=${1#TP=}
    fp=${2#FP=}
    fn=${3#FN=}
    p=${5#P=}
    r=${6#R=}
    f1=${7#F1=}

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$stem" "$tp" "$fp" "$fn" "$p" "$r" "$f1" >> "$SUMMARY"

done

cat "$SUMMARY"
