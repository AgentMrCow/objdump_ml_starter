#!/bin/bash
set -eu

OUT_DIR="out"
LOG_DIR="$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.tsv"
MODEL_PATH=${MODEL_PATH:-models/start_detector.joblib}
BIN_GLOB=${BIN_GLOB:-data/build/linux/O3/*_stripped}
TOL=8

if [ -n "${THRESH+x}" ]; then
    PRED_THRESH="$THRESH"
else
    PRED_THRESH="0.50"
fi

POST_FILTER=${POST_FILTER:-on}

mkdir -p "$OUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_batch_predict.log"
: > "$LOG_FILE"

printf 'file\tTP\tFP\tFN\tP\tR\tF1\tmean_err\tmedian_err\n' > "$SUMMARY"

for bin in $BIN_GLOB; do
    if [ ! -f "$bin" ]; then
        continue
    fi
    base=$(basename "$bin")
    stem=${base%_stripped}
    pred_out="$OUT_DIR/${stem}.json"

    echo "Predicting for $bin (threshold=$PRED_THRESH, post_filter=$POST_FILTER)" >> "$LOG_FILE"
    python src/predict_starts.py --bin "$bin" --model_path "$MODEL_PATH" --out "$pred_out" --threshold "$PRED_THRESH" --post_filter "$POST_FILTER" >> "$LOG_FILE" 2>&1

    opt_level=$(basename "$(dirname "$bin")")
    label_path="data/labels/linux/${opt_level}/${stem}_sym.functions_truth.json"
    eval_output=$(python src/eval_starts.py --pred "$pred_out" --truth_glob "$label_path" --tolerance "$TOL")
    echo "$eval_output" >> "$LOG_FILE"

    # Parse eval_output tokens
    set -- $eval_output
    tp=0
    fp=0
    fn=0
    p=0
    r=0
    f1=0
    mean_err=0
    median_err=0
    for token in "$@"; do
        case "$token" in
            TP=*) tp=${token#TP=};;
            FP=*) fp=${token#FP=};;
            FN=*) fn=${token#FN=};;
            P=*) p=${token#P=};;
            R=*) r=${token#R=};;
            F1=*) f1=${token#F1=};;
            mean_err=*) mean_err=${token#mean_err=};;
            median_err=*) median_err=${token#median_err=};;
        esac
    done

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$stem" "$tp" "$fp" "$fn" "$p" "$r" "$f1" "$mean_err" "$median_err" >> "$SUMMARY"

done

cat "$SUMMARY"
