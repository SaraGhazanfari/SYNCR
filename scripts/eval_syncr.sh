#!/bin/bash
# Evaluate one model on all eight SYNCR tasks with the settings used in the paper.
#
#   MODEL=Qwen/Qwen3-VL-8B-Instruct bash scripts/eval_syncr.sh
#
# Environment variables:
#   MODEL         Hugging Face id or API model name (required)
#   N             items per task (default 500; the paper's API models and human baseline use 100)
#   ADAPTER_PATH  optional LoRA adapter applied on top of MODEL
#   CONDITION     video (default) | novideo | single  -- the evidence controls
#   BATCH_SIZE    overrides each config's batch size (e.g. 1 for LLaVA-OneVision-7B)
#   CONFIGS       space-separated config paths (default: all of configs/*.conf)
#
# Per-item predictions go to eval_predictions/hf/<model tag>/<task>__n<N>.jsonl, and a summary line per
# task is appended to eval_results/<model tag>.tsv. Then run eval.stats_report on the predictions.
set -uo pipefail

MODEL="${MODEL:?set MODEL, e.g. MODEL=Qwen/Qwen3-VL-8B-Instruct}"
N="${N:-500}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
CONDITION="${CONDITION:-video}"
CONFIGS="${CONFIGS:-$(ls configs/*.conf)}"

EXTRA=()
[ -n "$ADAPTER_PATH" ] && EXTRA+=(--adapter_path "$ADAPTER_PATH")
case "$CONDITION" in
    video) ;;
    novideo) EXTRA+=(--no-video) ;;
    single) EXTRA+=(--single-video) ;;
    *) echo "unknown CONDITION=$CONDITION"; exit 1 ;;
esac

TAG="$(echo "${ADAPTER_PATH:-$MODEL}" | sed 's#^\./##; s#/#__#g')"
mkdir -p eval_results
RESULTS="eval_results/${TAG}.tsv"

for conf in $CONFIGS; do
    unset filepath fps batch_size max_new_tokens
    # shellcheck disable=SC1090
    source "$conf"
    if [ ! -f "$filepath" ]; then
        echo "missing $filepath -- run scripts/prepare_data.py first"; exit 1
    fi
    echo "== $(basename "$conf" .conf): $filepath (fps $fps)"
    out="$(mktemp)"
    python -m eval.evaluate \
        --model_version "$MODEL" \
        --filepath "$filepath" \
        --fps "$fps" \
        --batch_size "${BATCH_SIZE:-$batch_size}" \
        --max-new-tokens "${max_new_tokens:-2048}" \
        --sample-num "$N" \
        "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee "$out"
    acc="$(grep 'Final Accuracy for' "$out" | tail -1)"
    rm -f "$out"
    printf '%s\t%s\t%s\n' "$CONDITION" "$(basename "$conf" .conf)" "${acc:-FAILED}" >> "$RESULTS"
done
echo "Summary: $RESULTS"
