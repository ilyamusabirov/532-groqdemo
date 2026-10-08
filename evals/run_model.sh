#!/usr/bin/env bash
# Run every config for ONE model (serial, gentle on Groq's per-model tokens-per-minute cap).
# Usage: evals/run_model.sh <groq-model-id> [epochs]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; EPOCHS="${2:-3}"
for config in baseline datadesc format format_query; do
  uv run inspect eval evals/qc_eval.py -T model="$MODEL" -T config="$config" -T dataset=titanic \
    --epochs "$EPOCHS" --log-dir evals/logs --max-samples 3 --display plain
done
for config in baseline format format_query; do
  uv run inspect eval evals/qc_eval.py -T model="$MODEL" -T config="$config" -T dataset=crashes \
    --epochs "$EPOCHS" --log-dir evals/logs --max-samples 3 --display plain
done
