#!/usr/bin/env bash
# Run every config for ONE model (serial, gentle on Groq's per-model tokens-per-minute cap).
# Usage: evals/run_model.sh <groq-model-id> [epochs]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="$1"; EPOCHS="${2:-3}"
# MAX_SAMPLES=1 runs at Groq's 250k tokens/min per-model cap with almost no 429 retries (slow, clean).
# MAX_SAMPLES=3 is ~2-3x faster but pays for it with retries (see evals/README.md, "Rate limits").
MAX_SAMPLES="${MAX_SAMPLES:-1}"
PROVIDER="${PROVIDER:-groq}"   # groq | anthropic | openai (key in .env); e.g. PROVIDER=anthropic evals/run_model.sh claude-haiku-4-5
for config in baseline datadesc format format_query; do
  uv run inspect eval evals/qc_eval.py -T model="$MODEL" -T provider="$PROVIDER" -T config="$config" -T dataset=titanic \
    --epochs "$EPOCHS" --log-dir evals/logs --max-samples "$MAX_SAMPLES" --display plain
done
for config in baseline format format_query; do
  uv run inspect eval evals/qc_eval.py -T model="$MODEL" -T provider="$PROVIDER" -T config="$config" -T dataset=crashes \
    --epochs "$EPOCHS" --log-dir evals/logs --max-samples "$MAX_SAMPLES" --display plain
done
