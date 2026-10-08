#!/usr/bin/env bash
# Full model × config matrix. Models run in parallel (Groq rate limits are per model),
# configs run serially within a model. Then builds evals/results/summary.md.
# Usage: evals/run_matrix.sh [epochs]   (default 3; ~45 min wall time at 3 epochs)
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCHS="${1:-3}"
mkdir -p evals/logs evals/results
MODELS=(qwen/qwen3.6-27b qwen/qwen3.8-27b openai/gpt-oss-20b openai/gpt-oss-120b)
pids=()
for model in "${MODELS[@]}"; do
  evals/run_model.sh "$model" "$EPOCHS" > "evals/results/run_${model//\//_}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
uv run python evals/report.py
{ printf -- '---\ntitle: Eval report\n---\n\n'; cat evals/results/summary.md; } > docs/eval-report.md
exit $status
