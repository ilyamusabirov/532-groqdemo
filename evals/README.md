# querychat × Groq evals

An [inspect_ai](https://inspect.aisi.org.uk/) suite that drives the **real querychat stack**
(its system prompt and tools, via chatlas) against Groq-served open models and scores what the
model did, deterministically. It answers the lecture's questions with data instead of anecdotes:

- Does the model pick the right querychat tool for a filter, a question, a reset, an off-topic request?
- Is the SQL right? (Run against DuckDB and compared to a precomputed truth.)
- Does `data_description` fix vocabulary traps ("children", "steerage", "travelling alone")?
- Do the `extra_instructions` templates from the slides get followed?
- When a template demands stats, do the numbers come from a `querychat_query` call, from memory, or from nowhere?

## Run

```bash
uv sync --all-groups                       # installs inspect-ai, duckdb, pytest
uv run pytest                              # unit tests for the scoring logic (no API calls)
uv run python evals/build_dataset.py       # regenerate evals/data/*.jsonl after editing INTENTS

# one cell of the matrix
uv run inspect eval evals/qc_eval.py -T model=qwen/qwen3.8-27b -T config=format_query -T dataset=titanic \
  --epochs 3 --log-dir evals/logs --max-samples 3 --display plain
uv run inspect view                        # browse per-sample logs

# everything (4 models in parallel, ~45 min), then the summary table
evals/run_matrix.sh 3
uv run python evals/report.py              # -> evals/results/summary.md + summary.csv
```

Keep `--max-samples` at 3 or so: every request carries the ~4.5k-token querychat system prompt and Groq's
tokens-per-minute cap is per model. The solver retries 429s with a fresh conversation.

## Matrix

| Dimension | Values |
|---|---|
| models | `qwen/qwen3.6-27b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b` |
| configs (titanic) | `baseline`, `datadesc`, `format`, `format_query` |
| configs (crashes) | `baseline`, `format`, `format_query` |
| datasets | `titanic` (58 phrasings over 17 intents), `crashes` (fake data the model cannot know; 11 phrasings) |

Configs are the slide examples, verbatim, in `prompts.py`. `format` is the Monday template (stats demanded,
no instruction to query); `format_query` is the Wednesday template that says "use querychat_query".

## Scorers

| scorer | applies to | CORRECT when |
|---|---|---|
| `tool_choice` | all | the set of action tools called matches what the config allows (schema lookup ignored; API error = fail) |
| `filter_rows` | filter intents | the `update_dashboard` SQL selects exactly an acceptable row count |
| `answer_correct` | question intents | an acceptable truth value appears in the reply (±0.06 or rounded) |
| `format_adherence` | filter replies under `format*` | 🎯 📊 💡 lines present (+ suggestion span on titanic) |
| `number_honesty` | filter replies under `format*` | stats are right **and** came from a `querychat_query` call |

`number_honesty` also records a verdict per reply: `honest`, `wrong`, `unverified` (right numbers, no query:
recalled from training data), `silent`. The report tabulates those.

Ambiguity is encoded in the data, not papered over: "women who survived" accepts both `sex='female'` (233)
and `who='woman'` (205); "children" accepts `who='child'` or `age < 16` (both 83).

## Files

- `prompts.py` — slide prompts and the config matrix
- `datasets.py` — the two dataframes
- `build_dataset.py` — intents × phrasings → `data/*.jsonl`, targets computed with DuckDB
- `solver.py` — runs one querychat conversation per sample, records tools / SQL / text / errors
- `checks.py` — pure scoring functions (unit-tested in `tests/`)
- `scorers.py` — inspect wrappers around `checks.py`
- `qc_eval.py` — the `@task`
- `report.py` — logs → model × config × scorer table
- `run_model.sh`, `run_matrix.sh` — runners

## Known provider quirks (findings, not bugs in this suite)

- Groq validates tool-call arguments against the schema strictly. querychat marks `collapsed` as required
  but its prompt says the model may omit it; when gpt-oss does omit it, Groq rejects the call
  (`Tool call validation failed ... missing properties: 'collapsed'`). Recorded as an API error.
- Qwen models sometimes write `class = 'FIRST'`; DuckDB string comparison is case-sensitive, so the filter
  returns zero rows. `filter_rows` catches it.
