# 532-groqdemo

**Site:** <https://ilyamusabirov.github.io/532-groqdemo/> — instructions, [porting guide](https://ilyamusabirov.github.io/532-groqdemo/porting.html), [eval report](https://ilyamusabirov.github.io/532-groqdemo/eval-report.html), [cost estimates](https://ilyamusabirov.github.io/532-groqdemo/cost.html).

Minimal [querychat](https://posit-dev.github.io/querychat/) demo running on
[Groq](https://console.groq.com/) with an open-weight Qwen model. DSCI 532.

Ask the sidebar things like *"Show only women who survived"* or
*"filter to first class passengers"*. querychat turns the question into SQL,
runs it against the Titanic dataset, and updates the table.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # creates .venv from pyproject.toml + uv.lock (evals included)
cp .env.example .env     # paste your Groq key into GROQ_API_KEY
uv run shiny run app.py
```

Then open http://127.0.0.1:8000/.

## Switching models

The only provider-specific line is in `app.py`:

```python
client=ChatGroq(model="qwen/qwen3.8-27b"),
```

Any other chatlas provider works the same way, e.g. `ChatGithub(model="gpt-4.1-mini")`
with a `GITHUB_TOKEN`, or `ChatAnthropic()` with an `ANTHROPIC_API_KEY`.

## Evals: what we found

**Results** (4 Groq models × 4 prompt configs × 58 Titanic phrasings × 3 epochs, 3,180 runs; [full report](https://ilyamusabirov.github.io/532-groqdemo/eval-report.html)).
qwen3.8-27b is the most reliable querychat driver (tool choice 0.84, correct filter SQL 0.91, 2% API errors);
`data_description` lifts vocabulary-trap filters from ~0.6-0.8 to 0.8-1.0; the query-forcing `extra_instructions`
cut hallucinated stats by two thirds. gpt-oss-120b looks bad mostly because Groq rejects a third of its tool calls
(querychat's `collapsed` argument is declared required but the prompt says to omit it).

**Cost** ([details](https://ilyamusabirov.github.io/532-groqdemo/cost.html)). The matrix cost about $26, $24 of it on
Qwen: a querychat message is ~3 calls × ~4k-token prompt and Groq caches prompts only on gpt-oss. Per message:
qwen3.8 $0.0115, gpt-oss-20b $0.0008. 150 students × 120 messages: $264 on qwen3.8, $17 on gpt-oss-20b.

**Recommendation for a class.** Free-tier keys cannot run querychat: one message (9-13k tokens) exceeds the
8k-tokens/minute cap and the 200k/day cap allows ~16 messages. Hand out developer-tier keys or a shared org key,
and spread the class across models, since the 250k-tokens/minute cap is per model (~19 Qwen messages/min per key).
Default to `qwen/qwen3.8-27b` for demos where correctness matters and `openai/gpt-oss-20b` where cost does,
tolerating its ~7% rejected tool calls; avoid gpt-oss-120b with querychat until querychat's tool schema is fixed.
Wrap `.chat()` in try/except in any student-facing app.

`evals/` holds an [inspect_ai](https://inspect.aisi.org.uk/) suite that runs the real querychat stack
against four Groq models under the prompt customizations from the lecture (`data_description`,
`extra_instructions`) and scores tool choice, SQL correctness, format adherence and whether reported
numbers came from a query or from memory. See [`evals/README.md`](evals/README.md).

```bash
uv run pytest                 # scoring logic, no API calls
evals/run_matrix.sh 3         # full matrix (~1.5 h at the rate cap), then evals/results/summary.md
```

Porting other DSCI 532 demos to Groq, which models to use, and known quirks: see [`CLAUDE.md`](CLAUDE.md).
