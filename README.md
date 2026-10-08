# 532-groqdemo

Minimal [querychat](https://posit-dev.github.io/querychat/) demo running on
[Groq](https://console.groq.com/) with an open-weight Qwen model. DSCI 532.

Ask the sidebar things like *"Show only women who survived"* or
*"filter to first class passengers"*. querychat turns the question into SQL,
runs it against the Titanic dataset, and updates the table.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # creates .venv from pyproject.toml + uv.lock
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

## Evals

`evals/` holds an [inspect_ai](https://inspect.aisi.org.uk/) suite that runs the real querychat stack
against four Groq models under the prompt customizations from the lecture (`data_description`,
`extra_instructions`) and scores tool choice, SQL correctness, format adherence and whether reported
numbers came from a query or from memory. See [`evals/README.md`](evals/README.md).

```bash
uv sync --all-groups
uv run pytest                 # scoring logic, no API calls
evals/run_matrix.sh 3         # full matrix (~45 min), then evals/results/summary.md
```
