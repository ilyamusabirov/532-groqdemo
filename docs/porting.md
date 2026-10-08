---
title: Porting 532 demos to Groq
---

# Porting a 532 LLM demo to Groq

Mirrors the guide in the repo's `CLAUDE.md`. Everything in the lecture code goes through
[chatlas](https://posit-dev.github.io/chatlas/) (Python) or ellmer (R), so porting is a client swap
plus an API key. Groq's OpenAI-compatible endpoint means tool calling, streaming and structured output
work as they do elsewhere.

## Which model

| use | model id | why |
|---|---|---|
| default for querychat and tool demos | `qwen/qwen3.8-27b` | best tool choice and SQL in our evals, zero Groq schema errors |
| second opinion / "does another model agree" | `qwen/qwen3.6-27b` | close second; some schema errors |
| cheap and fast plain chat | `openai/gpt-oss-20b` | fine for chat; 40 schema errors with querychat |
| avoid with querychat for now | `openai/gpt-oss-120b` | Groq rejects ~1 in 3 of its `querychat_query` calls (see quirks) |

Model ids drift. List what the key can see with:

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv('.env'); import os, openai
c = openai.OpenAI(base_url='https://api.groq.com/openai/v1', api_key=os.environ['GROQ_API_KEY'])
print(*sorted(m.id for m in c.models.list().data), sep='\n')"
```

Not served under this key as of 2026-10-07: embedding models and vision models. So the RAG demo keeps
its local/other embedder and only the generation step moves to Groq, and the lecture 6 image/PDF
structured-output demos cannot be ported as-is.

## The one-line change, per lecture pattern

```python
# .env
GROQ_API_KEY="gsk_..."

# lecture 5/6/7 apps and scripts: ChatGithub / ChatAnthropic / ChatOpenAI ->
from chatlas import ChatGroq
client = ChatGroq(model="qwen/qwen3.8-27b")

# 532-dbdemo apps that use ChatAuto(): no code change, set in .env
CHATLAS_CHAT_PROVIDER_MODEL="groq/qwen/qwen3.8-27b"

# querychat
qc = querychat.QueryChat(df, "name", client=ChatGroq(model="qwen/qwen3.8-27b"), ...)

# R / ellmer
chat <- ellmer::chat_groq(model = "qwen/qwen3.8-27b")
```

Everything else (`system_prompt`, `register_tool`, `.chat()` / `.stream()`, `ui.Chat`, `extra_instructions`,
`data_description`) stays the same.

## Running things

```bash
uv sync                      # all groups, incl. evals
uv run shiny run app.py      # a Shiny app
uv run python script.py      # a script
uv run pytest                # tests
```

No `pip install`, no `requirements.txt`: dependencies live in `pyproject.toml`, add with `uv add <pkg>`.

## Rate limits and cost

- The key's cap is **250k tokens per minute per model**. querychat's system prompt is ~4.5k tokens, so a
  classroom of students hammering one model will hit 429s. Spread demos across the two Qwen models, or
  keep `--max-samples 3` in evals and let the solver's retries absorb it.
- Different models have separate caps, so parallel runs across models are free; serial within a model.

## Known quirks (found by the evals)

1. **Groq validates tool arguments strictly.** querychat declares `collapsed` on `querychat_query` as
   required while its prompt tells the model it may omit it. gpt-oss-120b omits it often, qwen3.6 and
   gpt-oss-20b sometimes, qwen3.8 never; Groq rejects the call with
   `Tool call validation failed ... missing properties: 'collapsed'`. Catch exceptions around `.chat()` in
   demos, or prefer qwen3.8.
2. **Case-sensitive strings.** Models sometimes write `class = 'FIRST'`; DuckDB returns zero rows silently.
   `data_description` listing the actual values reduces this.
3. **Qwen reasoning text.** Qwen models emit a reasoning block; chatlas keeps it as separate content. To show
   only the answer, take `ContentText` parts of the last turn (see `evals/solver.py`).
4. **"women" is ambiguous** on Titanic: `sex='female'` (233 survivors) vs `who='woman'` (205). Say "female"
   in demos, or treat both as right in checks.
