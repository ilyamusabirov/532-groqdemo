"""Aggregate inspect logs into evals/results/summary.md (+ summary.csv).

    uv run python evals/report.py [--log-dir evals/logs] [--out evals/results/summary.md]

Structure of the report: narrative summary -> API errors -> what was tested -> detail blocks -> volume.
Every number is computed from the logs; the prose templates only pick which numbers to quote."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

SCORERS = ["tool_choice", "filter_rows", "answer_correct", "format_adherence", "number_honesty"]
CONFIG_ORDER = ["baseline", "datadesc", "format", "format_query"]
SHORT = {"qwen/qwen3.8-27b": "qwen3.8-27b", "qwen/qwen3.6-27b": "qwen3.6-27b",
         "openai/gpt-oss-20b": "gpt-oss-20b", "openai/gpt-oss-120b": "gpt-oss-120b"}


# ---------------------------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------------------------

def _error_kind(err: str | None) -> str | None:
    if not err:
        return None
    if "collapsed" in err:
        return "schema: `collapsed` missing"
    if "validation failed" in err or "Failed to parse tool call" in err or "Failed to call a function" in err:
        return "malformed tool call"
    if "RateLimit" in err:
        return "rate limit (retries exhausted)"
    return err.split(":")[0][:40]


def load_rows(log_dir: Path) -> tuple[pd.DataFrame, dict]:
    rows, meta = [], {"epochs": set(), "max_samples": set(), "started": []}
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name)
        if log.status != "success" or not log.samples:
            continue
        m = log.eval.metadata or {}
        meta["epochs"].add(log.eval.config.epochs)
        meta["max_samples"].add(log.eval.config.max_samples)
        meta["started"].append(log.eval.created)
        for s in log.samples:
            st = s.store or {}
            row = {
                "model": SHORT.get(m.get("groq_model", "?"), m.get("groq_model", "?")),
                "config": m.get("config", "?"), "dataset": m.get("dataset", "?"),
                "intent": s.metadata.get("intent"), "family": s.metadata.get("family"),
                "trap": bool(s.metadata.get("trap")), "epoch": s.epoch, "input": s.input,
                "error": bool(st.get("error")), "error_kind": _error_kind(st.get("error")),
                "api_calls": 1 + len(st.get("tools") or []), "retries": st.get("retries") or 0,
                "honesty": (s.scores["number_honesty"].metadata or {}).get("verdict") if s.scores and "number_honesty" in s.scores else None,
            }
            for name in SCORERS:
                sc = s.scores.get(name) if s.scores else None
                row[name] = None if sc is None or sc.value == "N" else (1.0 if sc.value == "C" else 0.0)
            rows.append(row)
    return pd.DataFrame(rows), meta


# ---------------------------------------------------------------------------------------------
# table helpers
# ---------------------------------------------------------------------------------------------

def pct_table(d: pd.DataFrame, scorer: str, configs: list[str] | None = None) -> pd.DataFrame:
    """model × config table of accuracy (as %), over samples where the scorer applied; n in the header."""
    d = d[d[scorer].notna()]
    t = (d.groupby(["model", "config"])[scorer].mean() * 100).round(0).unstack("config")
    cols = [c for c in (configs or CONFIG_ORDER) if c in t.columns]
    t = t[cols]
    n = d.groupby("config").size()
    t.columns = [f"{c} (n={n[c] // d['model'].nunique()})" for c in cols]
    t.index.name = "model"
    return t.astype("Int64")


def md(t: pd.DataFrame) -> str:
    if isinstance(t.index, pd.MultiIndex):
        t = t.reset_index()
    return t.to_markdown(index=not isinstance(t.index, pd.RangeIndex)) + "\n"


def col(t: pd.DataFrame, prefix: str) -> str:
    """Column of a pct_table whose config name starts with `prefix` (headers carry n)."""
    return next(c for c in t.columns if c.startswith(prefix))


def best(t: pd.DataFrame, col: str) -> tuple[str, int]:
    s = t[col].dropna()
    return s.idxmax(), int(s.max())


# ---------------------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------------------

def section_summary(df: pd.DataFrame) -> str:
    ti = df[df["dataset"] == "titanic"]
    tc = pct_table(ti, "tool_choice"); fr = pct_table(ti, "filter_rows"); ac = pct_table(ti, "answer_correct")
    fmt = pct_table(ti, "format_adherence", ["format", "format_query"])
    traps = pct_table(ti[ti["trap"]], "filter_rows")
    err = (df.groupby("model")["error"].mean() * 100).round(0)
    hon = ti[ti["honesty"].notna()]
    bad = hon.assign(bad=hon["honesty"].isin(["wrong", "unverified"])).groupby(["model", "config"])["bad"].mean().unstack("config") * 100

    composite = pd.concat([tc.mean(axis=1), fr.mean(axis=1), ac.mean(axis=1)], axis=1).mean(axis=1)
    best_model = composite.idxmax(); best_tc = int(tc.loc[best_model, col(tc, "format_query")])
    lines = [
        "## Summary\n",
        f"- **Most reliable querychat driver: {best_model}** (highest mean of tool choice, filter SQL and answer accuracy "
        f"across configs: " + ", ".join(f"{m} {v:.0f}" for m, v in composite.sort_values(ascending=False).items()) + "). "
        f"Under the lecture's query-forcing template it picked the "
        f"right tool {best_tc}% of the time, wrote filter SQL that selected the right rows {fr.loc[best_model, col(fr, 'format_query')]}% of the time, "
        f"and had a {err[best_model]:.0f}% API-error rate.",
        f"- **`data_description` works.** On the vocabulary traps (children, steerage, travelling alone) adding it moved "
        f"correct filters from {int(traps[col(traps, 'baseline')].mean())}% to "
        f"{int(traps[col(traps, 'datadesc')].mean())}% averaged over models"
        + "".join(f"; {m}: {traps.loc[m, col(traps, 'baseline')]}→{traps.loc[m, col(traps, 'datadesc')]}" for m in traps.index) + ".",
        f"- **Telling the model to query changes where stats come from.** Share of filter replies with wrong or unverified "
        f"numbers, format-only → query-forcing template: "
        + "; ".join(f"{m}: {bad.loc[m, 'format']:.0f}%→{bad.loc[m, 'format_query']:.0f}%" for m in bad.index)
        + ". Replies that give no numbers at all are counted separately as `silent` (section 6).",
        f"- **Questions are mostly answered right** ({int(ac[col(ac, 'baseline')].mean())}% baseline average), the misses are "
        f"models answering from memory without a query.",
        f"- **Format templates are followed {int(fmt.iloc[:, -1].mean())}% of the time on average**, "
        f"from {int(fmt.iloc[:, -1].min())}% to {int(fmt.iloc[:, -1].max())}% by model.",
        f"- **API errors are a provider issue, not model behaviour.** Error rate by model: "
        + ", ".join(f"{m} {err[m]:.0f}%" for m in err.index) + ". Almost all are Groq rejecting a tool call "
        "because querychat declares `collapsed` as required while its prompt says it may be omitted. "
        "An errored request scores as a failure everywhere, so high-error models are penalised beyond their behaviour.",
    ]
    return "\n".join(lines) + "\n\n"


def section_errors(df: pd.DataFrame) -> str:
    vol = df.groupby("model").agg(requests=("error", "size"), errors=("error", "sum"))
    vol["error rate"] = (vol["errors"] / vol["requests"] * 100).round(1).astype(str) + "%"
    kinds = df[df["error"]].groupby(["model", "error_kind"]).size().unstack(fill_value=0)
    t = vol.join(kinds).fillna(0)
    for c in t.columns:
        if c != "error rate":
            t[c] = t[c].astype(int)
    t.index.name = "model"
    return ("## API errors\n\n*Requests that raised an exception instead of completing. "
            "`schema: collapsed missing` = Groq's strict tool-argument validation rejecting a `querychat_query` call "
            "(querychat bug, see README). `malformed tool call` = the model emitted unparsable arguments. "
            "`rate limit` = still 429 after 6 retries.*\n\n" + md(t))


def section_what(df: pd.DataFrame) -> str:
    ti = df[df["dataset"] == "titanic"]
    fam = ti[ti["epoch"] == 1].groupby("family").agg(intents=("intent", "nunique"), phrasings=("input", "nunique"))
    fam["expected tool"] = {"filter": "`querychat_update_dashboard`", "question": "`querychat_query`",
                            "reset": "`querychat_reset_dashboard`", "out_of_scope": "none"}
    fam["example"] = {"filter": "Show only first class passengers / Shwo me frist class pasengers / kids only",
                      "question": "How many women survived? / avg fare, pclass 1",
                      "reset": "start over", "out_of_scope": "What's the weather in Vancouver today?"}
    fam = fam.loc[["filter", "question", "reset", "out_of_scope"]]
    return f"""## What was tested

**Models** (all served by Groq): qwen3.8-27b, qwen3.6-27b, gpt-oss-20b, gpt-oss-120b. Every request runs the real
querychat stack (its system prompt and tools, via chatlas); nothing is re-implemented.

**Prompt configs**, copied verbatim from the DSCI 532 lecture 5 slides:

| config | what the model sees beyond querychat's default prompt |
|---|---|
| `baseline` | nothing extra |
| `datadesc` | `data_description`: column meanings (survived 1/0, pclass 1=First…, who child = under 16, alone…) |
| `format` | `datadesc` + the Monday `extra_instructions`: a 4-line reply template with 🎯 Filter / 📊 Stats / 💡 Insight / 🔍 Try next |
| `format_query` | `datadesc` + the Wednesday template that adds "use querychat_query to calculate row count and survival rate" |

**Query families** on Titanic; each intent has 3–5 phrasings (canonical, terse, typo, indirect, formal):

{md(fam)}
Trap intents (children, steerage, travelling alone) use vocabulary that only `data_description` defines.
"Women" is ambiguous (`sex='female'` → 233, `who='woman'` → 205); both are accepted.

**Crashes dataset**: a fake 100-row planet-crash table the models cannot know from training data. Same families
(11 phrasings). Any number stated without a query there is invented.

**Scorers** (deterministic; LLM judges are not used):

| scorer | applies to | counts as correct when |
|---|---|---|
| tool_choice | every request | the set of action tools called matches the config's expectation (schema lookups ignored); an API error is a failure |
| filter_rows | filter intents | the `update_dashboard` SQL, run in DuckDB, selects exactly an acceptable row count |
| answer_correct | question intents | an acceptable truth value appears in the reply (±0.06 or rounded) |
| format_adherence | filter replies under `format*` | the 🎯 📊 💡 lines are present (+ a clickable suggestion span) |
| number_honesty | filter replies under `format*` | stated rows and survival rate are right **and** came from a `querychat_query` call |

`number_honesty` also labels each reply: **honest** (queried, right), **wrong** (numbers don't match the data),
**unverified** (right numbers, no query: recalled from training data), **silent** (no numbers given).

Every run is repeated for 3 epochs because tool choice is stochastic. Tables report the percentage of requests
scored correct among those where the scorer applied; `n` is requests per model in that column.
"""


def section_details(df: pd.DataFrame) -> str:
    ti = df[df["dataset"] == "titanic"]
    out = ["## Results in detail\n"]
    out.append("### 1. Does the model call the right querychat tool?\n\n*All Titanic requests. Under `format_query` a filter "
               "must call both `update_dashboard` and `query`; under `format` the extra query is optional.*\n\n" + md(pct_table(ti, "tool_choice")))
    out.append("### 2. Is the filter SQL right?\n\n*Filter intents only. The model's SQL is executed in DuckDB and must return "
               "exactly an acceptable row count, so `class = 'FIRST'` (0 rows, case-sensitive) fails.*\n\n" + md(pct_table(ti, "filter_rows")))
    out.append("### 3. Do vocabulary traps need `data_description`?\n\n*Filter intents whose words the schema alone does not "
               "define: children (child = under 16), steerage (third class), travelling alone. Compare `baseline` with `datadesc`.*\n\n"
               + md(pct_table(ti[ti["trap"]], "filter_rows")))
    out.append("### 4. Are questions answered correctly?\n\n*Question intents only (counts, averages, max fare, a survival rate). "
               "The reply must contain the right number.*\n\n" + md(pct_table(ti, "answer_correct")))
    out.append("### 5. Is the reply template followed?\n\n*Filter replies under the two templates. Checks the three emoji-labelled "
               "lines and the suggestion span.*\n\n" + md(pct_table(ti, "format_adherence", ["format", "format_query"])))
    hon = ti[ti["honesty"].notna()]
    ct = hon.groupby(["model", "config"])["honesty"].value_counts().unstack(fill_value=0)
    for c in ["honest", "unverified", "wrong", "silent"]:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[["honest", "unverified", "wrong", "silent"]]
    ct["honest share"] = (ct["honest"] / ct.sum(axis=1) * 100).round(0).astype(int).astype(str) + "%"
    out.append("### 6. Where do the stats come from?\n\n*Filter replies under the two templates, counted per verdict. "
               "The lecture's claim is that `format` (stats demanded, no query instructed) invites numbers from memory, and "
               "`format_query` fixes it. `unverified` is the silent-hallucination case: correct numbers, no query.*\n\n" + md(ct))
    cr = df[df["dataset"] == "crashes"]
    if len(cr):
        crt = cr.groupby(["model", "config"])[["tool_choice", "filter_rows", "answer_correct", "number_honesty"]].mean() * 100
        crt = crt.round(0).astype("Int64").reindex(columns=["tool_choice", "filter_rows", "answer_correct", "number_honesty"])
        crt.columns = ["tool choice %", "filter SQL %", "answers %", "honest stats %"]
        crt = crt.reset_index().sort_values(["model", "config"], key=lambda c: c.map(lambda v: CONFIG_ORDER.index(v) if v in CONFIG_ORDER else v)).reset_index(drop=True)
        hc = cr[cr["honesty"].notna()].groupby(["model", "config"])["honesty"].value_counts().unstack(fill_value=0)
        out.append("### 7. Fake data: does anything get invented?\n\n*The crashes table is unknown to every model, so a stated "
                   "number that was not queried is fabricated. Percent correct per scorer; blank = not applicable.*\n\n" + md(crt)
                   + "\nVerdicts per reply:\n\n" + md(hc))
    return "\n".join(out)


def section_volume(df: pd.DataFrame, meta: dict) -> str:
    vol = df.groupby("model").agg(requests=("error", "size"), api_calls=("api_calls", "sum"),
                                  retry_attempts=("retries", "sum"))
    vol.loc["**total**"] = vol.sum()
    vol = vol.astype(int); vol.index.name = "model"
    ms = ", ".join(str(x) for x in sorted(meta["max_samples"], key=str))
    return ("## Volume and pacing\n\n*One request = one user message to querychat (a sample × epoch). API calls ≈ final answer + one per "
            "tool request. Retry attempts = extra fresh conversations after a 429.*\n\n" + md(vol) +
            f"\nRun with `max_samples={ms}` concurrent requests per model. Groq's developer tier caps each model at 250k tokens/min; "
            "a Qwen querychat request is ~12-13k tokens (no prompt caching), so one key sustains ~19 Qwen requests/min. "
            "`MAX_SAMPLES=1` paces at that rate with few retries; higher values finish sooner and recover via retries. "
            "gpt-oss prompts are cached and appear to count little toward the cap.\n")


# ---------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=str(Path(__file__).parent / "logs"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "summary.md"))
    args = ap.parse_args()
    df, meta = load_rows(Path(args.log_dir))
    if df.empty:
        print("no successful logs found in", args.log_dir)
        return
    when = min(meta["started"])[:10] if meta["started"] else datetime.now().date().isoformat()
    head = (f"# querychat × Groq: eval report\n\n*Run {when}: {df['model'].nunique()} models, "
            f"{df['config'].nunique()} prompt configs, {df[df['epoch']==1]['input'].nunique()} phrasings, "
            f"{', '.join(str(e) for e in sorted(meta['epochs']))} epochs, {len(df):,} requests. "
            f"Generated by `evals/report.py`; method in `evals/README.md`.*\n\n")
    body = head + section_summary(df) + section_errors(df) + "\n" + section_what(df) + "\n" + section_details(df) + "\n" + section_volume(df, meta)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body); df.to_csv(out.with_suffix(".csv"), index=False)
    print(body)
    print(f"\nwrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
