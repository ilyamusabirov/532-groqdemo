"""Aggregate inspect logs into one table: model × config × scorer accuracy over applicable samples.

    uv run python evals/report.py [--log-dir evals/logs] [--out evals/results/summary.md]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

SCORERS = ["tool_choice", "filter_rows", "answer_correct", "format_adherence", "number_honesty"]


def _error_kind(err: str | None) -> str | None:
    if not err:
        return None
    if "collapsed" in err:
        return "groq_schema_collapsed"   # querychat marks `collapsed` required; model omitted it; Groq rejects
    if "validation failed" in err or "Failed to parse tool call" in err or "Failed to call a function" in err:
        return "tool_call_malformed"
    if "RateLimit" in err:
        return "rate_limit_exhausted"    # after the solver's retries
    return err.split(":")[0][:40]


def load_rows(log_dir: Path) -> pd.DataFrame:
    rows = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name)
        if log.status != "success" or not log.samples:
            continue
        meta = log.eval.metadata or {}
        for s in log.samples:
            row = {
                "model": meta.get("groq_model", "?"),
                "config": meta.get("config", "?"),
                "dataset": meta.get("dataset", "?"),
                "intent": s.metadata.get("intent"),
                "family": s.metadata.get("family"),
                "trap": s.metadata.get("trap"),
                "epoch": s.epoch,
                "error": bool((s.store or {}).get("error")),
                "error_kind": _error_kind((s.store or {}).get("error")),
                "api_calls": 1 + len((s.store or {}).get("tools") or []),  # final answer + one per tool request
                "retries": (s.store or {}).get("retries") or 0,
                "honesty": (s.scores.get("number_honesty").metadata or {}).get("verdict") if s.scores.get("number_honesty") else None,
            }
            for name in SCORERS:
                sc = s.scores.get(name) if s.scores else None
                row[name] = None if sc is None or sc.value == "N" else (1.0 if sc.value == "C" else 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> str:
    parts = []
    vol = df.groupby("model").agg(requests=("error", "size"), api_calls=("api_calls", "sum"),
                                  retry_attempts=("retries", "sum"), errors=("error", "sum"))
    vol["error_rate"] = (vol["errors"] / vol["requests"]).round(3)
    total = vol.sum(numeric_only=True); total["error_rate"] = round(total["errors"] / total["requests"], 3)
    vol.loc["**total**"] = total
    for c in ["requests", "api_calls", "retry_attempts", "errors"]:
        vol[c] = vol[c].astype(int)
    parts.append("## Volume\n\nOne request = one user message to querychat (a sample × epoch). API calls are "
                 "approximate: the final answer plus one per tool request. Retry attempts are extra full "
                 "conversations started after a 429.\n\n" + vol.to_markdown() + "\n")
    errs = df[df["error"]]
    if len(errs):
        et = errs.groupby(["model", "error_kind"]).size().unstack(fill_value=0)
        parts.append("## API errors (all datasets and configs)\n\nAn errored sample scores INCORRECT on "
                     "`tool_choice` and on its family scorer, so models with many errors are penalised for "
                     "provider-side failures, not only for wrong behaviour. `groq_schema_collapsed` is a "
                     "querychat × Groq interop bug: querychat declares the `collapsed` argument of "
                     "`querychat_query` as required while its prompt tells the model it may omit it; Groq "
                     "validates tool arguments strictly and rejects the call.\n\n" + et.to_markdown() + "\n")
    for dataset, d in df.groupby("dataset"):
        acc = d.groupby(["model", "config"])[SCORERS].mean().round(2)
        n = d.groupby(["model", "config"]).size().rename("n")
        err = d.groupby(["model", "config"])["error"].sum().rename("api_errors")
        table = pd.concat([n, err, acc], axis=1)
        parts.append(f"## {dataset}\n\nAccuracy over applicable samples (blank = scorer not applicable).\n\n"
                     + table.to_markdown() + "\n")
        hon = d[d["honesty"].notna()]
        if len(hon):
            ct = hon.groupby(["model", "config"])["honesty"].value_counts().unstack(fill_value=0)
            parts.append("### Where did the stats come from? (filter replies under format configs)\n\n"
                         + ct.to_markdown() + "\n")
        traps = d[d["trap"] == True]  # noqa: E712
        if len(traps):
            t = traps.groupby(["model", "config"])["filter_rows"].mean().round(2).unstack("config")
            parts.append("### Vocabulary traps (children / steerage / alone): filter_rows accuracy\n\n"
                         + t.to_markdown() + "\n")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default=str(Path(__file__).parent / "logs"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "summary.md"))
    args = ap.parse_args()
    df = load_rows(Path(args.log_dir))
    if df.empty:
        print("no successful logs found in", args.log_dir)
        return
    md = summarize(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# querychat × Groq eval summary\n\n" + md)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(md)
    print(f"\nwrote {out} and {out.with_suffix('.csv')}  ({len(df)} sample-epochs)")


if __name__ == "__main__":
    main()
