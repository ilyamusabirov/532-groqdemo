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
                "honesty": (s.scores.get("number_honesty").metadata or {}).get("verdict") if s.scores.get("number_honesty") else None,
            }
            for name in SCORERS:
                sc = s.scores.get(name) if s.scores else None
                row[name] = None if sc is None or sc.value == "N" else (1.0 if sc.value == "C" else 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> str:
    parts = []
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
