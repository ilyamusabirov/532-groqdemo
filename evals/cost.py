"""Groq spend: what the eval matrix cost, and what a cohort of students would cost.

    uv run python evals/cost.py            # uses evals/logs for actual sample/retry counts
    uv run python evals/cost.py --no-logs  # scenarios only

Prices: Groq list prices per 1M tokens as of 2026-09 (third-party summary, verify in the Groq console).
Tokens per querychat request: measured 2026-10-07 on titanic with the format_query config (4 requests/model,
chatlas turn.tokens = Groq usage). Students' own datasets with wider schemas will cost more per request.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PRICES = {  # $ per 1M tokens: input, output, cached-input (50% off on gpt-oss only)
    "qwen/qwen3.8-27b": (0.80, 4.00, None),
    "qwen/qwen3.6-27b": (0.60, 3.00, None),
    "openai/gpt-oss-20b": (0.075, 0.30, 0.0375),
    "openai/gpt-oss-120b": (0.15, 0.60, 0.075),
}
# measured per querychat request: api_calls, input (uncached), output, cached input
PER_REQUEST = {
    "qwen/qwen3.8-27b": dict(calls=3.0, inp=13011, out=283, cached=0),
    "qwen/qwen3.6-27b": dict(calls=2.8, inp=11922, out=334, cached=0),
    "openai/gpt-oss-20b": dict(calls=2.2, inp=1440, out=1422, cached=7232),
    "openai/gpt-oss-120b": dict(calls=2.2, inp=5044, out=466, cached=3584),
}
FREE_TIER = dict(rpm=30, rpd=1000, tpm=8_000, tpd=200_000)
DEV_TIER = dict(rpm=1000, tpm=250_000)


def request_cost(model: str, history_factor: float = 1.0) -> float:
    pin, pout, pcache = PRICES[model]
    t = PER_REQUEST[model]
    usd = t["inp"] * history_factor * pin + t["out"] * pout
    if t["cached"] and pcache:
        usd += t["cached"] * history_factor * pcache
    return usd / 1e6


def matrix_cost(log_dir: Path) -> list[tuple[str, int, int, float]]:
    from inspect_ai.log import list_eval_logs, read_eval_log  # lazy: evals group only

    samples, retries = {}, {}
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info.name, header_only=False)
        m = log.eval.metadata["groq_model"]
        for s in log.samples:
            samples[m] = samples.get(m, 0) + 1
            retries[m] = retries.get(m, 0) + ((s.store or {}).get("retries") or 0)
    rows = []
    for m in samples:
        # a retried attempt re-sent roughly half a request before the 429 landed
        usd = (samples[m] + 0.5 * retries[m]) * request_cost(m)
        rows.append((m, samples[m], retries[m], usd))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-logs", action="store_true")
    ap.add_argument("--log-dir", default=str(Path(__file__).parent / "logs"))
    ap.add_argument("--students", type=int, default=150)
    args = ap.parse_args()

    print("## Cost per querychat request (one user message; ~3 API calls: schema, tool, answer)\n")
    print("| model | API calls | input tok | output tok | cached tok | $/request | requests per $1 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for m, t in PER_REQUEST.items():
        c = request_cost(m)
        print(f"| {m} | {t['calls']:.1f} | {t['inp']:,} | {t['out']:,} | {t['cached']:,} | ${c:.4f} | {1/c:,.0f} |")

    if not args.no_logs:
        rows = matrix_cost(Path(args.log_dir))
        if rows:
            print("\n## What the eval matrix in evals/logs cost (3 epochs, 4 models)\n")
            print("| model | sample-epochs | retry attempts | est. $ |")
            print("|---|---:|---:|---:|")
            tot = 0.0
            for m, n, r, usd in sorted(rows):
                tot += usd
                print(f"| {m} | {n} | {r} | ${usd:.2f} |")
            print(f"| **total** | | | **${tot:.2f}** |")
            print("\nPlus roughly 15% for the spikes and validation runs that preceded the matrix (qwen3.8 mostly).")

    N = args.students
    print(f"\n## Scenarios: {N} students building a querychat dashboard\n")
    print("Requests = user messages to querychat across the whole milestone (building, testing, demoing, TA grading).")
    print("Follow-up messages carry conversation history, so input grows: 1.3× applied.\n")
    scen = [("light: try it, wire it up, demo", 40), ("medium: iterate on prompts + data_description", 120),
            ("heavy: tune extra_instructions, many test runs", 300)]
    print("| scenario | requests/student | " + " | ".join(m.split('/')[-1] for m in PRICES) + " |")
    print("|---|---:|" + "---:|" * len(PRICES))
    for name, req in scen:
        cells = [f"${N * req * request_cost(m, 1.3):,.0f}" for m in PRICES]
        print(f"| {name} | {req} | " + " | ".join(cells) + " |")

    print("\n## Will it even run? Rate limits per key\n")
    print(f"Free tier: {FREE_TIER['tpm']:,} tokens/min, {FREE_TIER['tpd']:,} tokens/day, {FREE_TIER['rpd']} requests/day.")
    print(f"Developer tier: {DEV_TIER['tpm']:,} tokens/min per model, {DEV_TIER['rpm']} requests/min.\n")
    print("| model | tokens per request | free-tier requests/day | free-tier TPM ok? | dev-tier requests/min (one key) |")
    print("|---|---:|---:|---|---:|")
    for m, t in PER_REQUEST.items():
        tok = t["inp"] + t["out"] + t["cached"]
        per_day = FREE_TIER["tpd"] // tok
        tpm_ok = "no: one request bursts past 8k" if tok > FREE_TIER["tpm"] else "yes"
        dev_rpm = DEV_TIER["tpm"] // tok
        print(f"| {m} | {tok:,} | {per_day} | {tpm_ok} | ~{dev_rpm} |")
    print(f"\nA class of {N} on one developer key, all active in the same minute, needs ~{N}× one request's tokens "
          f"per minute per model; split the class across models or across a few keys.")


if __name__ == "__main__":
    main()
