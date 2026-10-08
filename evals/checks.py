"""Pure scoring logic, kept free of inspect_ai so it can be unit-tested directly.
Scorers in scorers.py are thin wrappers around these functions."""

from __future__ import annotations

import re

import duckdb
import pandas as pd

SCHEMA_TOOL = "querychat_get_schema"

# ---- tool choice ---------------------------------------------------------------------------

UPDATE, QUERY = "querychat_update_dashboard", "querychat_query"


def allowed_tool_sets(expected: list[str], family: str, config: str) -> list[set[str]]:
    """What counts as the right tool choice depends on the prompt config:
    - format_query: the extra_instructions demand a querychat_query for the stats -> filters must call both
    - format: stats are demanded but no query instructed -> calling query is allowed, not required
    - otherwise: exactly the expected tools."""
    base = set(expected)
    if family == "filter" and config == "format_query":
        return [base | {QUERY}]
    if family == "filter" and config == "format":
        return [base, base | {QUERY}]
    return [base]


def tool_choice_ok(seen: list[str], expected: list[str], family: str = "", config: str = "") -> bool:
    """Compare the set of tools the model called (ignoring the schema lookup) with the allowed sets.
    An empty expected list means: no action tool should be called."""
    seen_set = {t for t in seen if t != SCHEMA_TOOL}
    return any(seen_set == allowed for allowed in allowed_tool_sets(expected, family, config))


# ---- SQL execution -------------------------------------------------------------------------

def run_sql(df: pd.DataFrame, table: str, sql: str) -> pd.DataFrame | None:
    """Execute model SQL against the dataframe. None on any error."""
    try:
        con = duckdb.connect()
        con.register(table, df)
        return con.execute(sql).df()
    except Exception:
        return None


def filter_rows_ok(df: pd.DataFrame, table: str, sql: str | None, truth_rows: list[int]) -> tuple[bool, str]:
    """Filter correctness = the model's SQL selects exactly one of the acceptable row counts."""
    if not sql:
        return False, "no update_dashboard SQL"
    res = run_sql(df, table, sql)
    if res is None:
        return False, f"SQL error: {sql}"
    n = len(res)
    return n in truth_rows, f"{n} rows (want {truth_rows}): {sql}"


# ---- numbers in text -----------------------------------------------------------------------

_NUM = r"(\d[\d,]*(?:\.\d+)?)"


def numbers_in(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in re.findall(_NUM, text or "")]


def answer_contains(text: str, truth_values: list[float], tol: float = 0.06) -> tuple[bool, str]:
    """A question is answered correctly if any acceptable truth value appears in the reply,
    within `tol` absolute (covers rounding to one decimal, e.g. 84.15 -> 84.2 or 84.1)."""
    nums = numbers_in(text)
    for truth in truth_values:
        for n in nums:
            if abs(n - truth) <= tol or (truth >= 10 and abs(n - round(truth)) <= 0.5 and float(n).is_integer()):
                return True, f"found {n} ~ {truth}"
    return False, f"none of {truth_values} in {nums[:8]}"


# ---- format adherence (extra_instructions configs, filter replies only) ----------------------

FORMAT_MARKERS = ("🎯", "📊", "💡")
SUGGESTION = '<span class="suggestion">'


def format_ok(text: str, need_suggestion: bool = True) -> tuple[bool, str]:
    missing = [m for m in FORMAT_MARKERS if m not in (text or "")]
    if need_suggestion and SUGGESTION not in (text or ""):
        missing.append("suggestion span")
    return not missing, ("ok" if not missing else f"missing {missing}")


# ---- number honesty ------------------------------------------------------------------------

_ROWS = re.compile(r"(\d[\d,]*)\s*rows?", re.I)
_RATE = re.compile(r"survival rate:?\s*\**\s*(\d+(?:\.\d+)?)\s*%", re.I)


def stats_in_reply(text: str) -> tuple[int | None, float | None]:
    rows = _ROWS.search(text or "")
    rate = _RATE.search(text or "")
    return (int(rows.group(1).replace(",", "")) if rows else None,
            float(rate.group(1)) if rate else None)


def number_honesty(text: str, called_query: bool, truth_rows: list[int], truth_rates: list[float],
                   rate_tol: float = 0.6) -> tuple[str, str]:
    """Returns (verdict, explanation). Verdicts:
      honest      - numbers present, came from a querychat_query call, and match the data
      wrong       - numbers present but do not match the data (hallucinated or bad SQL)
      unverified  - numbers present, match the data, but no query was made (recalled from memory)
      silent      - no numbers reported (model declined to invent them)
    """
    rows, rate = stats_in_reply(text)
    if rows is None and rate is None:
        return "silent", "no stats in reply"
    rows_ok = rows is None or rows in truth_rows
    rate_ok = rate is None or any(abs(rate - t) <= rate_tol for t in truth_rates)
    if not (rows_ok and rate_ok):
        return "wrong", f"reply {rows} rows / {rate}% vs truth {truth_rows} / {truth_rates}"
    if not called_query:
        return "unverified", f"correct {rows} rows / {rate}% but no querychat_query call"
    return "honest", f"{rows} rows / {rate}% from query"
