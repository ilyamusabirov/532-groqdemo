"""Deterministic scorers. Each returns CORRECT / INCORRECT, or NOANSWER ("N") when the
check does not apply to the sample, so report.py can compute accuracy over applicable samples only."""

from __future__ import annotations

from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState

from evals import checks
from evals.datasets import FRAMES
from evals.prompts import FORMAT_CONFIGS

_FRAME_CACHE = {}


def _frame(name):
    if name not in _FRAME_CACHE:
        _FRAME_CACHE[name] = FRAMES[name]()
    return _FRAME_CACHE[name]


def _na(why: str) -> Score:
    return Score(value=NOANSWER, explanation=why)


@scorer(metrics=[accuracy()])
def tool_choice():
    """Did the model call exactly the expected querychat tool(s)? API errors count as failures."""

    async def score(state: TaskState, target: Target) -> Score:
        seen = state.store.get("tools") or []
        if state.store.get("error"):
            return Score(value=INCORRECT, answer=str(seen), explanation=f"API error: {state.store.get('error')}")
        family, config = state.metadata["family"], state.store.get("config") or ""
        ok = checks.tool_choice_ok(seen, state.metadata["expected_tools"], family, config)
        allowed = checks.allowed_tool_sets(state.metadata["expected_tools"], family, config)
        return Score(value=CORRECT if ok else INCORRECT, answer=str(seen),
                     explanation=f"allowed {[sorted(a) for a in allowed]} under {config}")

    return score


@scorer(metrics=[accuracy()])
def filter_rows():
    """Filter intents: the update_dashboard SQL selects the right rows (exact count match)."""

    async def score(state: TaskState, target: Target) -> Score:
        if state.metadata["family"] != "filter":
            return _na("not a filter intent")
        ok, why = checks.filter_rows_ok(_frame(state.metadata["dataset"]), state.metadata["dataset"],
                                        state.store.get("update_sql"), state.metadata["truth_rows"])
        return Score(value=CORRECT if ok else INCORRECT, answer=state.store.get("update_sql") or "", explanation=why)

    return score


@scorer(metrics=[accuracy()])
def answer_correct():
    """Question intents: the reply states an acceptable truth value."""

    async def score(state: TaskState, target: Target) -> Score:
        if state.metadata["family"] != "question":
            return _na("not a question intent")
        ok, why = checks.answer_contains(state.store.get("text") or "", state.metadata["truth_values"])
        return Score(value=CORRECT if ok else INCORRECT, answer=(state.store.get("text") or "")[:200], explanation=why)

    return score


@scorer(metrics=[accuracy()])
def format_adherence():
    """Filter replies under the extra_instructions configs follow the 4-line template."""

    async def score(state: TaskState, target: Target) -> Score:
        if state.metadata["family"] != "filter" or state.store.get("config") not in FORMAT_CONFIGS:
            return _na("format not demanded for this sample")
        need_span = state.metadata["dataset"] == "titanic"  # crashes template has no 🔍 line
        ok, why = checks.format_ok(state.store.get("text") or "", need_suggestion=need_span)
        return Score(value=CORRECT if ok else INCORRECT, answer=(state.store.get("text") or "")[:200], explanation=why)

    return score


@scorer(metrics=[accuracy()])
def number_honesty():
    """Filter replies under format configs: reported stats are correct AND came from a query.
    CORRECT = honest; INCORRECT = wrong or unverified (recalled from memory); NOANSWER = silent/n.a.
    The verdict string is kept in metadata for the report."""

    async def score(state: TaskState, target: Target) -> Score:
        if state.metadata["family"] != "filter" or state.store.get("config") not in FORMAT_CONFIGS:
            return _na("stats not demanded for this sample")
        called_query = "querychat_query" in (state.store.get("tools") or [])
        verdict, why = checks.number_honesty(state.store.get("text") or "", called_query,
                                             state.metadata["truth_rows"], state.metadata["truth_rates"])
        value = CORRECT if verdict == "honest" else NOANSWER if verdict == "silent" else INCORRECT
        return Score(value=value, answer=verdict, explanation=why, metadata={"verdict": verdict})

    return score


ALL_SCORERS = [tool_choice, filter_rows, answer_correct, format_adherence, number_honesty]
