"""inspect_ai solver that drives the real querychat stack (system prompt + tools) through chatlas.
Each sample gets a fresh QueryChat so conversations never leak between samples."""

from __future__ import annotations

import asyncio
import re
import time
import warnings

import querychat
from chatlas import ChatAnthropic, ChatGithub, ChatGroq, ChatOpenAI
from chatlas.types import ContentText, ContentToolRequest
from inspect_ai.solver import Generate, TaskState, solver

from evals.datasets import FRAMES
from evals.prompts import CONFIGS

warnings.filterwarnings("ignore", message="Visualization tools require")

_FRAME_CACHE = {}

PROVIDERS = {
    "groq": lambda model: ChatGroq(model=model),
    "github": lambda model: ChatGithub(model=model),          # GitHub Models (free tier), GITHUB_TOKEN
    "openai": lambda model: ChatOpenAI(model=model),          # OPENAI_API_KEY
    "anthropic": lambda model: ChatAnthropic(model=model),    # ANTHROPIC_API_KEY; chatlas enables 5-min prompt caching
}


def make_client(provider: str, model: str):
    if provider not in PROVIDERS:
        raise ValueError(f"provider {provider!r} not in {list(PROVIDERS)}")
    return PROVIDERS[provider](model)


def _frame(name):
    if name not in _FRAME_CACHE:
        _FRAME_CACHE[name] = FRAMES[name]()
    return _FRAME_CACHE[name]


def _one_attempt(model: str, dataset: str, config: str, user_message: str, reasoning: str = "default",
                 provider: str = "groq") -> dict:
    df = _frame(dataset)
    kwargs = CONFIGS[dataset][config]
    qc = querychat.QueryChat(
        df,
        dataset,
        tools=("filter", "query"),  # no visualize tool: matches the lecture apps
        client=make_client(provider, model),
        data_description=kwargs["data_description"],
        extra_instructions=kwargs["extra_instructions"],
    )
    update_sql: list[str] = []
    resets: list[bool] = []
    chat = qc.client(
        update_dashboard=lambda d: update_sql.append(d["query"]),
        reset_dashboard=lambda *args, **kw: resets.append(True),
    )
    out = {"tools": [], "update_sql": None, "query_sql": [], "text": "", "error": None, "reset": False, "retries": 0,
           "tokens_in": 0, "tokens_out": 0, "tokens_cached": 0}
    # Groq's reasoning_effort switch (Qwen: thinking vs instruct mode). "default" sends nothing; Groq only.
    call_kwargs = {"reasoning_effort": reasoning} if (reasoning != "default" and provider == "groq") else None
    try:
        chat.chat(user_message, echo="none", kwargs=call_kwargs)
    except Exception as e:  # Groq schema validation, rate limit, etc. Recorded, not raised.
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    turns = chat.get_turns()
    for turn in turns:
        if turn.role == "assistant" and turn.tokens:  # (input, output, cached) per API call, from provider usage
            out["tokens_in"] += int(turn.tokens[0] or 0)
            out["tokens_out"] += int(turn.tokens[1] or 0)
            out["tokens_cached"] += int(turn.tokens[2] or 0) if len(turn.tokens) > 2 else 0
        for c in turn.contents:
            if isinstance(c, ContentToolRequest):
                out["tools"].append(c.name)
                if c.name == "querychat_query":
                    out["query_sql"].append((c.arguments or {}).get("query", ""))
    if turns and turns[-1].role == "assistant":
        # text only: drops Qwen reasoning blocks and tool-call content
        out["text"] = "\n".join(c.text for c in turns[-1].contents if isinstance(c, ContentText)).strip()
    out["update_sql"] = update_sql[-1] if update_sql else None
    out["reset"] = bool(resets)
    return out


_RETRY_AFTER = re.compile(r"try again in ([\d.]+)(ms|s)")


def run_querychat(model: str, dataset: str, config: str, user_message: str, max_retries: int = 6,
                  reasoning: str = "default", provider: str = "groq") -> dict:
    """One-shot run with retries on Groq rate limits (429). Each retry starts a fresh conversation
    so a partially-completed tool loop never leaks into the recorded result."""
    for attempt in range(max_retries + 1):
        out = _one_attempt(model, dataset, config, user_message, reasoning, provider)
        out["retries"] = attempt
        err = out["error"] or ""
        if "RateLimitError" not in err or attempt == max_retries:
            return out
        m = _RETRY_AFTER.search(err)
        wait = (float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)) if m else 2.0
        time.sleep(min(wait, 20) + 1.0 + attempt)  # hinted wait + jitter-ish backoff
    return out


@solver
def querychat_solver(model: str, config: str, reasoning: str = "default", provider: str = "groq"):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        dataset = state.metadata["dataset"]
        result = await asyncio.to_thread(run_querychat, model, dataset, config, state.input_text, 6, reasoning, provider)
        for k, v in result.items():
            state.store.set(k, v)
        state.store.set("model", model)
        state.store.set("config", config)
        state.store.set("reasoning", reasoning)
        state.store.set("provider", provider)
        state.output.completion = result["text"] or (f"[ERROR] {result['error']}" if result["error"] else "")
        return state

    return solve
