"""inspect_ai task: querychat behaviour on Groq-served models.

    uv run inspect eval evals/qc_eval.py -T model=qwen/qwen3.8-27b -T config=format_query --epochs 3
    uv run inspect view

The model is driven by chatlas inside the solver, so inspect's own --model is not used
(a mock is declared to satisfy inspect)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset

from evals.prompts import CONFIGS
from evals.scorers import ALL_SCORERS
from evals.solver import querychat_solver

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
DATA = Path(__file__).parent / "data"


@task
def querychat_eval(model: str = "qwen/qwen3.8-27b", config: str = "baseline", dataset: str = "titanic") -> Task:
    if config not in CONFIGS[dataset]:
        raise ValueError(f"config {config!r} not defined for {dataset}; choose from {list(CONFIGS[dataset])}")
    return Task(
        dataset=json_dataset(str(DATA / f"{dataset}.jsonl")),
        solver=querychat_solver(model=model, config=config),
        scorer=[s() for s in ALL_SCORERS],
        model="mockllm/model",
        metadata={"groq_model": model, "config": config, "dataset": dataset},
        name=f"qc_{dataset}_{config}_{model.split('/')[-1]}",
    )
