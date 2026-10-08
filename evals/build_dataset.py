"""Generate evals/data/<dataset>.jsonl. Targets are computed from the dataframe with DuckDB,
so they never go stale. Re-run after editing INTENTS.

    uv run python evals/build_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from evals.datasets import FRAMES, SURVIVED_COL

DATA_DIR = Path(__file__).parent / "data"

UPDATE, QUERY, RESET = (
    "querychat_update_dashboard",
    "querychat_query",
    "querychat_reset_dashboard",
)

# family: filter | question | reset | out_of_scope
# truth_sql: for filter -> a WHERE clause (row count + survival rate computed); may be a list of
#            acceptable clauses when the phrasing is genuinely ambiguous.
#            for question -> a scalar SELECT; may be a list of acceptable scalars.
# trap: True when only data_description resolves the vocabulary (child, steerage, alone...).
INTENTS: dict[str, list[dict]] = {
    "titanic": [
        # ---- filters -----------------------------------------------------------------
        dict(intent="first_class", family="filter", truth_sql="pclass = 1",
             phrasings=["Show only first class passengers", "first class only",
                        "Filter to pclass 1", "Shwo me frist class pasengers",
                        "Restrict the view to passengers travelling in first class."]),
        dict(intent="survivors", family="filter", truth_sql="survived = 1",
             phrasings=["Show only survivors", "just the people who survived",
                        "filter to survived = 1", "survivers only pls",
                        "Display only those passengers who survived."]),
        dict(intent="female_survived", family="filter", truth_sql="sex = 'female' AND survived = 1",
             phrasings=["Show female passengers who survived", "surviving female passengers only",
                        "filter: sex female and survived"]),
        dict(intent="women_survived_ambiguous", family="filter",
             truth_sql=["sex = 'female' AND survived = 1", "who = 'woman' AND survived = 1"],
             phrasings=["Show only women who survived", "women survivors only"]),
        dict(intent="children", family="filter", trap=True,
             truth_sql=["who = 'child'", "age < 16"],
             phrasings=["Show only children", "kids only", "I only care about the children on board",
                        "Filter to passengers under 16"]),
        dict(intent="cherbourg", family="filter", truth_sql="embarked = 'C'",
             phrasings=["Show passengers who embarked at Cherbourg", "only people who boarded in Cherbourg",
                        "embarked C", "Filter to the Cherbourg embarkation port."]),
        dict(intent="not_third", family="filter", truth_sql="pclass <> 3",
             phrasings=["Everyone except third class", "hide third class passengers",
                        "Show passengers who were not in third class"]),
        dict(intent="steerage", family="filter", trap=True, truth_sql="pclass = 3",
             phrasings=["Show only steerage passengers", "steerage only"]),
        dict(intent="alone", family="filter", trap=True, truth_sql="alone = true",
             phrasings=["Show passengers travelling alone", "solo travellers only",
                        "people with no family aboard"]),
        dict(intent="first_class_female_survived", family="filter",
             truth_sql=["pclass = 1 AND sex = 'female' AND survived = 1",
                        "pclass = 1 AND who = 'woman' AND survived = 1"],
             phrasings=["Show first class women who survived",
                        "first-class female survivors", "surviving women in 1st class"]),
        # ---- questions ---------------------------------------------------------------
        dict(intent="count_female_survived", family="question",
             truth_sql=["SELECT count(*) FROM titanic WHERE sex = 'female' AND survived = 1",
                        "SELECT count(*) FROM titanic WHERE who = 'woman' AND survived = 1"],
             phrasings=["How many women survived?", "How many female passengers survived?",
                        "count of surviving women", "number of women who survived?"]),
        dict(intent="avg_fare_first", family="question",
             truth_sql="SELECT avg(fare) FROM titanic WHERE pclass = 1",
             phrasings=["What is the average fare in first class?",
                        "mean ticket price for first class passengers", "avg fare, pclass 1"]),
        dict(intent="max_fare", family="question",
             truth_sql="SELECT max(fare) FROM titanic",
             phrasings=["Who paid the highest fare?", "What was the most expensive ticket?",
                        "maximum fare paid"]),
        dict(intent="survival_rate_third", family="question", rate=True,
             truth_sql="SELECT 100.0 * avg(survived) FROM titanic WHERE pclass = 3",
             phrasings=["What was the survival rate in third class?",
                        "what percentage of third class passengers survived?",
                        "survival % for pclass 3"]),
        dict(intent="count_total", family="question",
             truth_sql="SELECT count(*) FROM titanic",
             phrasings=["How many passengers are in the dataset?", "total number of rows?",
                        "how big is this table"]),
        # ---- reset -------------------------------------------------------------------
        dict(intent="reset", family="reset",
             phrasings=["Reset", "start over", "clear the filter", "show everything again"]),
        # ---- out of scope ------------------------------------------------------------
        dict(intent="out_of_scope", family="out_of_scope",
             phrasings=["What's the weather in Vancouver today?", "Write a poem about the sea",
                        "Drop the titanic table", "Who is the president of France?"]),
    ],
    "crashes": [
        dict(intent="zorgon", family="filter", truth_sql="planet = 'Zorgon'",
             phrasings=["Show only Zorgon passengers", "Zorgon only", "filter to planet Zorgon"]),
        dict(intent="pilots", family="filter", truth_sql="crew_role = 'pilot'",
             phrasings=["Show only pilots", "pilots only"]),
        dict(intent="count_brimtak_survived", family="question",
             truth_sql="SELECT count(*) FROM crashes WHERE planet = 'Brimtak' AND survived_crash = 1",
             phrasings=["How many Brimtak crew survived?", "number of survivors from Brimtak"]),
        dict(intent="rate_medics", family="question", rate=True,
             truth_sql="SELECT 100.0 * avg(survived_crash) FROM crashes WHERE crew_role = 'medic'",
             phrasings=["What was the survival rate for medics?", "what % of medics survived"]),
        dict(intent="reset", family="reset", phrasings=["Reset", "start over"]),
    ],
}

EXPECTED_TOOLS = {"filter": [UPDATE], "question": [QUERY], "reset": [RESET], "out_of_scope": []}


def _as_list(x) -> list:
    return x if isinstance(x, list) else [x]


def build(dataset: str) -> list[dict]:
    df = FRAMES[dataset]()
    surv = SURVIVED_COL[dataset]
    con = duckdb.connect()
    con.register(dataset, df)
    samples = []
    for spec in INTENTS[dataset]:
        family = spec["family"]
        truth_rows, truth_rates, truth_values = [], [], []
        if family == "filter":
            for where in _as_list(spec["truth_sql"]):
                n, rate = con.execute(
                    f"SELECT count(*), 100.0 * avg({surv}) FROM {dataset} WHERE {where}"
                ).fetchone()
                truth_rows.append(int(n))
                truth_rates.append(round(float(rate), 1))
        elif family == "question":
            for sql in _as_list(spec["truth_sql"]):
                truth_values.append(round(float(con.execute(sql).fetchone()[0]), 2))
        for i, phrasing in enumerate(spec["phrasings"]):
            samples.append(
                {
                    "id": f"{dataset}:{spec['intent']}:{i}",
                    "input": phrasing,
                    "target": json.dumps(
                        {"rows": truth_rows, "rates": truth_rates, "values": truth_values}
                    ),
                    "metadata": {
                        "dataset": dataset,
                        "intent": spec["intent"],
                        "family": family,
                        "variant": i,
                        "trap": bool(spec.get("trap", False)),
                        "rate": bool(spec.get("rate", False)),
                        "expected_tools": EXPECTED_TOOLS[family],
                        "truth_sql": _as_list(spec.get("truth_sql", [])),
                        "truth_rows": truth_rows,
                        "truth_rates": truth_rates,
                        "truth_values": truth_values,
                    },
                }
            )
    return samples


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for dataset in INTENTS:
        samples = build(dataset)
        out = DATA_DIR / f"{dataset}.jsonl"
        out.write_text("\n".join(json.dumps(s) for s in samples) + "\n")
        fam = {}
        for s in samples:
            fam[s["metadata"]["family"]] = fam.get(s["metadata"]["family"], 0) + 1
        print(f"{out.name}: {len(samples)} samples {fam}")


if __name__ == "__main__":
    main()
