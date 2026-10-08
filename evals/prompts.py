"""System-prompt customizations under test, copied verbatim from the DSCI 532 lecture 5 slides
(slides/07-llm-dev-a.qmd, 07-llm-dev-b.qmd) and the lecture hallucination-proof script."""

# --- Titanic -------------------------------------------------------------------------------
DATA_DESCRIPTION = """
Titanic passenger manifest (891 passengers).

Column meanings:
- survived: 1 = survived, 0 = died
- pclass: ticket class — 1 = First (luxury), 2 = Second, 3 = Third (steerage)
- sex: passenger sex ('male' or 'female')
- age: age in years (some missing)
- sibsp: number of siblings or spouses aboard
- parch: number of parents or children aboard
- fare: ticket price in pounds sterling
- embarked: port of embarkation — C = Cherbourg, Q = Queenstown, S = Southampton
- who: 'man', 'woman', or 'child' (child = under 16)
- alone: True if travelling alone (sibsp=0 and parch=0)
- alive: 'yes' or 'no' (same as survived, string version)
- deck: cabin deck (A–G), many missing
"""

# Monday slide: format only, no instruction to query.
EXTRA_FORMAT = """
When filtering the data, always reply using EXACTLY this format:

🎯 **Filter:** [one sentence describing what was selected]
📊 **Stats:** [N rows] | survival rate: [X%] | overall: 38.4%
💡 **Insight:** [one sentence: did this group survive better or worse than average?]
🔍 **Try next:** <span class="suggestion">[a natural follow-up question]</span>
"""

# Wednesday slide / app-07e: tells the model to call querychat_query for the stats.
EXTRA_FORMAT_QUERY = """
When filtering the data, always reply using EXACTLY this format — no exceptions:

🎯 **Filter:** [one sentence describing what was selected]
📊 **Stats:** use querychat_query to calculate row count and survival rate, then write: [N rows] | survival rate: [X%] | overall: 38.4%
💡 **Insight:** [one sentence interpreting whether this group survived better or worse than average]
🔍 **Try next:** <span class="suggestion">[a natural follow-up question]</span>
"""

# --- Crashes (fake data from test_hallucination_proof.py; overall survival = 58%) ---------
CRASHES_FORMAT = """
Always reply using EXACTLY this format — no exceptions:

🎯 **Filter:** [one sentence describing what was selected]
📊 **Stats:** [N rows] | survival rate: [X%] | overall: 58%
💡 **Insight:** [one sentence]
"""

CRASHES_FORMAT_QUERY = """
When filtering, always reply using EXACTLY this format — no exceptions:

🎯 **Filter:** [one sentence describing what was selected]
📊 **Stats:** FIRST call querychat_query to calculate count and survival rate, then write: [N rows] | survival rate: [X%] | overall: 58%
💡 **Insight:** [one sentence]
"""

# config name -> QueryChat kwargs, per dataset
CONFIGS: dict[str, dict[str, dict[str, str | None]]] = {
    "titanic": {
        "baseline": {"data_description": None, "extra_instructions": None},
        "datadesc": {"data_description": DATA_DESCRIPTION, "extra_instructions": None},
        "format": {"data_description": DATA_DESCRIPTION, "extra_instructions": EXTRA_FORMAT},
        "format_query": {"data_description": DATA_DESCRIPTION, "extra_instructions": EXTRA_FORMAT_QUERY},
    },
    "crashes": {
        "baseline": {"data_description": None, "extra_instructions": None},
        "format": {"data_description": None, "extra_instructions": CRASHES_FORMAT},
        "format_query": {"data_description": None, "extra_instructions": CRASHES_FORMAT_QUERY},
    },
}

# Which configs demand the 4-line format on filter replies (format scorers apply there).
FORMAT_CONFIGS = {"format", "format_query"}

MODELS = [
    "qwen/qwen3.6-27b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
