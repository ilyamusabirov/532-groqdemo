import pandas as pd
import pytest

from evals import checks
from evals.datasets import crashes


def test_tool_choice_ignores_schema_tool():
    assert checks.tool_choice_ok(["querychat_get_schema", "querychat_query"], ["querychat_query"])
    assert not checks.tool_choice_ok(["querychat_update_dashboard"], ["querychat_query"])
    assert checks.tool_choice_ok(["querychat_get_schema"], [])
    assert not checks.tool_choice_ok(["querychat_update_dashboard", "querychat_query"], ["querychat_update_dashboard"])


def test_filter_rows_ok_counts_and_case():
    df = crashes()
    ok, why = checks.filter_rows_ok(df, "crashes", "SELECT * FROM crashes WHERE planet = 'Zorgon'", [30])
    assert ok, why
    ok, _ = checks.filter_rows_ok(df, "crashes", "SELECT * FROM crashes WHERE planet = 'ZORGON'", [30])
    assert not ok  # case-sensitive: zero rows, like the class = 'FIRST' bug seen in the spike
    ok, why = checks.filter_rows_ok(df, "crashes", "SELECT * FROM nope", [30])
    assert not ok and "SQL error" in why
    assert checks.filter_rows_ok(df, "crashes", None, [30])[0] is False


def test_answer_contains_rounding():
    assert checks.answer_contains("The average fare is **84.15**", [84.15])[0]
    assert checks.answer_contains("about 84.2 pounds", [84.15])[0]
    assert checks.answer_contains("**233 women survived**", [233.0, 205.0])[0]
    assert checks.answer_contains("205 survived", [233.0, 205.0])[0]
    assert not checks.answer_contains("roughly 300", [233.0])[0]
    assert checks.answer_contains("1,234 rows", [1234.0])[0]


def test_format_ok():
    good = '🎯 **Filter:** x\n📊 **Stats:** 216 rows | survival rate: 62.5% | overall: 38.4%\n💡 **Insight:** y\n🔍 **Try next:** <span class="suggestion">z</span>'
    assert checks.format_ok(good) == (True, "ok")
    ok, why = checks.format_ok("The dashboard now shows first class.")
    assert not ok and "🎯" in why
    assert checks.format_ok(good.split("🔍")[0], need_suggestion=False)[0]


def test_stats_in_reply():
    assert checks.stats_in_reply("📊 **Stats:** 216 rows | survival rate: 62.5% | overall: 38.4%") == (216, 62.5)
    assert checks.stats_in_reply("📊 **Stats:** [N rows] | survival rate: Unknown") == (None, None)
    assert checks.stats_in_reply("**Stats:** 1,234 rows | survival rate: **40%**") == (1234, 40.0)


@pytest.mark.parametrize(
    "text,called,verdict",
    [
        ("216 rows | survival rate: 62.5%", True, "honest"),
        ("216 rows | survival rate: 62.5%", False, "unverified"),
        ("200 rows | survival rate: 62.5%", True, "wrong"),
        ("216 rows | survival rate: 50%", False, "wrong"),
        ("Stats: unknown without query", False, "silent"),
    ],
)
def test_number_honesty(text, called, verdict):
    assert checks.number_honesty(text, called, [216], [63.0])[0] == verdict


def test_tool_choice_depends_on_config():
    U, Q = "querychat_update_dashboard", "querychat_query"
    assert checks.tool_choice_ok([U], [U], "filter", "baseline")
    assert not checks.tool_choice_ok([U, Q], [U], "filter", "baseline")
    assert checks.tool_choice_ok([U, Q], [U], "filter", "format_query")
    assert not checks.tool_choice_ok([U], [U], "filter", "format_query")
    assert checks.tool_choice_ok([U], [U], "filter", "format")
    assert checks.tool_choice_ok([U, Q], [U], "filter", "format")
    assert checks.tool_choice_ok([Q], [Q], "question", "format_query")  # questions unaffected
