"""Unit tests for the Agentic QA pipeline (no LLM calls — mocked)."""

from __future__ import annotations

import sqlite3

from agentic_qa.state import QAState
from agentic_qa.agents.executor import _parse_summary, execute
from agentic_qa.agents.writer import (
    _strip_markdown_fences,
    _build_per_test_files,
    _extract_plan_case_ids,
    _fallback_test_body,
)
from agentic_qa.agents.reporter import _slugify
from agentic_qa.reuse_engine import score_reuse_candidates
from agentic_qa.context_store import (
    finalize_memory_run,
    get_memory_schema_version,
    init_memory_store,
    list_memory_runs,
    load_plan_case_feedback,
    load_reuse_decision_events,
    load_run_feedback,
    load_test_file_feedback,
    save_reuse_decision_events,
    save_run_feedback,
    save_test_file_feedback,
    save_plan_case_feedback,
    start_memory_run,
)


# ── State ─────────────────────────────────────────────────────────────────────

def test_qa_state_defaults():
    state = QAState(target="https://example.com")
    assert state.target_type is None
    assert state.errors == []
    assert state.tests_passed == 0


def test_qa_state_with_type():
    state = QAState(target="https://api.example.com", target_type="api")
    assert state.target_type == "api"


# ── Writer helpers ─────────────────────────────────────────────────────────────

def test_strip_markdown_fences_with_python_fence():
    code = "```python\nimport pytest\n\ndef test_foo():\n    assert True\n```"
    result = _strip_markdown_fences(code)
    assert result.startswith("import pytest")
    assert "```" not in result


def test_strip_markdown_fences_plain_code():
    code = "import pytest\n\ndef test_foo():\n    assert True"
    result = _strip_markdown_fences(code)
    assert result == code


def test_strip_markdown_fences_generic_fence():
    code = "```\nimport os\n```"
    result = _strip_markdown_fences(code)
    assert "```" not in result
    assert "import os" in result


def test_build_per_test_files_splits_and_embeds_description():
    source = '''import pytest

# validates first scenario
def test_first_case():
    assert True

# validates second scenario
def test_second_case():
    assert True
'''
    modules = _build_per_test_files(source, timestamp="2026-05-11 00:00 UTC")
    assert len(modules) == 2
    assert "Description: validates first scenario" in modules[0]["code"]
    assert "Description: validates second scenario" in modules[1]["code"]


# ── Executor helpers ───────────────────────────────────────────────────────────

def test_parse_summary_all_pass():
    output = "5 passed in 1.23s"
    passed, failed, errors = _parse_summary(output)
    assert passed == 5
    assert failed == 0
    assert errors == 0


def test_parse_summary_mixed():
    output = "3 passed, 2 failed, 1 error in 4.56s"
    passed, failed, errors = _parse_summary(output)
    assert passed == 3
    assert failed == 2
    assert errors == 1


def test_parse_summary_empty():
    passed, failed, errors = _parse_summary("")
    assert passed == failed == errors == 0


def test_execute_no_test_file():
    state = QAState(target="https://example.com", test_file_path=None)
    result = execute(state)
    assert result["execution_success"] is False
    assert "errors" in result
    assert len(result["errors"]) > 0


# ── Reporter helpers ───────────────────────────────────────────────────────────

def test_slugify_url():
    slug = _slugify("https://api.example.com/v1")
    assert "://" not in slug
    assert " " not in slug


def test_slugify_path():
    slug = _slugify("/home/user/my project/code.py")
    assert " " not in slug


def test_slugify_max_length():
    long_url = "https://very-long-subdomain.example.com/api/v1/endpoint/with/many/parts"
    slug = _slugify(long_url)
    assert len(slug) <= 40


def test_memory_store_run_and_feedback(tmp_path):
    reports_dir = tmp_path / "reports"
    db_path = init_memory_store(reports_dir)
    assert db_path.exists()

    run_id = start_memory_run(
        reports_dir,
        project_name="payments-api",
        target="https://jsonplaceholder.typicode.com",
        target_type="api",
        strategy="smoke",
        model_provider="anthropic",
        model_name="claude-sonnet-4-6",
    )

    save_plan_case_feedback(
        reports_dir,
        run_id=run_id,
        case_id="TC-001",
        title="Get all posts",
        body_markdown="- TC-001: Get all posts",
        approved=True,
    )

    test_file = reports_dir / "projects" / "payments-api" / "tests" / "generated_test_tc001.py"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("def test_tc001():\n    assert True\n", encoding="utf-8")

    save_test_file_feedback(
        reports_dir,
        run_id=run_id,
        file_path=str(test_file),
        description="Get all posts",
        approved=True,
    )

    save_run_feedback(
        reports_dir,
        run_id=run_id,
        verdict="useful",
        defect_quality="true_defect",
        notes="Good output.",
    )

    finalize_memory_run(
        reports_dir,
        run_id=run_id,
        status="passed",
        tests_passed=1,
        tests_failed=0,
        tests_errors=0,
        report_path=None,
    )

    with sqlite3.connect(db_path) as conn:
        runs = conn.execute("SELECT status, tests_passed FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert runs is not None
        assert runs[0] == "passed"
        assert runs[1] == 1

        case_row = conn.execute("SELECT approved FROM plan_cases WHERE run_id = ?", (run_id,)).fetchone()
        assert case_row is not None
        assert case_row[0] == 1

        file_row = conn.execute("SELECT approved FROM test_files WHERE run_id = ?", (run_id,)).fetchone()
        assert file_row is not None
        assert file_row[0] == 1

    runs = list_memory_runs(reports_dir, project_name="payments-api", limit=5)
    assert runs
    assert runs[0]["id"] == run_id

    plan_rows = load_plan_case_feedback(reports_dir, run_id)
    assert plan_rows
    assert plan_rows[0]["approved"] == 1

    test_rows = load_test_file_feedback(reports_dir, run_id)
    assert test_rows
    assert test_rows[0]["approved"] == 1

    feedback_rows = load_run_feedback(reports_dir, run_id)
    assert feedback_rows
    assert feedback_rows[0]["verdict"] == "useful"

    save_reuse_decision_events(
        reports_dir,
        run_id=run_id,
        decisions=[
            {
                "case_id": "TC-001",
                "score": 0.91,
                "threshold_band": "high",
                "action": "reused_auto",
                "candidate_file": str(test_file),
                "generated_file": str(test_file),
                "requires_user_approval": False,
                "approved_override": False,
                "reason": "High match",
            }
        ],
    )
    reuse_rows = load_reuse_decision_events(reports_dir, run_id)
    assert reuse_rows
    assert reuse_rows[0]["action"] == "reused_auto"


def test_memory_schema_version_is_tracked(tmp_path):
    reports_dir = tmp_path / "reports"
    db_path = init_memory_store(reports_dir)

    version = get_memory_schema_version(reports_dir)
    assert version == 2

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'memory_schema_version'"
        ).fetchone()
        assert row is not None
        assert int(row[0]) == 2


def test_reuse_engine_score_only_outputs_decisions(tmp_path):
    tests_dir = tmp_path / "reports" / "projects" / "payments-api" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    existing = tests_dir / "generated_test_tc001_existing.py"
    existing.write_text(
        '"""Generated by Agentic QA.\nDescription: Get all posts\nGenerated: 2026-05-14\n"""\n'
        "import pytest\n\n"
        "def test_tc001_get_all_posts(client):\n"
        "    response = client.get('/posts')\n"
        "    assert response.status_code == 200\n",
        encoding="utf-8",
    )

    units = [
        {
            "test_name": "test_tc001_get_all_posts",
            "description": "Get all posts",
            "code": "def test_tc001_get_all_posts(client):\n    assert client.get('/posts').status_code == 200\n",
        }
    ]

    decisions = score_reuse_candidates(
        project_tests_dir=tests_dir,
        per_test_units=units,
        strategy="smoke",
        target_type="api",
    )

    assert len(decisions) == 1
    assert decisions[0]["mode"] == "score-only"
    assert "score" in decisions[0]
    assert decisions[0]["candidate_file"] is not None


def test_extract_plan_case_ids_deduplicates_and_normalizes():
    plan_text = """
    - TC-001: First
    - tc-002: Second
    - TC-001: Duplicate
    """
    case_ids = _extract_plan_case_ids(plan_text)
    assert case_ids == ["TC-001", "TC-002"]


def test_fallback_test_body_contains_case_reference():
    body = _fallback_test_body("TC-009")
    assert "TC-009" in body
    assert "assert True" in body
