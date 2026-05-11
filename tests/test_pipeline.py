"""Unit tests for the Agentic QA pipeline (no LLM calls — mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from agentic_qa.state import QAState
from agentic_qa.agents.executor import _parse_summary, execute
from agentic_qa.agents.writer import _strip_markdown_fences, _build_per_test_files
from agentic_qa.agents.reporter import _slugify


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
