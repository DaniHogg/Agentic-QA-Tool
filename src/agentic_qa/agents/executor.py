"""Executor agent — runs the generated test file and captures results."""

from __future__ import annotations

import re
import subprocess
import sys

from agentic_qa.state import QAState


def execute(state: QAState) -> dict:
    """Run the generated test file with pytest and capture output."""

    if not state.test_file_path:
        return {
            "execution_output": "No test file was generated.",
            "execution_success": False,
            "errors": state.errors + ["Executor: test_file_path is None"],
        }

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            state.test_file_path,
            "--tb=short",
            "-v",
            "--no-header",
            "--color=no",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = result.stdout + result.stderr
    success = result.returncode == 0
    passed, failed, errors = _parse_summary(output)

    return {
        "execution_output": output,
        "execution_success": success,
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_errors": errors,
    }


def _parse_summary(output: str) -> tuple[int, int, int]:
    """Extract pass/fail/error counts from pytest's short summary line."""
    # e.g. "5 passed, 2 failed, 1 error in 3.14s"
    passed = _extract(r"(\d+) passed", output)
    failed = _extract(r"(\d+) failed", output)
    errors = _extract(r"(\d+) error", output)
    return passed, failed, errors


def _extract(pattern: str, text: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0
