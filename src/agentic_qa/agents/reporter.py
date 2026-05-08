"""Reporter agent — synthesises results into a human-readable Markdown report."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_qa.state import QAState


_SYSTEM_PROMPT = """You are the QA Reporter in an autonomous multi-agent testing system.
You receive a test plan, the generated test code, and the pytest execution output.
Write a professional Markdown QA report with the following sections:

# QA Report — {target}

## Summary
- Date/Time
- Target
- Target Type
- Tests Run / Passed / Failed / Errors
- Overall Status: ✅ PASS or ❌ FAIL

## Test Results
A table with columns: ID | Name | Status | Notes

## Failure Analysis
For each failed or errored test, explain the likely root cause and a recommended fix.

## Coverage Gaps
List any important scenarios NOT covered by the generated tests.

## Recommendations
Actionable next steps for the development team.

---
Keep language professional but concise. Use emoji sparingly (✅ ❌ ⚠️ only).
"""


def report(state: QAState) -> dict:
    """Generate a Markdown QA report from execution results."""

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0.3)

    context = f"""
Target: {state.target}
Target Type: {state.target_type}
Tests Passed: {state.tests_passed}
Tests Failed: {state.tests_failed}
Tests Errors: {state.tests_errors}
Execution Success: {state.execution_success}

=== TEST PLAN ===
{state.test_plan}

=== GENERATED TEST CODE ===
{state.generated_tests}

=== PYTEST OUTPUT ===
{state.execution_output}
""".strip()

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT.format(target=state.target)),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    report_md = response.content

    # Save report to disk
    reports_dir = Path(os.getenv("REPORTS_DIR", "./reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_target = _slugify(state.target)
    report_path = reports_dir / f"report_{safe_target}_{ts}.md"
    report_path.write_text(report_md, encoding="utf-8")

    return {
        "report": report_md,
        "report_path": str(report_path.resolve()),
        "messages": state.messages + messages + [response],
    }


def _slugify(text: str) -> str:
    """Turn a URL or path into a safe filename segment."""
    import re
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    return text[:40].strip("_")
