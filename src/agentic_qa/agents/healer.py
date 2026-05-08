"""Healer agent — analyses failed tests and rewrites them to fix the failures."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from agentic_qa.state import QAState
from agentic_qa.agents.writer import _strip_markdown_fences
from agentic_qa.llm import get_chat_model


_SYSTEM_PROMPT = """You are the QA Self-Healer in an autonomous multi-agent testing system.
You receive pytest test code that failed, along with the failure output.

Your job:
1. Analyse each failure/error in the pytest output.
2. Rewrite the full test file so all tests pass.
3. Common fixes:
   - Wrong expected status codes → correct them based on what the API actually returned
   - Import errors → fix the import path or add missing imports
   - Assertion mismatches → fix the assertion to match actual API behaviour
   - Syntax errors → fix them
4. If a test is fundamentally broken (e.g., endpoint doesn't exist), mark it as xfail with a reason.

Output ONLY valid Python code — no markdown fences, no explanations.
Preserve all test function names exactly — just fix the body.
"""


def heal(state: QAState) -> dict:
    """Rewrite failing tests to attempt a passing run."""

    if state.execution_success:
        # Nothing to heal
        return {}

    llm = get_chat_model(temperature=0.1)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Original test code:\n\n{state.generated_tests}\n\n"
                f"Pytest output:\n\n{state.execution_output}"
            )
        ),
    ]

    response = llm.invoke(messages)
    fixed_code = _strip_markdown_fences(response.content)

    # Overwrite the test file with the healed version
    if state.test_file_path:
        Path(state.test_file_path).write_text(fixed_code, encoding="utf-8")

    return {
        "generated_tests": fixed_code,
        "messages": state.messages + messages + [response],
    }
