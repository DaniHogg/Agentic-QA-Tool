"""Planner agent — inspects the target and produces a structured test plan."""

from __future__ import annotations

import os
import httpx
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from agentic_qa.state import QAState
from agentic_qa.llm import get_chat_model


_SYSTEM_PROMPT = """You are the QA Test Planner in an autonomous multi-agent testing system.

Given information about a target (API, website, or codebase), produce a concise, structured
test plan in Markdown. The plan must include:

1. **Summary** — what the target is and what you intend to test
2. **Test Cases** — a numbered list, each with:
   - ID (TC-001, TC-002, ...)
   - Name (short descriptive title)
   - Type (positive / negative / edge-case / performance)
   - What to test
   - Expected outcome
3. **Risks & Assumptions** — any known gaps or assumptions

{max_tests_guidance}

**Important:** Do not include test cases that are functionally identical to previous cases. If two cases would test the same behavior, merge them or omit the duplicate. Prioritize unique coverage and avoid redundancy.

You must adapt the plan to the requested strategy:
- smoke: critical path only, high-signal checks, minimal set
- sanity: basic feature health and key integrations
- regression: broad and deep coverage across happy, negative, and edge paths
- custom: prioritise user-provided notes exactly
"""


def _fetch_api_context(target: str) -> str:
    """Return a summary of API endpoints from a URL or local spec file."""
    if target.startswith("http"):
        # Try common OpenAPI spec paths
        for suffix in ("", "/openapi.json", "/openapi.yaml", "/swagger.json", "/docs/openapi.json"):
            try:
                r = httpx.get(target.rstrip("/") + suffix, timeout=10, follow_redirects=True)
                if r.status_code == 200:
                    ct = r.headers.get("content-type", "")
                    if "json" in ct or "yaml" in ct or suffix.endswith((".json", ".yaml")):
                        return f"OpenAPI spec fetched from {target + suffix}:\n\n{r.text[:4000]}"
            except httpx.RequestError:
                continue
        # Fallback: treat as a plain base URL
        return f"REST API base URL: {target}\nNo spec found — infer endpoints from common REST conventions."
    else:
        path = Path(target)
        if path.exists():
            content = path.read_text(encoding="utf-8")[:4000]
            return f"Local spec file ({path.name}):\n\n{content}"
        return f"Target path not found: {target}"


def _fetch_web_context(target: str) -> str:
    """Return basic page context from a URL."""
    try:
        r = httpx.get(target, timeout=10, follow_redirects=True)
        # Trim HTML — just send the first 3000 chars
        return f"Web page at {target} (HTTP {r.status_code}):\n\n{r.text[:3000]}"
    except httpx.RequestError as exc:
        return f"Could not fetch {target}: {exc}"


def _fetch_code_context(target: str) -> str:
    """Return a summary of Python source from a file or directory."""
    path = Path(target)
    if not path.exists():
        return f"Path not found: {target}"
    if path.is_file():
        return f"Python file ({path.name}):\n\n{path.read_text(encoding='utf-8')[:4000]}"
    # Directory — collect first 3 .py files
    py_files = list(path.rglob("*.py"))[:3]
    parts = []
    for f in py_files:
        parts.append(f"### {f.relative_to(path)}\n{f.read_text(encoding='utf-8')[:1200]}")
    return f"Python codebase at {path}:\n\n" + "\n\n".join(parts)


def plan(state: QAState) -> dict:
    """Inspect the target and produce a structured test plan."""

    max_tests = int(os.getenv("MAX_TEST_CASES", "10"))
    limit_count = state.limit_test_count
    llm = get_chat_model(temperature=0.2)

    # Build guidance based on limit_test_count flag
    if limit_count:
        max_tests_guidance = f"Generate only the necessary test cases up to a maximum of {max_tests}. Do not artificially pad the plan — stop once all required coverage is achieved, even if below the max."
    else:
        max_tests_guidance = "Generate only the test cases that are necessary to cover the identified requirements and strategies. Do not artificially pad the plan to reach a target count — quality and coverage matter more than quantity."

    # Gather raw context based on target type
    tt = state.target_type or "api"
    if tt == "api":
        context = _fetch_api_context(state.target)
    elif tt == "web":
        context = _fetch_web_context(state.target)
    else:
        context = _fetch_code_context(state.target)

    system = _SYSTEM_PROMPT.format(max_tests_guidance=max_tests_guidance)
    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content=(
                f"Target type: {tt}\n"
                f"Requested strategy: {state.test_strategy}\n"
                f"User planning notes: {state.planning_notes or 'None'}\n\n"
                f"Target context:\n{context}"
            )
        ),
    ]

    response = llm.invoke(messages)

    return {
        "test_plan": response.content,
        "messages": state.messages + messages + [response],
    }
