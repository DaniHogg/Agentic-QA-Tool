"""Orchestrator agent — classifies the target and routes the pipeline."""

from __future__ import annotations

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_qa.state import QAState, TargetType


_SYSTEM_PROMPT = """You are the Orchestrator of an autonomous QA system.
Your only job is to classify the user-provided target into exactly one of three categories:

- "api"  — a REST API URL (http/https) OR a local OpenAPI/Swagger YAML/JSON spec file
- "web"  — a website URL intended for UI/browser testing
- "code" — a local file or directory containing Python source code to unit-test

Rules:
1. If the target starts with http/https AND looks like a raw API base (no HTML pages), output "api".
2. If the target starts with http/https AND appears to be a web application, output "web".
3. If the target is a file path ending in .yaml, .yml, or .json, output "api".
4. If the target is a file path to .py files or a directory, output "code".

Respond with ONLY the single word: api, web, or code. No explanation.
"""


def orchestrate(state: QAState) -> dict:
    """Classify the target and update state with target_type."""

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=0)

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Target: {state.target}"),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip().lower()

    # Guard: only accept known values, default to "api"
    target_type: TargetType = raw if raw in ("api", "web", "code") else "api"

    return {
        "target_type": target_type,
        "messages": state.messages + messages + [response],
    }
