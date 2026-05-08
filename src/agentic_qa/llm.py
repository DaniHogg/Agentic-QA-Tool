"""Provider-agnostic chat model factory for Agentic QA."""

from __future__ import annotations

import os

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


def get_chat_model(*, temperature: float):
    """Return a configured chat model based on LLM_PROVIDER env var.

    Supported values:
    - openai (default)
    - anthropic
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "anthropic":
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return ChatAnthropic(model=model, temperature=temperature)

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    return ChatOpenAI(model=model, temperature=temperature)
