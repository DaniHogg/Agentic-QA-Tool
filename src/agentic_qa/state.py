"""Shared state schema passed through the LangGraph pipeline."""

from __future__ import annotations

from typing import Annotated, Literal, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field


TargetType = Literal["api", "web", "code"]


class QAState(BaseModel):
    """State that flows through every node in the QA agent graph."""

    # ── Input ─────────────────────────────────────────────────────────────────
    target: str = Field(description="URL, file path, or OpenAPI spec path provided by the user.")
    target_type: Optional[TargetType] = Field(
        default=None,
        description="Classified by the Orchestrator: 'api', 'web', or 'code'.",
    )
    test_strategy: str = Field(
        default="smoke",
        description="User-selected strategy for planning scope: smoke, sanity, regression, or custom.",
    )
    planning_notes: Optional[str] = Field(
        default=None,
        description="Optional user notes to influence generated test cases.",
    )

    # ── Planner output ────────────────────────────────────────────────────────
    test_plan: Optional[str] = Field(
        default=None,
        description="Structured test plan produced by the Planner agent (markdown).",
    )

    # ── Writer output ─────────────────────────────────────────────────────────
    generated_tests: Optional[str] = Field(
        default=None,
        description="Raw Python test code produced by the Writer agent.",
    )
    test_file_path: Optional[str] = Field(
        default=None,
        description="Absolute path to the saved test file on disk.",
    )

    # ── Executor output ───────────────────────────────────────────────────────
    execution_output: Optional[str] = Field(
        default=None,
        description="Combined stdout/stderr from running the test file.",
    )
    execution_success: Optional[bool] = Field(
        default=None,
        description="True if pytest exited with code 0.",
    )
    tests_passed: int = Field(default=0)
    tests_failed: int = Field(default=0)
    tests_errors: int = Field(default=0)

    # ── Reporter output ───────────────────────────────────────────────────────
    report: Optional[str] = Field(
        default=None,
        description="Markdown report summarising the full QA run.",
    )
    report_path: Optional[str] = Field(
        default=None,
        description="Path where the report was saved.",
    )

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    self_heal: bool = Field(
        default=False,
        description="When True, failed tests are passed to the Healer agent before reporting.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal errors collected across agents.",
    )
    messages: list[BaseMessage] = Field(
        default_factory=list,
        description="Running message history available to each agent.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
