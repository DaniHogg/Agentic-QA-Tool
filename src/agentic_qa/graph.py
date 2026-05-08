"""LangGraph state graph wiring all QA agents together."""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from agentic_qa.state import QAState
from agentic_qa.agents.orchestrator import orchestrate
from agentic_qa.agents.planner import plan
from agentic_qa.agents.writer import write
from agentic_qa.agents.executor import execute
from agentic_qa.agents.healer import heal
from agentic_qa.agents.reporter import report


def _should_heal(state: QAState) -> str:
    """Conditional edge: route to healer if self_heal is enabled and tests failed."""
    if getattr(state, "self_heal", False) and not state.execution_success:
        return "healer"
    return "reporter"


def build_graph(self_heal: bool = False) -> StateGraph:
    """Construct and compile the QA agent pipeline graph.

    Args:
        self_heal: When True, a failed Executor run routes through the Healer
                   agent before re-running and then reporting.
    """

    builder = StateGraph(QAState)

    # Register nodes
    builder.add_node("orchestrator", orchestrate)
    builder.add_node("planner", plan)
    builder.add_node("writer", write)
    builder.add_node("executor", execute)
    builder.add_node("healer", heal)
    builder.add_node("executor2", execute)   # second run after healing
    builder.add_node("reporter", report)

    # Entry point
    builder.set_entry_point("orchestrator")

    builder.add_edge("orchestrator", "planner")
    builder.add_edge("planner", "writer")
    builder.add_edge("writer", "executor")

    if self_heal:
        # After first executor run: heal on failure, otherwise report
        builder.add_conditional_edges(
            "executor",
            _should_heal,
            {"healer": "healer", "reporter": "reporter"},
        )
        builder.add_edge("healer", "executor2")
        builder.add_edge("executor2", "reporter")
    else:
        builder.add_edge("executor", "reporter")

    builder.add_edge("reporter", END)

    return builder.compile()


# Default graph (no self-heal) — used by the CLI lazily
_graph_cache: dict[bool, StateGraph] = {}


def get_graph(self_heal: bool = False) -> StateGraph:
    """Return a compiled graph, building it once per (self_heal) combination."""
    if self_heal not in _graph_cache:
        _graph_cache[self_heal] = build_graph(self_heal=self_heal)
    return _graph_cache[self_heal]


# Backwards-compatible default instance
graph = get_graph(self_heal=False)
