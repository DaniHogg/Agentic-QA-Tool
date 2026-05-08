"""LangGraph state graph wiring all QA agents together."""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from agentic_qa.state import QAState
from agentic_qa.agents.orchestrator import orchestrate
from agentic_qa.agents.planner import plan
from agentic_qa.agents.writer import write
from agentic_qa.agents.executor import execute
from agentic_qa.agents.reporter import report


def build_graph() -> StateGraph:
    """Construct and compile the QA agent pipeline graph."""

    builder = StateGraph(QAState)

    # Register nodes
    builder.add_node("orchestrator", orchestrate)
    builder.add_node("planner", plan)
    builder.add_node("writer", write)
    builder.add_node("executor", execute)
    builder.add_node("reporter", report)

    # Entry point
    builder.set_entry_point("orchestrator")

    # Linear pipeline — every node flows to the next
    builder.add_edge("orchestrator", "planner")
    builder.add_edge("planner", "writer")
    builder.add_edge("writer", "executor")
    builder.add_edge("executor", "reporter")
    builder.add_edge("reporter", END)

    return builder.compile()


# Singleton graph instance
graph = build_graph()
