"""Persist and recall lightweight run context for the Streamlit UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunContext:
    """Minimal metadata saved per UI run for future recall."""

    started_at: str
    target: str
    provider: str
    strategy: str
    max_tests: int
    planning_notes: str
    mode: str
    self_heal: bool
    status: str
    exit_code: int
    project_name: str = "default"
    description_style: str = "standard"
    selected_existing_tests: list[str] | None = None
    run_description: str = ""
    report_path: str | None = None
    test_file_path: str | None = None


def _history_path(reports_dir: Path) -> Path:
    return reports_dir / "ui_run_history.jsonl"


def load_recent_runs(reports_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Load recent run records from jsonl history (newest first)."""
    path = _history_path(reports_dir)
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    rows.reverse()
    return rows[:limit]


def load_project_runs(reports_dir: Path, project_name: str, limit: int = 100) -> list[dict[str, Any]]:
    """Load run history filtered by project (newest first)."""
    recent = load_recent_runs(reports_dir, limit=10000)
    filtered = [row for row in recent if row.get("project_name", "default") == project_name]
    return filtered[:limit]


def append_run(reports_dir: Path, run: RunContext) -> None:
    """Append one run record to history jsonl."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = _history_path(reports_dir)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(run.__dict__, ensure_ascii=True) + "\n")


def compose_context_snippet(selected_runs: list[dict[str, Any]]) -> str:
    """Build planner notes from selected historical runs."""
    if not selected_runs:
        return ""

    lines: list[str] = ["Previous run context:"]
    for run in selected_runs:
        lines.append(
            "- "
            f"[{run.get('started_at', 'n/a')}] "
            f"target={run.get('target', 'n/a')} "
            f"strategy={run.get('strategy', 'n/a')} "
            f"status={run.get('status', 'n/a')} "
            f"exit={run.get('exit_code', 'n/a')} "
            f"description={run.get('run_description', '').strip() or 'n/a'}"
        )
    return "\n".join(lines)


def now_iso() -> str:
    """UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()
