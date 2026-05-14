"""Persist and recall lightweight run context for the Streamlit UI."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CURRENT_MEMORY_SCHEMA_VERSION = 1


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


def _knowledge_dir(reports_dir: Path) -> Path:
    return reports_dir / "knowledge"


def _db_path(reports_dir: Path) -> Path:
    return _knowledge_dir(reports_dir) / "memory.db"


def init_memory_store(reports_dir: Path) -> Path:
    """Ensure phase-1 project memory database and tables exist."""
    db_path = _db_path(reports_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
                _create_phase1_tables(conn)
                _ensure_schema_version(conn)

    return db_path


def get_memory_schema_version(reports_dir: Path) -> int:
        """Return active memory schema version for diagnostics and migrations."""
        db_path = init_memory_store(reports_dir)
        with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                        "SELECT value FROM schema_meta WHERE key = 'memory_schema_version'"
                ).fetchone()
                if not row:
                        return CURRENT_MEMORY_SCHEMA_VERSION
                return int(row[0])


def _create_phase1_tables(conn: sqlite3.Connection) -> None:
        """Create all phase-1 memory tables if they do not already exist."""
        conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    target TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    model_provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tests_passed INTEGER DEFAULT 0,
                    tests_failed INTEGER DEFAULT 0,
                    tests_errors INTEGER DEFAULT 0,
                    report_path TEXT,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );

                CREATE TABLE IF NOT EXISTS plan_cases (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body_markdown TEXT NOT NULL,
                    approved INTEGER NOT NULL,
                    feedback_reason TEXT DEFAULT '',
                    edited_body_markdown TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS test_files (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    description TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    endpoint_signature TEXT DEFAULT '',
                    approved INTEGER NOT NULL,
                    feedback_reason TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS run_feedback (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    defect_quality TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    artifact_ref_id TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    quality_score REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES projects(id)
                );
                """
        )


def _ensure_schema_version(conn: sqlite3.Connection) -> None:
        """Track and migrate schema version using schema_meta table."""
        now = now_iso()
        conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
        )

        row = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'memory_schema_version'"
        ).fetchone()

        if not row:
                conn.execute(
                        """
                        INSERT INTO schema_meta (key, value, updated_at)
                        VALUES ('memory_schema_version', ?, ?)
                        """,
                        (str(CURRENT_MEMORY_SCHEMA_VERSION), now),
                )
                return

        current = int(row[0])
        if current > CURRENT_MEMORY_SCHEMA_VERSION:
                raise RuntimeError(
                        f"memory.db schema version {current} is newer than supported "
                        f"{CURRENT_MEMORY_SCHEMA_VERSION}"
                )

        if current < CURRENT_MEMORY_SCHEMA_VERSION:
                _apply_memory_migrations(conn, current, CURRENT_MEMORY_SCHEMA_VERSION)
                conn.execute(
                        """
                        UPDATE schema_meta
                        SET value = ?, updated_at = ?
                        WHERE key = 'memory_schema_version'
                        """,
                        (str(CURRENT_MEMORY_SCHEMA_VERSION), now),
                )


def _apply_memory_migrations(conn: sqlite3.Connection, from_version: int, to_version: int) -> None:
        """Apply forward-only schema migrations between known versions."""
        # No structural migrations yet. Placeholder is intentional for phase-2+ evolution.
        for _version in range(from_version + 1, to_version + 1):
                pass


def _project_id(project_name: str) -> str:
    slug = "-".join(project_name.strip().lower().split())
    return slug or "default"


def _ensure_project(conn: sqlite3.Connection, project_name: str, now: str) -> str:
    project_id = _project_id(project_name)
    conn.execute(
        """
        INSERT INTO projects (id, name, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          name = excluded.name,
          updated_at = excluded.updated_at
        """,
        (project_id, project_name, now, now),
    )
    return project_id


def start_memory_run(
    reports_dir: Path,
    *,
    project_name: str,
    target: str,
    target_type: str,
    strategy: str,
    model_provider: str,
    model_name: str,
) -> str:
    """Create a phase-1 run record and return run_id."""
    db_path = init_memory_store(reports_dir)
    run_id = uuid.uuid4().hex
    now = now_iso()

    with sqlite3.connect(db_path) as conn:
        project_id = _ensure_project(conn, project_name, now)
        conn.execute(
            """
            INSERT INTO runs (
              id, project_id, started_at, target, target_type, strategy,
              model_provider, model_name, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                project_id,
                now,
                target,
                target_type,
                strategy,
                model_provider,
                model_name,
                "pending",
            ),
        )

    return run_id


def finalize_memory_run(
    reports_dir: Path,
    *,
    run_id: str,
    status: str,
    tests_passed: int,
    tests_failed: int,
    tests_errors: int,
    report_path: str | None,
) -> None:
    """Mark run complete with outcome summary."""
    db_path = init_memory_store(reports_dir)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE runs
            SET finished_at = ?,
                status = ?,
                tests_passed = ?,
                tests_failed = ?,
                tests_errors = ?,
                report_path = ?
            WHERE id = ?
            """,
            (
                now_iso(),
                status,
                tests_passed,
                tests_failed,
                tests_errors,
                report_path,
                run_id,
            ),
        )


def save_plan_case_feedback(
    reports_dir: Path,
    *,
    run_id: str,
    case_id: str,
    title: str,
    body_markdown: str,
    approved: bool,
    reason: str = "",
    edited_body_markdown: str | None = None,
) -> None:
    """Persist per-case user decision for plan review stage."""
    db_path = init_memory_store(reports_dir)
    plan_case_id = f"{run_id}_{case_id.lower()}"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO plan_cases (
              id, run_id, case_id, title, body_markdown, approved,
              feedback_reason, edited_body_markdown, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              title = excluded.title,
              body_markdown = excluded.body_markdown,
              approved = excluded.approved,
              feedback_reason = excluded.feedback_reason,
              edited_body_markdown = excluded.edited_body_markdown
            """,
            (
                plan_case_id,
                run_id,
                case_id,
                title,
                body_markdown,
                1 if approved else 0,
                reason,
                edited_body_markdown or "",
                now_iso(),
            ),
        )


def save_test_file_feedback(
    reports_dir: Path,
    *,
    run_id: str,
    file_path: str,
    description: str,
    approved: bool,
    reason: str = "",
    endpoint_signature: str = "",
) -> None:
    """Persist per-file user decision for test review stage."""
    db_path = init_memory_store(reports_dir)
    path_obj = Path(file_path)
    content = ""
    try:
        content = path_obj.read_text(encoding="utf-8")
    except OSError:
        content = ""
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    row_id = f"{run_id}_{path_obj.name}"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO test_files (
              id, run_id, file_name, file_path, description,
              content_hash, endpoint_signature, approved, feedback_reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              description = excluded.description,
              content_hash = excluded.content_hash,
              endpoint_signature = excluded.endpoint_signature,
              approved = excluded.approved,
              feedback_reason = excluded.feedback_reason
            """,
            (
                row_id,
                run_id,
                path_obj.name,
                file_path,
                description,
                content_hash,
                endpoint_signature,
                1 if approved else 0,
                reason,
                now_iso(),
            ),
        )


def save_run_feedback(
    reports_dir: Path,
    *,
    run_id: str,
    verdict: str,
    defect_quality: str,
    notes: str = "",
) -> None:
    """Persist post-run reviewer feedback for phase-1 memory."""
    db_path = init_memory_store(reports_dir)
    feedback_id = uuid.uuid4().hex
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO run_feedback (id, run_id, verdict, defect_quality, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (feedback_id, run_id, verdict, defect_quality, notes, now_iso()),
        )


def list_memory_runs(
    reports_dir: Path,
    *,
    project_name: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List stored runs from phase-1 memory store, optionally filtered by project."""
    db_path = _db_path(reports_dir)
    if not db_path.exists():
        return []

    query = (
        """
        SELECT r.id, p.name AS project_name, r.started_at, r.finished_at,
               r.target, r.target_type, r.strategy, r.model_provider, r.model_name,
               r.status, r.tests_passed, r.tests_failed, r.tests_errors, r.report_path
        FROM runs r
        JOIN projects p ON p.id = r.project_id
        """
    )
    params: list[Any] = []
    if project_name:
        query += " WHERE p.name = ?"
        params.append(project_name)
    query += " ORDER BY r.started_at DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def load_plan_case_feedback(reports_dir: Path, run_id: str) -> list[dict[str, Any]]:
    """Load persisted plan-case decisions for one run."""
    db_path = _db_path(reports_dir)
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, run_id, case_id, title, approved, feedback_reason, edited_body_markdown, created_at
            FROM plan_cases
            WHERE run_id = ?
            ORDER BY case_id
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def load_test_file_feedback(reports_dir: Path, run_id: str) -> list[dict[str, Any]]:
    """Load persisted test-file decisions for one run."""
    db_path = _db_path(reports_dir)
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, run_id, file_name, file_path, description,
                   content_hash, endpoint_signature, approved, feedback_reason, created_at
            FROM test_files
            WHERE run_id = ?
            ORDER BY file_name
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def load_run_feedback(reports_dir: Path, run_id: str) -> list[dict[str, Any]]:
    """Load post-run reviewer feedback entries for one run."""
    db_path = _db_path(reports_dir)
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, run_id, verdict, defect_quality, notes, created_at
            FROM run_feedback
            WHERE run_id = ?
            ORDER BY created_at DESC
            """,
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]


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
