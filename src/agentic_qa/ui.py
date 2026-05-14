"""Streamlit UI for Agentic QA.

This wraps the existing CLI flow so the core engine stays in one place.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

from agentic_qa.context_store import (
    RunContext,
    append_run,
    compose_context_snippet,
    finalize_memory_run,
    init_memory_store,
    list_memory_runs,
    load_plan_case_feedback,
    load_project_runs,
    load_recent_runs,
    load_reuse_decision_events,
    load_run_feedback,
    load_test_file_feedback,
    now_iso,
    save_plan_case_feedback,
    save_reuse_decision_events,
    save_run_feedback,
    save_test_file_feedback,
    start_memory_run,
)
from agentic_qa.state import QAState
from agentic_qa.agents.orchestrator import orchestrate
from agentic_qa.agents.planner import plan
from agentic_qa.agents.writer import write
from agentic_qa.agents.executor import execute
from agentic_qa.agents.healer import heal
from agentic_qa.agents.reporter import report
from agentic_qa.cli import _write_html_report


def launch() -> None:
    """Entry-point script for `agentic-qa-ui`."""
    script = Path(__file__).with_name("ui_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script)], check=True)


def _label(run: dict) -> str:
    return (
        f"{run.get('started_at', 'n/a')} | "
        f"{run.get('strategy', 'n/a')} | "
        f"{run.get('status', 'n/a')} | "
        f"{run.get('target', 'n/a')}"
    )


def _project_root(reports_dir: Path, project: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", project.strip().lower()).strip("-") or "default"
    return reports_dir / "projects" / safe


def _project_test_files(reports_dir: Path, project: str) -> list[Path]:
    tests_dir = _project_root(reports_dir, project) / "tests"
    if not tests_dir.exists():
        return []
    return sorted(tests_dir.glob("*.py"), reverse=True)


def _extract_test_description(path: Path) -> str:
    """Read a concise test description from module metadata/comments."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return path.stem

    desc_match = re.search(r"^Description:\s*(.+)$", text, flags=re.MULTILINE)
    if desc_match:
        return desc_match.group(1).strip()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            body = stripped.lstrip("#").strip()
            if body:
                return body
            continue
        if not stripped:
            continue
        break

    return path.stem.replace("_", " ")


def _merge_state(state: QAState, updates: dict) -> QAState:
    merged = state.model_dump()
    merged.update(updates)
    return QAState(**merged)


def _run_node(state: QAState, fn, label: str) -> QAState:
    with st.spinner(f"{label}..."):
        updates = fn(state)
    return _merge_state(state, updates or {})


def _extract_plan_cases(plan_text: str) -> tuple[str, list[dict[str, str]]]:
    """Split planner markdown into individual TC blocks for approvals."""
    lines = plan_text.splitlines()
    case_rows: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        m = re.search(r"\b(TC-\d+)\b", line, flags=re.IGNORECASE)
        if m:
            case_rows.append((idx, m.group(1).upper(), line.strip()))

    if not case_rows:
        return "", [{"id": "TC-001", "title": "Generated Test Case", "body": plan_text.strip()}]

    header = "\n".join(lines[:case_rows[0][0]]).strip()
    cases: list[dict[str, str]] = []
    for i, (start, case_id, title_line) in enumerate(case_rows):
        end = case_rows[i + 1][0] if i + 1 < len(case_rows) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        title = re.sub(r"^[\-\*\d\.\s]*", "", title_line).strip()
        cases.append({"id": case_id, "title": title or case_id, "body": body})

    return header, cases


def _build_approved_plan(header: str, cases: list[dict[str, str]], approved_ids: set[str]) -> str:
    selected = [c["body"] for c in cases if c["id"] in approved_ids]
    parts = [p for p in [header.strip(), "\n\n".join(selected).strip()] if p]
    return "\n\n".join(parts).strip()


def _preview_test_tabs(paths: list[str], key_prefix: str) -> None:
    if not paths:
        return
    labels = []
    for path in paths:
        p = Path(path)
        labels.append(f"{_extract_test_description(p)} ({p.name})")

    tabs = st.tabs(labels)
    for tab, path, label in zip(tabs, paths, labels):
        with tab:
            st.write(label)
            try:
                st.code(Path(path).read_text(encoding="utf-8")[:10000], language="python")
            except OSError:
                st.warning("Could not read selected test file.")
            st.checkbox(
                "Approve this test",
                value=st.session_state.get(f"{key_prefix}_{path}", True),
                key=f"{key_prefix}_{path}",
            )
            st.text_input(
                "Feedback reason (optional)",
                value=st.session_state.get(f"{key_prefix}_reason_{path}", ""),
                key=f"{key_prefix}_reason_{path}",
                placeholder="Why approve/reject this test?",
            )


def _reset_workflow_state() -> None:
    """Clear in-progress plan/test approval workflow state."""
    for key in list(st.session_state.keys()):
        if (
            key.startswith("approve_plan_")
            or key.startswith("approve_test_")
            or key.startswith("plan_reason_")
            or key.startswith("plan_edit_")
            or key.startswith("approve_test_reason_")
            or key.startswith("reuse_mid_")
        ):
            del st.session_state[key]

    st.session_state.workflow_stage = "idle"
    st.session_state.workflow_state = None
    st.session_state.workflow_run_id = None
    st.session_state.workflow_feedback_saved = False
    st.session_state.workflow_plan_header = ""
    st.session_state.workflow_plan_cases = []
    st.session_state.workflow_generated_files = []


def _ui_main() -> None:
    st.set_page_config(page_title="Agentic QA UI", layout="wide")
    st.title("Agentic QA UI")
    st.caption("Configure runs, approve strategy choices, view logs, and recall context from previous runs.")

    reports_dir = Path(os.getenv("REPORTS_DIR", "./reports")).resolve()
    os.environ["REPORTS_DIR"] = str(reports_dir)
    init_memory_store(reports_dir)
    projects_base = reports_dir / "projects"
    existing_projects = sorted([p.name for p in projects_base.iterdir() if p.is_dir()]) if projects_base.exists() else []

    if "workflow_stage" not in st.session_state:
        st.session_state.workflow_stage = "idle"
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = None
    if "workflow_plan_header" not in st.session_state:
        st.session_state.workflow_plan_header = ""
    if "workflow_plan_cases" not in st.session_state:
        st.session_state.workflow_plan_cases = []
    if "workflow_generated_files" not in st.session_state:
        st.session_state.workflow_generated_files = []
    if "workflow_run_id" not in st.session_state:
        st.session_state.workflow_run_id = None
    if "workflow_feedback_saved" not in st.session_state:
        st.session_state.workflow_feedback_saved = False
    if "quiet_mode" not in st.session_state:
        st.session_state.quiet_mode = False
    if "quiet_show_plan" not in st.session_state:
        st.session_state.quiet_show_plan = False
    if "quiet_show_tests" not in st.session_state:
        st.session_state.quiet_show_tests = False

    with st.sidebar:
        st.subheader("Run Context Recall")
        history = load_recent_runs(reports_dir)

        if history:
            options = {_label(row): row for row in history}
            selected_labels = st.multiselect(
                "Use notes from previous runs",
                list(options.keys()),
                default=[],
            )
            recalled_context = compose_context_snippet([options[label] for label in selected_labels])
            if recalled_context:
                st.text_area("Recalled context preview", recalled_context, height=180)
        else:
            recalled_context = ""
            st.info("No previous UI runs stored yet.")

    col1, col2 = st.columns(2)
    with col1:
        target = st.text_input("Target URL/path", value="https://jsonplaceholder.typicode.com")
        provider = st.selectbox("Provider", ["anthropic", "openai"], index=0)
        model = st.text_input("Model override (optional)", value="")
        strategy = st.selectbox("Strategy", ["smoke", "sanity", "regression", "custom"], index=0)
        max_tests = st.number_input("Max test cases", min_value=1, max_value=100, value=10)
        selected_project = st.selectbox(
            "Project",
            existing_projects + ["(new project)"],
            index=0 if existing_projects else 0,
        )
        if selected_project == "(new project)":
            project_name = st.text_input("New project name", value="default")
        else:
            project_name = selected_project

    with col2:
        quiet_mode = st.toggle(
            "Quiet Mode",
            value=st.session_state.quiet_mode,
            help="When off, Detailed mode shows plan and tests by default. When on, compact output is used.",
        )
        st.session_state.quiet_mode = quiet_mode
        run_mode = "quiet" if quiet_mode else "detailed"

        st.caption("Quiet Mode Sub-options")
        show_plan = st.checkbox(
            "Show Approved Plan In Quiet Mode",
            value=st.session_state.quiet_show_plan,
            disabled=not quiet_mode,
            help="Only applies when Quiet Mode is enabled.",
        )
        show_tests = st.checkbox(
            "Show Approved Tests In Quiet Mode",
            value=st.session_state.quiet_show_tests,
            disabled=not quiet_mode,
            help="Only applies when Quiet Mode is enabled.",
        )

        if not quiet_mode:
            show_plan = False
            show_tests = False

        st.session_state.quiet_show_plan = show_plan
        st.session_state.quiet_show_tests = show_tests

        report_format = st.selectbox("Report format", ["markdown", "html"], index=1)
        self_heal = st.checkbox("Enable self-heal", value=False)
        description_style = st.selectbox("Test description style", ["standard", "gherkin"], index=0)
        run_with_existing = st.checkbox("Run selected previous tests with new tests", value=False)
        deduplicate = st.checkbox("Reuse duplicate tests when possible", value=True)

    prior_tests = _project_test_files(reports_dir, project_name)
    selected_existing_tests: list[str] = []
    if prior_tests:
        st.markdown("### Previous Tests In Project")
        labels = {
            f"{_extract_test_description(f)} ({f.name})": str(f.resolve())
            for f in prior_tests
        }
        selected = st.multiselect(
            "Select previous tests (description)",
            list(labels.keys()),
            default=[],
        )
        selected_existing_tests = [labels[item] for item in selected]

        with st.expander("Selected test descriptions"):
            if selected:
                for item in selected:
                    st.write(f"- {item}")
            else:
                st.caption("No previous tests selected.")
    else:
        st.info("No previous tests found for this project yet.")

    notes = st.text_area(
        "Planning notes",
        value="",
        height=120,
        help="Use this to steer generated tests toward your application specifics.",
    )
    run_description = st.text_input(
        "Run description",
        value="",
        help="Optional short summary for this run (saved in project run report).",
    )

    st.markdown("### Project Run Report")
    project_runs = load_project_runs(reports_dir, project_name, limit=100)
    if project_runs:
        rows = []
        for row in project_runs:
            rows.append(
                {
                    "Date": row.get("started_at", "n/a"),
                    "Description": row.get("run_description", "") or "(no description)",
                    "Status": row.get("status", "n/a"),
                    "Exit": row.get("exit_code", "n/a"),
                    "Strategy": row.get("strategy", "n/a"),
                    "Target": row.get("target", "n/a"),
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Export helpers for portfolio evidence and review workflows.
        csv_headers = ["Date", "Description", "Status", "Exit", "Strategy", "Target"]
        csv_lines = [",".join(csv_headers)]
        for item in rows:
            csv_lines.append(
                ",".join(
                    '"' + str(item.get(h, "")).replace('"', '""') + '"'
                    for h in csv_headers
                )
            )
        csv_text = "\n".join(csv_lines)
        json_text = json.dumps(rows, indent=2, ensure_ascii=True)

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                label="Export Run Report (CSV)",
                data=csv_text,
                file_name=f"{project_name}_run_report.csv",
                mime="text/csv",
            )
        with export_col2:
            st.download_button(
                label="Export Run Report (JSON)",
                data=json_text,
                file_name=f"{project_name}_run_report.json",
                mime="application/json",
            )
    else:
        st.info("No previous runs recorded for this project yet.")

    with st.expander("Project Memory Snapshot"):
        memory_runs = list_memory_runs(reports_dir, project_name=project_name, limit=5)
        if memory_runs:
            st.dataframe(memory_runs, use_container_width=True, hide_index=True)
        else:
            st.caption("No memory runs stored yet for this project.")

    if recalled_context:
        merged_notes = (notes.strip() + "\n\n" + recalled_context).strip() if notes.strip() else recalled_context
    else:
        merged_notes = notes.strip()

    st.markdown("### Run")
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        run_clicked = st.button("Generate Test Plan", type="primary")
    with run_col2:
        reset_clicked = st.button("Reset Current Workflow")

    if reset_clicked:
        _reset_workflow_state()
        st.success("Workflow reset. You can start a new plan now.")
        st.rerun()

    if run_clicked:
        os.environ["LLM_PROVIDER"] = provider
        os.environ["MAX_TEST_CASES"] = str(int(max_tests))
        if model.strip():
            if provider == "anthropic":
                os.environ["ANTHROPIC_MODEL"] = model.strip()
            else:
                os.environ["OPENAI_MODEL"] = model.strip()

        state = QAState(
            target=target,
            self_heal=self_heal,
            test_strategy=strategy,
            planning_notes=merged_notes or None,
            project_name=project_name,
            description_style=description_style,
            run_with_existing_tests=run_with_existing,
            existing_test_files=selected_existing_tests,
            deduplicate_tests=deduplicate,
        )
        state = _run_node(state, orchestrate, "Orchestrator")
        state = _run_node(state, plan, "Planner")

        active_model = model.strip() or (
            os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
            if provider == "anthropic"
            else os.getenv("OPENAI_MODEL", "gpt-4o")
        )
        run_id = start_memory_run(
            reports_dir,
            project_name=project_name,
            target=target,
            target_type=state.target_type or "api",
            strategy=strategy,
            model_provider=provider,
            model_name=active_model,
        )

        header, cases = _extract_plan_cases(state.test_plan or "")
        for case in cases:
            key = f"approve_plan_{case['id']}"
            if key not in st.session_state:
                st.session_state[key] = True

        st.session_state.workflow_state = state.model_dump()
        st.session_state.workflow_run_id = run_id
        st.session_state.workflow_feedback_saved = False
        st.session_state.workflow_plan_header = header
        st.session_state.workflow_plan_cases = cases
        st.session_state.workflow_generated_files = []
        st.session_state.workflow_stage = "plan_review"
        st.rerun()

    if st.session_state.workflow_stage == "plan_review" and st.session_state.workflow_state:
        st.markdown("### Plan Approval")
        cases = st.session_state.workflow_plan_cases
        tabs = st.tabs([f"{c['id']}" for c in cases])
        for tab, case in zip(tabs, cases):
            with tab:
                st.markdown(f"**{case['title']}**")
                st.markdown(case["body"])
                st.checkbox(
                    "Approve this test case",
                    value=st.session_state.get(f"approve_plan_{case['id']}", True),
                    key=f"approve_plan_{case['id']}",
                )
                st.text_input(
                    "Feedback reason (optional)",
                    value=st.session_state.get(f"plan_reason_{case['id']}", ""),
                    key=f"plan_reason_{case['id']}",
                    placeholder="Why approve/reject this case?",
                )
                st.text_area(
                    "Edited case text (optional)",
                    value=st.session_state.get(f"plan_edit_{case['id']}", ""),
                    key=f"plan_edit_{case['id']}",
                    height=140,
                )

        plan_col1, plan_col2 = st.columns(2)
        with plan_col1:
            if st.button("Approve All Test Cases"):
                for case in cases:
                    st.session_state[f"approve_plan_{case['id']}"] = True
                st.rerun()
        with plan_col2:
            if st.button("Generate Tests From Approved Cases", type="primary"):
                approved = {c["id"] for c in cases if st.session_state.get(f"approve_plan_{c['id']}", False)}
                if not approved:
                    st.warning("Select at least one approved test case to continue.")
                else:
                    run_id = st.session_state.workflow_run_id
                    revised_cases = []
                    if run_id:
                        for case in cases:
                            is_approved = case["id"] in approved
                            reason = st.session_state.get(f"plan_reason_{case['id']}", "").strip()
                            edited_body = st.session_state.get(f"plan_edit_{case['id']}", "").strip()
                            selected_body = edited_body if edited_body else case["body"]
                            save_plan_case_feedback(
                                reports_dir,
                                run_id=run_id,
                                case_id=case["id"],
                                title=case["title"],
                                body_markdown=selected_body,
                                approved=is_approved,
                                reason=reason,
                                edited_body_markdown=edited_body or None,
                            )
                            revised_cases.append({**case, "body": selected_body})
                    else:
                        revised_cases = cases

                    state = QAState(**st.session_state.workflow_state)
                    approved_plan = _build_approved_plan(
                        st.session_state.workflow_plan_header,
                        revised_cases,
                        approved,
                    )
                    state = _merge_state(state, {"test_plan": approved_plan})
                    state = _run_node(state, write, "Writer")

                    generated_files = list(state.generated_test_files or [])
                    if not generated_files and state.test_file_path:
                        generated_files = [state.test_file_path]

                    for path in generated_files:
                        key = f"approve_test_{path}"
                        if key not in st.session_state:
                            st.session_state[key] = True

                    st.session_state.workflow_state = state.model_dump()
                    st.session_state.workflow_generated_files = generated_files
                    st.session_state.workflow_stage = "test_review"
                    st.rerun()

    if st.session_state.workflow_stage == "test_review" and st.session_state.workflow_state:
        st.markdown("### Test File Approval")
        state = QAState(**st.session_state.workflow_state)
        generated_files = st.session_state.workflow_generated_files

        if state.reuse_decisions:
            summary_rows = []
            for decision in state.reuse_decisions:
                summary_rows.append(
                    {
                        "Case": decision.get("case_id", "n/a"),
                        "Score": decision.get("score", 0.0),
                        "Band": decision.get("threshold_band", "n/a"),
                        "Action": decision.get("action", "n/a"),
                        "Coverage": decision.get("coverage_status", "covered"),
                        "Needs Approval": bool(decision.get("requires_user_approval", False)),
                        "Reason": decision.get("reason", ""),
                    }
                )
            st.markdown("#### Reuse Scoring")
            st.dataframe(summary_rows, use_container_width=True, hide_index=True)

            if state.coverage_guard_generated:
                st.info(
                    "Coverage guard generated fallback tests for: "
                    + ", ".join(state.coverage_guard_generated)
                )
            if state.coverage_gaps:
                st.error(
                    "Unresolved coverage gaps remain: "
                    + ", ".join(state.coverage_gaps)
                )

            mid_band = [
                d for d in state.reuse_decisions
                if d.get("threshold_band") == "mid"
                and d.get("candidate_file")
                and d.get("generated_file")
            ]
            if mid_band:
                with st.expander("Mid-Band Reuse Approval"):
                    st.caption("Choose whether to reuse the existing candidate for mid-confidence matches.")
                    for decision in mid_band:
                        case_id = str(decision.get("case_id", "case"))
                        generated_file = str(decision.get("generated_file", ""))
                        key = f"reuse_mid_{case_id}_{generated_file}"
                        st.checkbox(
                            (
                                f"Use existing candidate for {case_id} "
                                f"(score {float(decision.get('score', 0.0)):.2f})"
                            ),
                            value=st.session_state.get(key, False),
                            key=key,
                        )

        _preview_test_tabs(generated_files, "approve_test")

        test_col1, test_col2 = st.columns(2)
        with test_col1:
            if st.button("Approve All Tests"):
                for path in generated_files:
                    st.session_state[f"approve_test_{path}"] = True
                st.rerun()
        with test_col2:
            if st.button("Execute Approved Tests", type="primary"):
                approved_files = [p for p in generated_files if st.session_state.get(f"approve_test_{p}", False)]
                if not approved_files:
                    st.warning("Approve at least one generated test file to run execution.")
                else:
                    selected_reuse_candidates: set[str] = set()
                    finalized_decisions: list[dict] = []
                    for decision in state.reuse_decisions:
                        if decision.get("threshold_band") != "mid":
                            continue
                        candidate = decision.get("candidate_file")
                        generated = decision.get("generated_file")
                        if not candidate or not generated:
                            continue
                        key = f"reuse_mid_{decision.get('case_id', 'case')}_{generated}"
                        if st.session_state.get(key, False):
                            selected_reuse_candidates.add(str(candidate))

                    effective_files: list[str] = []
                    for path in approved_files:
                        replacement = path
                        for decision in state.reuse_decisions:
                            if decision.get("threshold_band") != "mid":
                                continue
                            generated = str(decision.get("generated_file", ""))
                            candidate = str(decision.get("candidate_file", ""))
                            key = f"reuse_mid_{decision.get('case_id', 'case')}_{generated}"
                            if path == generated and st.session_state.get(key, False) and candidate:
                                replacement = candidate
                                break
                        if replacement not in effective_files:
                            effective_files.append(replacement)

                    for decision in state.reuse_decisions:
                        decision_copy = dict(decision)
                        decision_copy["approved_override"] = False
                        if decision.get("threshold_band") == "mid":
                            generated = str(decision.get("generated_file", ""))
                            candidate = str(decision.get("candidate_file", ""))
                            key = f"reuse_mid_{decision.get('case_id', 'case')}_{generated}"
                            if st.session_state.get(key, False) and candidate:
                                decision_copy["approved_override"] = True
                                decision_copy["action"] = "reused_mid_approved"
                                decision_copy["generated_file"] = candidate
                            else:
                                decision_copy["action"] = "generated_mid_declined"
                        finalized_decisions.append(decision_copy)

                    run_id = st.session_state.workflow_run_id
                    if run_id:
                        review_targets = set(generated_files) | selected_reuse_candidates
                        for test_path in sorted(review_targets):
                            reason = st.session_state.get(f"approve_test_reason_{test_path}", "").strip()
                            approved_flag = test_path in effective_files
                            if test_path in selected_reuse_candidates and approved_flag and not reason:
                                reason = "Selected as mid-band reuse candidate"
                            save_test_file_feedback(
                                reports_dir,
                                run_id=run_id,
                                file_path=test_path,
                                description=_extract_test_description(Path(test_path)),
                                approved=approved_flag,
                                reason=reason,
                            )

                    state = QAState(**st.session_state.workflow_state)
                    state = _merge_state(
                        state,
                        {
                            "generated_test_files": effective_files,
                            "test_file_path": effective_files[0],
                        },
                    )

                    state = _run_node(state, execute, "Executor")
                    if state.self_heal and not state.execution_success:
                        state = _run_node(state, heal, "Healer")
                        state = _run_node(state, execute, "Executor Retry")
                    state = _run_node(state, report, "Reporter")

                    if report_format == "html" and state.report and state.report_path:
                        html_report_path = _write_html_report(state)
                        state = _merge_state(state, {"report_path": html_report_path})

                    status = "passed" if state.execution_success else "failed"
                    append_run(
                        reports_dir,
                        RunContext(
                            started_at=now_iso(),
                            target=target,
                            provider=provider,
                            strategy=strategy,
                            max_tests=int(max_tests),
                            planning_notes=merged_notes,
                            mode=run_mode,
                            self_heal=self_heal,
                            project_name=project_name,
                            description_style=description_style,
                            selected_existing_tests=selected_existing_tests,
                            run_description=run_description.strip(),
                            status=status,
                            exit_code=0 if state.execution_success else 1,
                            report_path=state.report_path,
                            test_file_path=state.test_file_path,
                        ),
                    )

                    if run_id:
                        finalize_memory_run(
                            reports_dir,
                            run_id=run_id,
                            status=status,
                            tests_passed=state.tests_passed,
                            tests_failed=state.tests_failed,
                            tests_errors=state.tests_errors,
                            report_path=state.report_path,
                        )
                        save_reuse_decision_events(
                            reports_dir,
                            run_id=run_id,
                            decisions=finalized_decisions,
                        )

                    state = _merge_state(state, {"reuse_decisions": finalized_decisions})

                    st.session_state.workflow_state = state.model_dump()
                    st.session_state.workflow_stage = "completed"
                    st.rerun()

    if st.session_state.workflow_stage == "completed" and st.session_state.workflow_state:
        state = QAState(**st.session_state.workflow_state)
        run_id = st.session_state.workflow_run_id
        st.markdown("### Outcome")
        if state.execution_success:
            st.success("Run completed successfully.")
        else:
            st.error("Run failed.")

        if state.duplicate_reused:
            st.info(
                "Duplicate test reuse enabled: existing test was reused instead of writing a new duplicate."
                + (f" Source: {state.duplicate_source_path}" if state.duplicate_source_path else "")
            )

        if state.reuse_decisions:
            with st.expander("Reuse Decision Summary"):
                st.dataframe(state.reuse_decisions, use_container_width=True, hide_index=True)

        if state.coverage_guard_generated:
            st.caption(
                "Coverage guard generated fallback tests for: "
                + ", ".join(state.coverage_guard_generated)
            )
        if state.coverage_gaps:
            st.warning(
                "Coverage gaps still unresolved: "
                + ", ".join(state.coverage_gaps)
            )

        if run_mode == "detailed" or show_plan:
            if state.test_plan:
                st.markdown("### Approved Plan")
                st.markdown(state.test_plan)

        if run_mode == "detailed" or show_tests:
            files = list(state.generated_test_files or [])
            if not files and state.test_file_path:
                files = [state.test_file_path]
            if files:
                st.markdown("### Approved Test Files")
                preview_tabs = st.tabs([Path(p).name for p in files])
                for tab, path in zip(preview_tabs, files):
                    with tab:
                        try:
                            st.code(Path(path).read_text(encoding="utf-8")[:10000], language="python")
                        except OSError:
                            st.warning("Could not read generated test file.")

        if state.execution_output and run_mode == "detailed":
            with st.expander("Execution Output"):
                st.code(state.execution_output[:15000], language="text")

        if state.report_path:
            st.write(f"Report: {state.report_path}")
            try:
                report_text = Path(state.report_path).read_text(encoding="utf-8")
                if state.report_path.endswith(".md"):
                    st.markdown(report_text)
                else:
                    st.code(report_text[:10000], language="html")
            except OSError:
                st.warning("Report file path was reported but could not be read.")

        st.markdown("### Post-Run Feedback")
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            verdict = st.selectbox(
                "Overall usefulness",
                ["useful", "not_useful"],
                index=0,
                key="postrun_verdict",
            )
        with feedback_col2:
            defect_quality = st.selectbox(
                "Failure quality",
                ["true_defect", "test_issue", "mixed"],
                index=0,
                key="postrun_defect_quality",
            )
        feedback_notes = st.text_area(
            "Post-run notes (optional)",
            value="",
            key="postrun_notes",
            height=120,
        )

        if st.button("Save Post-Run Feedback"):
            if run_id:
                save_run_feedback(
                    reports_dir,
                    run_id=run_id,
                    verdict=verdict,
                    defect_quality=defect_quality,
                    notes=feedback_notes.strip(),
                )
                st.session_state.workflow_feedback_saved = True
                st.success("Post-run feedback saved to project memory.")
            else:
                st.warning("No active memory run ID found for this workflow.")

        if run_id:
            saved_plan = load_plan_case_feedback(reports_dir, run_id)
            saved_tests = load_test_file_feedback(reports_dir, run_id)
            saved_feedback = load_run_feedback(reports_dir, run_id)
            saved_reuse = load_reuse_decision_events(reports_dir, run_id)
            st.caption(
                "Saved memory records: "
                f"{len(saved_plan)} plan decisions, {len(saved_tests)} test decisions, "
                f"{len(saved_feedback)} run feedback item(s), {len(saved_reuse)} reuse decision event(s)."
            )


if __name__ == "__main__":
    _ui_main()
