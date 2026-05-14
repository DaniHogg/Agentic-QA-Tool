"""Agentic QA CLI — entry point for the multi-agent QA pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime, timezone, timedelta

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.text import Text

load_dotenv(override=True)

app = typer.Typer(
    name="agentic-qa",
    help="Autonomous QA agent — plans, writes, runs, and reports tests for APIs, UIs, and Python codebases.",
    add_completion=False,
)

console = Console()


def _interactive_setup(
    *,
    target: Optional[str],
    strategy: str,
    max_tests: int,
    self_heal: bool,
    planning_notes: str,
    format: str,
    project: str,
    description_style: str,
) -> tuple[str, str, int, bool, str, str, str, str]:
    """Prompt user for run configuration before starting pipeline."""

    console.print(Rule("[bold]Run Setup[/bold]"))

    resolved_target = target or typer.prompt("Target URL/path")

    strategy_options = "smoke/sanity/regression/custom"
    chosen_strategy = typer.prompt(
        f"Test strategy ({strategy_options})",
        default=(strategy or "smoke").lower(),
    ).strip().lower()

    suggested = {"smoke": 5, "sanity": 8, "regression": 20, "custom": max_tests}
    chosen_max = typer.prompt(
        "Max test cases",
        default=suggested.get(chosen_strategy, max_tests),
        type=int,
    )

    notes = planning_notes.strip()
    if not notes:
        notes = typer.prompt(
            "Planner notes (optional)",
            default="",
            show_default=False,
        ).strip()

    chosen_self_heal = typer.confirm("Enable self-heal retry on failures?", default=self_heal)
    chosen_format = typer.prompt("Report format (markdown/html)", default=format).strip().lower()
    chosen_project = typer.prompt("Project bucket name", default=project).strip() or "default"
    chosen_desc_style = typer.prompt(
        "Test description style (standard/gherkin)",
        default=description_style,
    ).strip().lower()

    return (
        resolved_target,
        chosen_strategy,
        chosen_max,
        chosen_self_heal,
        notes,
        chosen_format,
        chosen_project,
        chosen_desc_style,
    )


def _apply_node(state, fn: Callable, label: str):
    """Execute one agent node and merge its output into QAState."""
    from agentic_qa.state import QAState

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"[cyan]{label}...", total=None)
        updates = fn(state)

    merged = state.model_dump()
    if updates:
        merged.update(updates)
    return QAState(**merged)


def _run_node(state, fn: Callable, label: str, detailed: bool):
    """Run a node with optional user-visible step logging."""
    if detailed:
        console.print(Rule(f"[bold]{label}[/bold]"))
        console.print(f"[cyan]Starting:[/cyan] {label}")

    updated = _apply_node(state, fn, label)

    if detailed:
        console.print(f"[green]Completed:[/green] {label}")

    return updated


def _render_intermediate_outputs(state, *, detailed: bool, show_plan: bool, show_tests: bool) -> None:
    """Display plan/code/results based on selected verbosity mode."""
    if detailed:
        if state.test_plan:
            console.print(Rule("[bold]Test Plan[/bold]"))
            console.print(Markdown(state.test_plan))

        if state.generated_tests:
            console.print(Rule("[bold]Generated Tests[/bold]"))
            console.print(state.generated_tests)

        if state.execution_output:
            console.print(Rule("[bold]Execution Output[/bold]"))
            console.print(state.execution_output)
        return

    if show_plan and state.test_plan:
        console.print(Rule("[bold]Test Plan[/bold]"))
        console.print(Markdown(state.test_plan))

    if show_tests and state.generated_tests:
        console.print(Rule("[bold]Generated Tests[/bold]"))
        console.print(state.generated_tests)


def _plan_approval_loop(state):
    """Show plan, collect approval, and apply requested edits before execution."""
    from agentic_qa.state import QAState

    if not state.test_plan:
        return state

    while True:
        choice = typer.prompt(
            "Plan action: approve (a), request changes (r), add cases (n), cancel (c)",
            default="a",
        ).strip().lower()

        if choice in ("a", "approve"):
            console.print("[green]Plan approved. Continuing to test generation.[/green]")
            return state

        if choice in ("c", "cancel"):
            console.print("[yellow]Run cancelled before execution.[/yellow]")
            raise typer.Exit(code=0)

        if choice in ("r", "request", "request changes"):
            request = typer.prompt("Describe the changes you want")
            revised = _revise_test_plan(
                current_plan=state.test_plan,
                request=request,
                mode="edit",
                strategy=state.test_strategy,
            )
            state = QAState(**{**state.model_dump(), "test_plan": revised})
            console.print(Rule("[bold]Revised Test Plan[/bold]"))
            console.print(Markdown(revised))
            continue

        if choice in ("n", "add", "add cases"):
            request = typer.prompt("List additional test cases or coverage requests")
            revised = _revise_test_plan(
                current_plan=state.test_plan,
                request=request,
                mode="add",
                strategy=state.test_strategy,
            )
            state = QAState(**{**state.model_dump(), "test_plan": revised})
            console.print(Rule("[bold]Updated Test Plan[/bold]"))
            console.print(Markdown(revised))
            continue

        console.print("[yellow]Invalid choice. Enter a, r, n, or c.[/yellow]")


def _revise_test_plan(*, current_plan: str, request: str, mode: str, strategy: str) -> str:
    """Use the configured LLM to revise the plan per user feedback."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from agentic_qa.llm import get_chat_model

    llm = get_chat_model(temperature=0.2)

    system = (
        "You are a QA Planner revising an existing Markdown test plan. "
        "Return only the full revised Markdown plan. Keep existing test case IDs where possible. "
        "Ensure the result remains actionable for pytest test generation."
    )
    mode_text = "Add new test cases and keep the existing structure" if mode == "add" else "Apply edits to existing test cases"
    user = (
        f"Strategy: {strategy}\n"
        f"Revision mode: {mode_text}\n"
        f"User request: {request}\n\n"
        f"Current plan:\n{current_plan}"
    )

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content.strip()


@app.command("run")
def run(
    target: Optional[str] = typer.Argument(
        None,
        help="URL (API or web) or local file/directory path to test.",
    ),
    provider: str = typer.Option(
        None,
        "--provider", "-p",
        help="LLM provider: 'openai' or 'anthropic'.",
        envvar="LLM_PROVIDER",
    ),
    model: str = typer.Option(
        None,
        "--model", "-m",
        help="Model name for the chosen provider.",
    ),
    max_tests: int = typer.Option(
        10,
        "--max-tests", "-n",
        help="Target number of test cases for the Planner to generate.",
        envvar="MAX_TEST_CASES",
    ),
    strategy: str = typer.Option(
        "smoke",
        "--strategy", "-s",
        help="Test strategy: smoke, sanity, regression, or custom.",
    ),
    planning_notes: str = typer.Option(
        "",
        "--planning-notes",
        help="Extra guidance for the Planner (scope, risks, must-have cases).",
    ),
    project: str = typer.Option(
        "default",
        "--project",
        help="Project bucket name for storing generated tests.",
    ),
    description_style: str = typer.Option(
        "standard",
        "--description-style",
        help="Generated test description style: standard or gherkin.",
    ),
    run_with_existing_tests: bool = typer.Option(
        False,
        "--run-with-existing-tests",
        help="Run selected existing tests together with newly generated tests.",
    ),
    existing_test: list[str] = typer.Option(
        None,
        "--existing-test",
        help="Path to an existing test file to include (repeat flag for multiple).",
    ),
    deduplicate_tests: bool = typer.Option(
        True,
        "--deduplicate-tests/--no-deduplicate-tests",
        help="Reuse existing tests in the project folder when duplicate content is detected.",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Use interactive setup and plan approval before execution.",
    ),
    reports_dir: str = typer.Option(
        "./reports",
        "--reports-dir", "-o",
        help="Directory where reports and generated tests are saved.",
        envvar="REPORTS_DIR",
    ),
    show_tests: bool = typer.Option(
        False,
        "--show-tests",
        help="Print the generated test code to stdout.",
    ),
    show_plan: bool = typer.Option(
        False,
        "--show-plan",
        help="Print the test plan to stdout.",
    ),
    self_heal: bool = typer.Option(
        False,
        "--self-heal",
        help="If tests fail, automatically rewrite and re-run them once using the Healer agent.",
    ),
    format: str = typer.Option(
        "markdown",
        "--format", "-f",
        help="Report output format: 'markdown' (default) or 'html'.",
    ),
    mode: str = typer.Option(
        "detailed",
        "--mode",
        help="Run output mode: 'detailed' (default) or 'quiet'.",
    ),
) -> None:
    """Run the full Agentic QA pipeline against TARGET."""

    if interactive:
        target, strategy, max_tests, self_heal, planning_notes, format, project, description_style = _interactive_setup(
            target=target,
            strategy=strategy,
            max_tests=max_tests,
            self_heal=self_heal,
            planning_notes=planning_notes,
            format=format,
            project=project,
            description_style=description_style,
        )

    if not target:
        console.print("[red]Error:[/red] TARGET is required. Pass it or use --interactive.")
        raise typer.Exit(code=1)

    strategy = strategy.strip().lower()
    if strategy not in ("smoke", "sanity", "regression", "custom"):
        console.print("[red]Error:[/red] --strategy must be smoke, sanity, regression, or custom.")
        raise typer.Exit(code=1)

    description_style = description_style.strip().lower()
    if description_style not in ("standard", "gherkin"):
        console.print("[red]Error:[/red] --description-style must be 'standard' or 'gherkin'.")
        raise typer.Exit(code=1)

    mode = mode.strip().lower()
    if mode not in ("detailed", "quiet"):
        console.print("[red]Error:[/red] --mode must be 'detailed' or 'quiet'.")
        raise typer.Exit(code=1)
    detailed_mode = mode == "detailed"

    active_provider = (provider or os.getenv("LLM_PROVIDER", "openai")).strip().lower()
    _validate_env(active_provider)

    if format not in ("markdown", "html"):
        console.print("[red]Error:[/red] --format must be 'markdown' or 'html'.")
        raise typer.Exit(code=1)
    if active_provider not in ("openai", "anthropic"):
        console.print("[red]Error:[/red] --provider must be 'openai' or 'anthropic'.")
        raise typer.Exit(code=1)

    # Override env vars from CLI flags
    os.environ["LLM_PROVIDER"] = active_provider
    if model:
        if active_provider == "anthropic":
            os.environ["ANTHROPIC_MODEL"] = model
        else:
            os.environ["OPENAI_MODEL"] = model
    os.environ["TEST_STRATEGY"] = strategy
    os.environ["MAX_TEST_CASES"] = str(max_tests)
    os.environ["REPORTS_DIR"] = reports_dir

    # Import here so env vars are set before agent initialisation
    from agentic_qa.state import QAState
    from agentic_qa.agents.orchestrator import orchestrate
    from agentic_qa.agents.planner import plan
    from agentic_qa.agents.writer import write
    from agentic_qa.agents.executor import execute
    from agentic_qa.agents.healer import heal
    from agentic_qa.agents.reporter import report

    heal_label = "  [yellow]+self-heal[/yellow]" if self_heal else ""
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold cyan]Agentic QA[/bold cyan]  [dim]|[/dim]  "
            f"[white]{target}[/white]  [dim]|[/dim]  "
            f"project: [green]{project}[/green]  [dim]|[/dim]  "
            f"mode: [green]{mode}[/green]  [dim]|[/dim]  "
            f"strategy: [green]{strategy}[/green]  [dim]|[/dim]  "
            f"provider: [green]{active_provider}[/green]  [dim]|[/dim]  "
            f"model: [green]{_active_model_name(active_provider)}[/green]"
            f"{heal_label}"
        ),
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()

    state = QAState(
        target=target,
        self_heal=self_heal,
        test_strategy=strategy,
        planning_notes=planning_notes or None,
        project_name=project,
        description_style=description_style,
        run_with_existing_tests=run_with_existing_tests,
        existing_test_files=existing_test or [],
        deduplicate_tests=deduplicate_tests,
    )

    # Phase 1: Plan and review
    state = _run_node(state, orchestrate, "Orchestrator", detailed_mode)
    state = _run_node(state, plan, "Planner", detailed_mode)

    if detailed_mode and state.test_plan:
        console.print(Rule("[bold]Proposed Test Plan[/bold]"))
        console.print(Markdown(state.test_plan))

    if interactive:
        state = _plan_approval_loop(state)
    elif show_plan and state.test_plan and not detailed_mode:
        console.print(Rule("[bold]Approved Test Plan[/bold]"))
        console.print(Markdown(state.test_plan))

    # Phase 2: Generate and execute only after approval
    state = _run_node(state, write, "Writer", detailed_mode)
    state = _run_node(state, execute, "Executor", detailed_mode)

    if self_heal and not state.execution_success:
        state = _run_node(state, heal, "Healer", detailed_mode)
        state = _run_node(state, execute, "Executor (retry)", detailed_mode)

    state = _run_node(state, report, "Reporter", detailed_mode)

    final_state = state

    # ── Intermediate outputs ──────────────────────────────────────────────────
    _render_intermediate_outputs(
        final_state,
        detailed=detailed_mode,
        show_plan=show_plan,
        show_tests=show_tests,
    )

    # ── HTML report (optional) ────────────────────────────────────────────────
    html_path: Optional[str] = None
    if format == "html" and final_state.report and final_state.report_path:
        html_path = _write_html_report(final_state)

    # ── Results summary ───────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Results[/bold]"))

    status_icon = "✅" if final_state.execution_success else "❌"
    console.print(
        f"  {status_icon}  "
        f"[green]{final_state.tests_passed} passed[/green]  "
        f"[red]{final_state.tests_failed} failed[/red]  "
        f"[yellow]{final_state.tests_errors} errors[/yellow]"
    )

    if final_state.test_file_path:
        console.print(f"  [dim]Tests file :[/dim] {final_state.test_file_path}")
    if final_state.duplicate_reused and final_state.duplicate_source_path:
        console.print(f"  [dim]Duplicate reused :[/dim] {final_state.duplicate_source_path}")
    if html_path:
        console.print(f"  [dim]HTML report :[/dim] {html_path}")
    elif final_state.report_path:
        console.print(f"  [dim]Report     :[/dim] {final_state.report_path}")

    console.print()

    if final_state.errors:
        console.print("[yellow]Warnings:[/yellow]")
        for err in final_state.errors:
            console.print(f"  [yellow]⚠[/yellow] {err}")
        console.print()

    raise typer.Exit(code=0 if final_state.execution_success else 1)


def _write_html_report(state) -> str:
    """Convert the Markdown report to a self-contained HTML file."""
    from datetime import datetime, timezone

    md = state.report or ""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Minimal Markdown → HTML conversion (headings, bold, tables, code, lists)
    html_body = _md_to_html(md)

    status_class = "pass" if state.execution_success else "fail"
    status_text = "PASS ✅" if state.execution_success else "FAIL ❌"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agentic QA Report — {state.target}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0d1117; color: #c9d1d9; margin: 0; padding: 2rem; line-height: 1.6; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    header {{ border-bottom: 1px solid #30363d; margin-bottom: 2rem; padding-bottom: 1rem; }}
    header h1 {{ font-size: 1.5rem; color: #58a6ff; margin: 0 0 0.5rem; }}
    .badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 12px;
              font-weight: 600; font-size: 0.9rem; }}
    .badge.pass {{ background: #1a4731; color: #3fb950; }}
    .badge.fail {{ background: #4a1e1e; color: #f85149; }}
    .meta {{ display: flex; gap: 2rem; font-size: 0.85rem; color: #8b949e; margin-top: 0.5rem; }}
    h2 {{ color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 0.3rem; }}
    h3 {{ color: #79c0ff; }}
    code {{ background: #161b22; padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.9em; }}
    pre {{ background: #161b22; padding: 1rem; border-radius: 6px; overflow-x: auto;
           border: 1px solid #30363d; }}
    pre code {{ background: none; padding: 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th {{ background: #161b22; color: #58a6ff; text-align: left; padding: 0.5rem 0.75rem;
          border: 1px solid #30363d; }}
    td {{ padding: 0.5rem 0.75rem; border: 1px solid #21262d; }}
    tr:nth-child(even) {{ background: #161b22; }}
    ul, ol {{ padding-left: 1.5rem; }}
    footer {{ margin-top: 3rem; font-size: 0.8rem; color: #484f58;
              border-top: 1px solid #21262d; padding-top: 1rem; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Agentic QA Report</h1>
      <span class="badge {status_class}">{status_text}</span>
      <div class="meta">
        <span>Target: <strong>{state.target}</strong></span>
        <span>Type: {state.target_type}</span>
        <span>Passed: {state.tests_passed} | Failed: {state.tests_failed} | Errors: {state.tests_errors}</span>
        <span>Generated: {ts}</span>
      </div>
    </header>
    <main>
      {html_body}
    </main>
    <footer>Generated by <strong>Agentic QA</strong> — autonomous multi-agent test system</footer>
  </div>
</body>
</html>"""

    md_path = Path(state.report_path)
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)


def _md_to_html(md: str) -> str:
    """Minimal Markdown-to-HTML conversion (no external deps required)."""
    import re
    lines = md.split("\n")
    out = []
    in_code = False
    in_table = False

    for line in lines:
        # Code fences
        if line.startswith("```"):
            if in_code:
                out.append("</code></pre>")
                in_code = False
            else:
                lang = line[3:].strip() or ""
                out.append(f'<pre><code class="language-{lang}">')
                in_code = True
            continue
        if in_code:
            out.append(line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Tables
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                out.append("<table>")
                in_table = True
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue  # skip separator row
            tag = "th" if not any("<td>" in o or "<th>" in o for o in out[-5:]) else "td"
            out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        elif in_table:
            out.append("</table>")
            in_table = False

        # Headings
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{m.group(2)}</h{lvl}>")
            continue

        # Lists
        m = re.match(r"^[-*]\s+(.*)", line)
        if m:
            out.append(f"<li>{m.group(1)}</li>")
            continue
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            out.append(f"<li>{m.group(1)}</li>")
            continue

        # Inline bold/code
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)

        if line.strip():
            out.append(f"<p>{line}</p>")
        else:
            out.append("")

    if in_table:
        out.append("</table>")
    if in_code:
        out.append("</code></pre>")

    return "\n".join(out)


def _artifact_candidates(reports_dir: Path) -> list[Path]:
    """Return generated artifact files eligible for cleanup."""
    files: list[Path] = []
    files.extend(reports_dir.glob("report_*.md"))
    files.extend(reports_dir.glob("report_*.html"))
    files.extend(reports_dir.glob("report_*.json"))
    files.extend(reports_dir.rglob("generated_tests_*.py"))
    return sorted(set(files))


@app.command("clean")
def clean(
    reports_dir: str = typer.Option(
        "./reports",
        "--reports-dir", "-o",
        help="Directory containing generated tests/reports.",
        envvar="REPORTS_DIR",
    ),
    days: int = typer.Option(
        7,
        "--days", "-d",
        min=0,
        help="Delete artifacts older than this many days.",
    ),
    all_files: bool = typer.Option(
        False,
        "--all",
        help="Delete all generated artifacts regardless of age.",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Actually delete files. Without this flag, performs a dry run.",
    ),
) -> None:
    """Clean generated test scripts and report artifacts."""

    root = Path(reports_dir)
    if not root.exists():
        console.print(f"[yellow]Reports directory not found:[/yellow] {root}")
        raise typer.Exit(code=0)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    candidates = _artifact_candidates(root)

    to_remove: list[Path] = []
    for path in candidates:
        if all_files:
            to_remove.append(path)
            continue

        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if mtime < cutoff:
            to_remove.append(path)

    if not to_remove:
        console.print("[green]No generated artifacts matched cleanup criteria.[/green]")
        raise typer.Exit(code=0)

    mode = "APPLY" if apply else "DRY RUN"
    console.print(Rule(f"[bold]Artifact Cleanup ({mode})[/bold]"))
    for path in to_remove:
        console.print(f"  - {path}")

    if not apply:
        console.print("\n[yellow]Dry run only.[/yellow] Re-run with [bold]--apply[/bold] to delete.")
        raise typer.Exit(code=0)

    deleted = 0
    for path in to_remove:
        try:
            path.unlink(missing_ok=True)
            deleted += 1
        except OSError as exc:
            console.print(f"[red]Failed to delete[/red] {path}: {exc}")

    console.print(f"\n[green]Deleted {deleted} generated artifact(s).[/green]")


def _active_model_name(provider: str) -> str:
    """Return the currently configured model name for the selected provider."""
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    return os.getenv("OPENAI_MODEL", "gpt-4o")


def _validate_env(provider: str) -> None:
    """Abort early if required environment variables are missing for provider."""
    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            console.print(
                "[red]Error:[/red] ANTHROPIC_API_KEY is not set.\n"
                "Copy [bold].env.example[/bold] -> [bold].env[/bold] and add your key."
            )
            raise typer.Exit(code=1)
        return

    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]Error:[/red] OPENAI_API_KEY is not set.\n"
            "Copy [bold].env.example[/bold] -> [bold].env[/bold] and add your key."
        )
        raise typer.Exit(code=1)


@app.command("version")
def version() -> None:
    """Print the current version of Agentic QA."""
    from agentic_qa import __version__
    console.print(f"agentic-qa v{__version__}")


if __name__ == "__main__":
    app()
