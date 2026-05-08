"""Agentic QA CLI — entry point for the multi-agent QA pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.text import Text

load_dotenv()

app = typer.Typer(
    name="agentic-qa",
    help="Autonomous QA agent — plans, writes, runs, and reports tests for APIs, UIs, and Python codebases.",
    add_completion=False,
)

console = Console()


@app.command()
def run(
    target: str = typer.Argument(
        ...,
        help="URL (API or web) or local file/directory path to test.",
    ),
    model: str = typer.Option(
        None,
        "--model", "-m",
        help="OpenAI model to use (default: gpt-4o).",
        envvar="OPENAI_MODEL",
    ),
    max_tests: int = typer.Option(
        10,
        "--max-tests", "-n",
        help="Target number of test cases for the Planner to generate.",
        envvar="MAX_TEST_CASES",
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
) -> None:
    """Run the full Agentic QA pipeline against TARGET."""

    _validate_env()

    if format not in ("markdown", "html"):
        console.print("[red]Error:[/red] --format must be 'markdown' or 'html'.")
        raise typer.Exit(code=1)

    # Override env vars from CLI flags
    if model:
        os.environ["OPENAI_MODEL"] = model
    os.environ["MAX_TEST_CASES"] = str(max_tests)
    os.environ["REPORTS_DIR"] = reports_dir

    # Import here so env vars are set before LangChain initialises
    from agentic_qa.graph import get_graph
    from agentic_qa.state import QAState

    active_graph = get_graph(self_heal=self_heal)

    heal_label = "  [yellow]+self-heal[/yellow]" if self_heal else ""
    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold cyan]Agentic QA[/bold cyan]  [dim]|[/dim]  "
            f"[white]{target}[/white]  [dim]|[/dim]  "
            f"model: [green]{os.getenv('OPENAI_MODEL', 'gpt-4o')}[/green]"
            f"{heal_label}"
        ),
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()

    initial_state = QAState(target=target, self_heal=self_heal)

    steps = [
        ("orchestrator", "Orchestrating — classifying target"),
        ("planner",      "Planning — analysing target and designing test cases"),
        ("writer",       "Writing — generating pytest test code"),
        ("executor",     "Executing — running tests"),
    ]
    if self_heal:
        steps += [
            ("healer",    "Healing — rewriting failing tests"),
            ("executor2", "Re-executing — running healed tests"),
        ]
    steps.append(("reporter", "Reporting — compiling QA report"))

    final_state: QAState | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Starting pipeline…", total=len(steps))

        for i, (node, label) in enumerate(steps):
            progress.update(task, description=f"[cyan]{label}…", completed=i)

            stream = active_graph.stream(
                initial_state.model_dump() if final_state is None
                else final_state.model_dump(),
                {"recursion_limit": 30},
            )

            for chunk in stream:
                if node in chunk:
                    merged = (final_state.model_dump() if final_state else initial_state.model_dump())
                    merged.update(chunk[node])
                    final_state = QAState(**merged)
                    break

        progress.update(task, description="[green]Pipeline complete ✓", completed=len(steps))

    if final_state is None:
        console.print("[red]Pipeline produced no output. Check your OPENAI_API_KEY.[/red]")
        raise typer.Exit(code=1)

    # ── Optional verbose output ───────────────────────────────────────────────
    if show_plan and final_state.test_plan:
        console.print(Rule("[bold]Test Plan[/bold]"))
        console.print(Markdown(final_state.test_plan))

    if show_tests and final_state.generated_tests:
        console.print(Rule("[bold]Generated Tests[/bold]"))
        console.print(final_state.generated_tests)

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
    import re
    from datetime import datetime, timezone

    md = state.report or ""
    reports_dir = Path(os.getenv("REPORTS_DIR", "./reports"))
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


def _validate_env() -> None:
    """Abort early if required environment variables are missing."""
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]Error:[/red] OPENAI_API_KEY is not set.\n"
            "Copy [bold].env.example[/bold] → [bold].env[/bold] and add your key."
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
