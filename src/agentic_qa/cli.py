"""Agentic QA CLI — entry point for the multi-agent QA pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
) -> None:
    """Run the full Agentic QA pipeline against TARGET."""

    _validate_env()

    # Override env vars from CLI flags
    if model:
        os.environ["OPENAI_MODEL"] = model
    os.environ["MAX_TEST_CASES"] = str(max_tests)
    os.environ["REPORTS_DIR"] = reports_dir

    # Import here so env vars are set before LangChain initialises
    from agentic_qa.graph import graph
    from agentic_qa.state import QAState

    console.print()
    console.print(Panel(
        Text.from_markup(
            f"[bold cyan]Agentic QA[/bold cyan]  [dim]|[/dim]  "
            f"[white]{target}[/white]  [dim]|[/dim]  "
            f"model: [green]{os.getenv('OPENAI_MODEL', 'gpt-4o')}[/green]"
        ),
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()

    initial_state = QAState(target=target)

    steps = [
        ("orchestrator", "Orchestrating — classifying target"),
        ("planner",      "Planning — analysing target and designing test cases"),
        ("writer",       "Writing — generating pytest test code"),
        ("executor",     "Executing — running tests"),
        ("reporter",     "Reporting — compiling QA report"),
    ]

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

            # Stream one node at a time for live updates
            stream = graph.stream(
                initial_state.model_dump() if final_state is None
                else final_state.model_dump(),
                {"recursion_limit": 25},
            )

            for chunk in stream:
                if node in chunk:
                    merged = (final_state.model_dump() if final_state else initial_state.model_dump())
                    merged.update(chunk[node])
                    final_state = QAState(**merged)
                    break  # one node per iteration

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
    if final_state.report_path:
        console.print(f"  [dim]Report     :[/dim] {final_state.report_path}")

    console.print()

    if final_state.errors:
        console.print("[yellow]Warnings:[/yellow]")
        for err in final_state.errors:
            console.print(f"  [yellow]⚠[/yellow] {err}")
        console.print()

    raise typer.Exit(code=0 if final_state.execution_success else 1)


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
