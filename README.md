# Agentic QA

> **Multi-agent AI system that autonomously plans, writes, executes, and reports tests for REST APIs, web UIs, and Python codebases.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LLM Provider](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Anthropic-green.svg)](https://www.anthropic.com/)
[![pytest](https://img.shields.io/badge/tests-pytest-yellow.svg)](https://pytest.org/)

---

## What It Does

Point Agentic QA at any target and it runs a full autonomous QA cycle:

```
You: agentic-qa run https://jsonplaceholder.typicode.com

Orchestrator  →  "This is a REST API"
Planner       →  Fetches endpoints, designs 10 test cases (positive + negative + edge)
Writer        →  Generates runnable pytest code using httpx
Executor      →  Runs pytest, captures output
Reporter      →  Produces a Markdown QA report with pass/fail analysis
```

All without you writing a single line of test code.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  LangGraph Pipeline                  │
│                                                     │
│  [Orchestrator] → [Planner] → [Writer]             │
│                                    ↓                │
│                            [Executor]               │
│                                    ↓                │
│                            [Reporter]               │
└─────────────────────────────────────────────────────┘
```

| Agent | Responsibility |
|-------|----------------|
| **Orchestrator** | Classifies target as `api`, `web`, or `code` |
| **Planner** | Inspects target (fetches spec/page/source) and designs a test plan |
| **Writer** | Converts the plan to runnable `pytest` code (httpx / Playwright / unittest) |
| **Executor** | Runs the test file in a subprocess, captures stdout/stderr |
| **Reporter** | Analyses results and writes a Markdown QA report |

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/your-username/agentic-qa.git
cd agentic-qa
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
playwright install chromium   # only needed for web targets
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set LLM_PROVIDER + matching API key
```

### 3. Run

```bash
# Test a public REST API
agentic-qa run https://jsonplaceholder.typicode.com

# Interactive run setup + approval gate (recommended)
# Prompts for strategy (smoke/sanity/regression/custom), max tests,
# planning notes, and asks for plan approval before execution.
agentic-qa run https://jsonplaceholder.typicode.com --interactive

# Non-interactive run (CI-friendly)
agentic-qa run https://jsonplaceholder.typicode.com --no-interactive --strategy smoke

# Test a website (UI / accessibility)
agentic-qa run https://example.com

# Test a local OpenAPI spec
agentic-qa run ./my-api/openapi.yaml

# Test a Python module
agentic-qa run ./src/mypackage/utils.py

# Show the test plan and generated tests too
agentic-qa run https://jsonplaceholder.typicode.com --show-plan --show-tests

# Auto-fix failing tests with the Healer agent
agentic-qa run https://jsonplaceholder.typicode.com --self-heal

# Get a self-contained HTML report instead of Markdown
agentic-qa run https://api.example.com --format html

# Use Claude
agentic-qa run https://api.example.com --provider anthropic --model claude-sonnet-4-6

# Use OpenAI
agentic-qa run https://api.example.com --provider openai --model gpt-4o

# Customise test count
agentic-qa run https://api.example.com --max-tests 15
```

### 4. Find your report

```
reports/
├── generated_tests_a1b2c3d4.py   ← runnable pytest file
└── report_jsonplaceholder_typicode_com_20260508_120000.md
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` / `-p` | `openai` | LLM provider: `openai` or `anthropic` |
| `--model` / `-m` | provider default | Model name for selected provider |
| `--max-tests` / `-n` | `10` | Target number of test cases |
| `--strategy` / `-s` | `smoke` | Test strategy: `smoke`, `sanity`, `regression`, `custom` |
| `--planning-notes` | empty | Extra planning instructions for required scenarios |
| `--interactive` / `--no-interactive` | `--interactive` | Enable setup prompts and plan approval before execution |
| `--reports-dir` / `-o` | `./reports` | Output directory |
| `--self-heal` | off | Rewrite and re-run failing tests via the Healer agent |
| `--format` / `-f` | `markdown` | Report format: `markdown` or `html` |
| `--show-plan` | off | Print the test plan to stdout |
| `--show-tests` | off | Print the generated test code to stdout |

---

## Human-In-The-Loop Flow

When `--interactive` is enabled, Agentic QA now pauses after planning and lets you:

- Approve the proposed test cases before execution
- Request alterations to existing test cases
- Add additional test cases or coverage requests
- Cancel the run before any tests are generated or executed

---

## Running the Unit Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — agent state graph and pipeline orchestration
- **[LangChain OpenAI](https://python.langchain.com/)** and **[LangChain Anthropic](https://python.langchain.com/)** — provider-selectable LLM integration
- **[httpx](https://www.python-httpx.org/)** — async-ready HTTP client for API target inspection and generated tests
- **[Playwright](https://playwright.dev/python/)** — browser automation for web targets
- **[pytest](https://pytest.org/)** — test runner for generated tests and self-tests
- **[Typer](https://typer.tiangolo.com/) + [Rich](https://rich.readthedocs.io/)** — CLI and terminal output

---

## Why This Is Interesting (for Recruiters)

- **Real agentic architecture** — not a single LLM call, but a coordinated multi-node graph with shared state
- **Generates runnable artefacts** — the output is actual `.py` test files you can inspect, version, and re-run
- **Target-agnostic** — one tool handles APIs, UIs, and source code
- **Observable** — every agent's reasoning is captured in the shared message history; the report explains failures

---

## Roadmap

- [ ] Parallel agent execution for large test suites
- [ ] GitHub Actions integration (run on PR open) ✅ done
- [ ] HTML report with embedded test code viewer ✅ done
- [ ] Self-healing: automatically retry failed tests with LLM-suggested fixes ✅ done
- [ ] Support for GraphQL and gRPC targets
- [ ] `--watch` mode: re-run on file change
