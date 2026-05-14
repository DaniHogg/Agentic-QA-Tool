# Project Memory Blueprint

## Purpose
Create a durable, feedback-driven knowledge library that improves planning, test generation, duplicate matching, and run quality over time without requiring model retraining.

## Outcomes
- Retain user approvals, rejections, edits, and post-run feedback.
- Reuse prior high-quality test structures automatically.
- Detect duplicates before writing new test files.
- Rank historical examples by quality and relevance.

## Phase 1: Feedback Capture and Durable Memory

### Scope
1. Capture feedback at plan review, test review, and post-run review.
2. Persist run artifacts, approvals, edits, and outcomes with stable IDs.
3. Provide basic project-level retrieval (latest and matching target type).

### SQLite Schema
```sql
CREATE TABLE projects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE runs (
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

CREATE TABLE plan_cases (
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

CREATE TABLE test_files (
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

CREATE TABLE run_feedback (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  verdict TEXT NOT NULL,
  defect_quality TEXT NOT NULL,
  notes TEXT DEFAULT '',
  created_at TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE TABLE memory_items (
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
```

### File Structure
```text
knowledge/
  runs/
    {run_id}/
      plan.json
      approved_plan.json
      test_index.json
      run_feedback.json
  exports/
    project_{project_name}_memory_snapshot.json
```

### Acceptance Criteria
1. Every plan case and test file stores explicit feedback state.
2. Query returns approved test files for a given project.
3. Project memory snapshot export is available.

## Phase 2: Retrieval and Duplicate Matching

### Scope
1. Add hard-filter retrieval by project, target type, and strategy.
2. Add semantic candidate retrieval and quality re-rank.
3. Add duplicate scoring before writing tests.

### Duplicate Score
Use weighted score:

S = 0.55 * H + 0.25 * E + 0.20 * M

- H: hash match (0 or 1)
- E: endpoint or signature similarity (0 to 1)
- M: semantic similarity (0 to 1)

### Decision Thresholds
1. S >= 0.85: propose auto-reuse
2. 0.70 <= S < 0.85: ask for user confirmation
3. S < 0.70: generate new

### Optional Table
```sql
CREATE TABLE duplicate_events (
  id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  candidate_test_file_id TEXT NOT NULL,
  incoming_hash TEXT NOT NULL,
  score REAL NOT NULL,
  decision TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(project_id) REFERENCES projects(id)
);
```

### Acceptance Criteria
1. Duplicate suggestions appear before write.
2. User override path exists and is logged.
3. Duplicate decision includes explainable score breakdown.

## Phase 3: Prompt Shaping and Quality Loop

### Scope
1. Inject top approved examples and avoid-pattern examples into prompts.
2. Update quality scores based on user feedback and run outcomes.
3. Add reusable rule memory entries (naming, fixtures, assertion style).

### Quality Update Heuristic
Q_new = Q_old + 0.4A + 0.3P - 0.2F - 0.1H

- A: user approval (0 or 1)
- P: run passed (0 or 1)
- F: flaky indicator (0 or 1)
- H: self-heal needed (0 or 1)

### Acceptance Criteria
1. Planner and writer receive ranked memory snippets.
2. Low-quality patterns are down-ranked over time.
3. UI can show why a memory item was used.

## Phase 4: Analytics and Optional Training Prep

### Scope
1. Add trend metrics: approval rate, duplicate reuse, first-pass pass rate, self-heal rate.
2. Add export pipeline for approved or edited pairs for possible fine-tuning later.

### JSONL Export Shape
```json
{"task":"plan_case_refinement","input":"...","output":"...","project":"payments-api"}
{"task":"test_generation_refinement","input":"...","output":"...","project":"payments-api"}
```

### Acceptance Criteria
1. Project-level trend report exists.
2. Export dataset is reproducible and versioned.

## Module Mapping For Current Codebase

### src/agentic_qa/context_store.py
Add new methods:
- save_run_record(run_meta: dict) -> str
- save_plan_case_feedback(run_id: str, case_id: str, approved: bool, reason: str, edited_body: str | None = None) -> None
- save_test_file_feedback(run_id: str, file_path: str, approved: bool, reason: str) -> None
- save_run_feedback(run_id: str, verdict: str, defect_quality: str, notes: str) -> None
- get_memory_candidates(project_name: str, target_type: str, strategy: str, limit: int = 25) -> list[dict]

### src/agentic_qa/ui.py
Integration points:
1. Plan tabs: persist per-case decisions on approval action.
2. Test tabs: persist per-file approvals or rejections.
3. Completed stage: add post-run feedback controls and save.
4. Add Memory Usage panel showing top retrieved examples and scores.

### src/agentic_qa/agents/planner.py
Before prompt call:
1. Pull top memory candidates from context store.
2. Build concise memory snippet section.
3. Add snippet to HumanMessage for plan generation.

### src/agentic_qa/agents/writer.py
Before writing files:
1. Compute duplicate score against approved memory test files.
2. If threshold matched, return reuse recommendation.
3. Log duplicate decision event.

### src/agentic_qa/agents/executor.py
After execution:
1. Emit structured outcome signals (pass, fail, errors).
2. Persist outcome for quality scoring updates.

### src/agentic_qa/cli.py
Add optional switches:
- --memory-enabled/--no-memory-enabled
- --memory-limit <int>
- --min-duplicate-score <float>
- --save-run-feedback

## Potential Blockers and Mitigations
1. Feedback quality may be inconsistent.
   - Mitigation: require a reason tag for reject and edit actions.
2. Retrieval context can become noisy.
   - Mitigation: cap injected items and rank by quality and recency.
3. Duplicate matching false positives.
   - Mitigation: require score breakdown and human confirmation in mid band.
4. Metadata drift across versions.
   - Mitigation: include schema_version in metadata_json.
5. Secret leakage in stored artifacts.
   - Mitigation: redact tokens and keys before persistence.

## Rollout Plan
1. Week 1: implement schema, persistence, and basic retrieval.
2. Week 2: duplicate scoring, reuse prompts, and decision logging.
3. Week 3: prompt shaping with quality re-rank.
4. Week 4: analytics and export flow.

## Definition of Done
1. Memory-backed retrieval is active for planner and writer.
2. User feedback from approvals is persisted and reused.
3. Duplicate suggestions are explainable and overrideable.
4. Trend metrics and memory snapshot export are available.