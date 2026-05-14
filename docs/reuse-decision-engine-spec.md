# Reuse Decision Engine Specification

## Purpose
Define a deterministic policy for deciding when to reuse an existing test versus generating a new one, using stored project memory and current run intent.

## Design Goals
1. Reduce duplicate tests without missing coverage.
2. Prefer reliable approved tests where confidence is high.
3. Generate new tests when intent is novel, risky, or low-confidence.
4. Keep decisions explainable with score breakdowns.

## Decision Policy

### High-Level Strategy
Use reuse-first for high-confidence and high-quality matches, then generate for uncovered or uncertain cases.

### Decision Order
1. Hard gating (eligibility)
2. Candidate scoring
3. Confidence threshold decision
4. Coverage gap fallback generation

## Inputs

### Current Intent Input
Per planned test case:
1. case_id
2. title
3. description/body
4. strategy (smoke, sanity, regression, custom)
5. target_type (api, web, code)
6. project_name

### Historical Candidate Input
From memory and filesystem:
1. approved flag
2. feedback reason and post-run notes
3. content_hash
4. description
5. endpoint signature (if available)
6. quality signals (pass/fail, self-heal usage, rejection history)
7. recency (last successful run)

## Hard Gates (Must Pass)
Reject candidate before scoring if any fail:
1. Candidate belongs to different project bucket (unless cross-project mode is enabled later).
2. Candidate target_type differs from current target_type.
3. Candidate has explicit rejection without later approval.
4. Candidate is flagged flaky or repeatedly self-healed.
5. Candidate was generated for incompatible strategy profile.

## Scoring Model

### Composite Score
S = 0.35 * I + 0.25 * M + 0.20 * O + 0.20 * Q

Where:
1. I = Intent similarity (what it tests)
2. M = Method similarity (how it tests)
3. O = Origin alignment (goal alignment with prior approved case intent)
4. Q = Quality confidence from historical run outcomes

Each component is normalized to 0.0 to 1.0.

### Component Definitions

#### Intent Similarity (I)
Compare plan case title/body with candidate description and stored case text.
Suggested calculation:
1. semantic_similarity(title + body, description + stored_case_text)
2. keyword overlap for endpoint, resource, behavior terms

#### Method Similarity (M)
Compare testing style and structure:
1. fixture usage pattern
2. assertion style (status/schema/content)
3. API method or UI action profile
4. setup/teardown complexity

#### Origin Alignment (O)
Check if candidate originally served similar approved plan intent:
1. same or similar case objective
2. similar risk category (positive/negative/edge)

#### Quality Confidence (Q)
Derived from memory outcomes:
1. approved in review stage
2. passed on first execution
3. no repeated failures
4. no self-heal dependency trend

## Thresholds
1. S >= 0.85: Reuse directly
2. 0.70 <= S < 0.85: Present reuse recommendation and require explicit approval
3. S < 0.70: Generate new

### Coverage Guard
Even when reusing high-score candidates:
1. Ensure all approved plan intents are covered.
2. Generate new tests for uncovered intents.

## Tie-Breaker Rules
When multiple candidates exceed threshold:
1. Higher Q (quality confidence)
2. More recent successful run
3. Better strategy compatibility
4. Lower maintenance complexity (smaller, cleaner, fewer brittle selectors)

## Data Requirements (Phase 2)

### Extend test_files metadata
Store in metadata_json or new fields:
1. test_type (positive, negative, edge)
2. endpoint signature or action signature
3. fixture profile
4. assertion profile

### Extend run quality signals
1. first_pass_success
2. self_heal_required
3. flaky_indicator
4. rejection_count

## Integration Points in Current Codebase

### Planner Stage
File: [src/agentic_qa/agents/planner.py](src/agentic_qa/agents/planner.py)
1. Parse plan cases into structured intent units.
2. Attach retrieval hints per case for writer.

### Writer Stage
File: [src/agentic_qa/agents/writer.py](src/agentic_qa/agents/writer.py)
1. Replace simple duplicate check with decision-engine call per plan case.
2. For each case, either:
   - reuse candidate test file, or
   - generate new test file
3. Return per-case decision rationale in state updates.

### Memory Query Layer
File: [src/agentic_qa/context_store.py](src/agentic_qa/context_store.py)
1. Add query for candidate retrieval by project, target_type, strategy.
2. Add quality signal extraction helper.
3. Add reuse decision logging table (optional early, required later for analytics).

### UI Stage
File: [src/agentic_qa/ui.py](src/agentic_qa/ui.py)
1. Show per-case decision:
   - reused vs generated
   - score and component breakdown
2. In mid-confidence band, require user approval.

## Suggested New State Fields
File: [src/agentic_qa/state.py](src/agentic_qa/state.py)
1. reuse_decisions: list of per-case decision objects
2. generated_from_scratch_count: int
3. reused_count: int
4. coverage_gaps: list of uncovered intent IDs

## Decision Object Contract
Use this structure for each planned case decision:

```json
{
  "case_id": "TC-003",
  "decision": "reused",
  "candidate_file": "/abs/path/generated_test_tc003_xxx.py",
  "score": 0.89,
  "components": {
    "intent": 0.91,
    "method": 0.84,
    "origin": 0.88,
    "quality": 0.92
  },
  "threshold_band": "high",
  "reason": "High intent and quality match",
  "requires_user_approval": false
}
```

## Manual Override Rules
1. User can force generate even for high-score reuse.
2. User can force reuse for mid-score candidate.
3. Overrides are logged for future threshold tuning.

## Rollout Plan
1. Step 1: Implement retrieval + scoring without changing writer behavior, log scores only.
2. Step 2: Enable reuse for high band, preserve manual review for mid band.
3. Step 3: Add coverage gap generation and UI explanation panel.
4. Step 4: Tune weights/thresholds from observed outcomes.

## Initial Defaults
1. enable_reuse_engine = true
2. high_threshold = 0.85
3. mid_threshold = 0.70
4. quality_minimum_for_reuse = 0.60
5. require_approval_for_mid_band = true

## Validation Metrics
Track before/after:
1. duplicate test creation rate
2. first-pass test success rate
3. reviewer approval rate
4. number of manual overrides
5. coverage gap count per run

## Out of Scope (Current Spec)
1. Cross-project reuse by default.
2. Embedding model selection and infrastructure details.
3. Automated pruning of old memory entries.
