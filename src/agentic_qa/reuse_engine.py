"""Score-only reuse decision engine scaffold.

This module computes reuse candidate scores but does not alter generation behavior.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import os
from pathlib import Path
import re


def score_reuse_candidates(
    *,
    project_tests_dir: Path,
    per_test_units: list[dict[str, str]],
    strategy: str,
    target_type: str,
) -> list[dict]:
    """Return score-only reuse recommendations for each generated unit.

    This is phase-2 scaffolding in score-only mode: decisions are informational
    and should not change file write/reuse behavior yet.
    """
    candidates = [p for p in project_tests_dir.glob("*.py") if p.name != "conftest.py"]
    cfg = _reuse_config_from_env()

    decisions: list[dict] = []
    for unit in per_test_units:
        case_id = _case_id_from_test_name(unit.get("test_name", "generated_test"))
        if not candidates:
            decisions.append(
                {
                    "case_id": case_id,
                    "decision": "generate_new",
                    "candidate_file": None,
                    "score": 0.0,
                    "components": {"intent": 0.0, "method": 0.0, "origin": 0.0, "quality": 0.0},
                    "threshold_band": "low",
                    "reason": "No prior project tests available",
                    "requires_user_approval": False,
                    "mode": "score-only",
                }
            )
            continue

        best = None
        for path in candidates:
            try:
                existing_code = path.read_text(encoding="utf-8")
            except OSError:
                continue

            intent = _intent_similarity(unit.get("description", ""), _extract_test_description(existing_code, path))
            method = _method_similarity(unit.get("code", ""), existing_code)
            origin = _origin_alignment(unit.get("test_name", ""), path.name)
            quality = _quality_proxy(existing_code)
            score = round(
                cfg["weight_intent"] * intent
                + cfg["weight_method"] * method
                + cfg["weight_origin"] * origin
                + cfg["weight_quality"] * quality,
                4,
            )

            candidate = {
                "case_id": case_id,
                "candidate_file": str(path.resolve()),
                "score": score,
                "components": {
                    "intent": round(intent, 4),
                    "method": round(method, 4),
                    "origin": round(origin, 4),
                    "quality": round(quality, 4),
                },
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        if best is None:
            decisions.append(
                {
                    "case_id": case_id,
                    "decision": "generate_new",
                    "candidate_file": None,
                    "score": 0.0,
                    "components": {"intent": 0.0, "method": 0.0, "origin": 0.0, "quality": 0.0},
                    "threshold_band": "low",
                    "reason": "Unable to read candidate files",
                    "requires_user_approval": False,
                    "mode": "score-only",
                }
            )
            continue

        band = _score_band(
            best["score"],
            high_threshold=cfg["threshold_high"],
            mid_threshold=cfg["threshold_mid"],
            quality=best["components"]["quality"],
            quality_minimum=cfg["quality_minimum"],
        )
        decision = "reuse_candidate" if band in ("high", "mid") else "generate_new"
        decisions.append(
            {
                "case_id": best["case_id"],
                "decision": decision,
                "candidate_file": best["candidate_file"],
                "score": best["score"],
                "components": best["components"],
                "threshold_band": band,
                "reason": _decision_reason(band, strategy, target_type),
                "requires_user_approval": band == "mid",
                "mode": "score-only",
                "config": cfg,
            }
        )

    return decisions


def _case_id_from_test_name(test_name: str) -> str:
    match = re.search(r"(tc\d+)", test_name, flags=re.IGNORECASE)
    if match:
        token = match.group(1).upper()
        return f"{token[:2]}-{token[2:]}"
    return test_name


def _extract_test_description(code: str, path: Path) -> str:
    match = re.search(r"^Description:\s*(.+)$", code, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return path.stem.replace("_", " ")


def _intent_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _method_similarity(a: str, b: str) -> float:
    keys = ["httpx", "playwright", "client", "browser", "assert", "pytest", "status_code"]
    a_tokens = {k for k in keys if k in a}
    b_tokens = {k for k in keys if k in b}
    if not a_tokens and not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return overlap / union if union else 0.0


def _origin_alignment(test_name: str, file_name: str) -> float:
    if not test_name.strip() or not file_name.strip():
        return 0.0
    return SequenceMatcher(None, test_name.lower(), file_name.lower()).ratio()


def _quality_proxy(code: str) -> float:
    low_signals = ["xfail", "skip", "TODO", "pass  #"]
    if any(signal.lower() in code.lower() for signal in low_signals):
        return 0.35
    if "assert" not in code:
        return 0.4
    return 0.8


def _score_band(
    score: float,
    *,
    high_threshold: float,
    mid_threshold: float,
    quality: float,
    quality_minimum: float,
) -> str:
    if quality < quality_minimum:
        return "low"
    if score >= high_threshold:
        return "high"
    if score >= mid_threshold:
        return "mid"
    return "low"


def _decision_reason(band: str, strategy: str, target_type: str) -> str:
    if band == "high":
        return f"High similarity and quality match for {target_type} {strategy} case"
    if band == "mid":
        return "Moderate match; candidate should be reviewed before reuse"
    return "Low confidence match; generating new test is safer"


def _reuse_config_from_env() -> dict[str, float]:
    """Load tunable scoring config from env with safe defaults."""
    cfg = {
        "weight_intent": _env_float("REUSE_WEIGHT_INTENT", 0.35),
        "weight_method": _env_float("REUSE_WEIGHT_METHOD", 0.25),
        "weight_origin": _env_float("REUSE_WEIGHT_ORIGIN", 0.20),
        "weight_quality": _env_float("REUSE_WEIGHT_QUALITY", 0.20),
        "threshold_high": _env_float("REUSE_THRESHOLD_HIGH", 0.85),
        "threshold_mid": _env_float("REUSE_THRESHOLD_MID", 0.70),
        "quality_minimum": _env_float("REUSE_QUALITY_MINIMUM", 0.60),
    }

    # Normalize weights when custom values do not sum to 1.0
    weight_sum = cfg["weight_intent"] + cfg["weight_method"] + cfg["weight_origin"] + cfg["weight_quality"]
    if weight_sum > 0:
        cfg["weight_intent"] = cfg["weight_intent"] / weight_sum
        cfg["weight_method"] = cfg["weight_method"] / weight_sum
        cfg["weight_origin"] = cfg["weight_origin"] / weight_sum
        cfg["weight_quality"] = cfg["weight_quality"] / weight_sum

    return {k: round(v, 4) for k, v in cfg.items()}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
