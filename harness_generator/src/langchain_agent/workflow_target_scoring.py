from __future__ import annotations

import re
from typing import Any, Callable


def clamp_score(value: float, *, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, float(value)))


def target_component_coverage_gap(item: dict[str, Any]) -> float:
    explicit = item.get("coverage_gap")
    if explicit is not None:
        try:
            return clamp_score(float(explicit))
        except Exception:
            pass
    depth_score = max(0, int(item.get("depth_score") or 0))
    depth_class = str(item.get("depth_class") or "").strip().lower()
    target_type = str(item.get("target_type") or "").strip().lower()
    base = min(7.0, float(depth_score) / 2.0)
    if depth_class == "deep":
        base += 2.0
    elif depth_class == "medium":
        base += 1.0
    if target_type in {"parser", "decoder", "archive"}:
        base += 1.0
    return clamp_score(base)


def target_component_complexity(item: dict[str, Any]) -> float:
    depth_score = max(0, int(item.get("depth_score") or 0))
    risk_signals = list(item.get("risk_signals") or [])
    base = min(8.0, float(depth_score) / 3.0)
    base += min(2.0, 0.4 * float(len(risk_signals)))
    return clamp_score(base)


def target_component_api_relevance(
    item: dict[str, Any],
    *,
    runtime_viability_rank_fn: Callable[[str], int],
) -> float:
    runtime_rank = runtime_viability_rank_fn(str(item.get("runtime_viability") or ""))
    target_type = str(item.get("target_type") or "").strip().lower()
    api = str(item.get("api") or "")
    score = 2.0 + float(runtime_rank) * 2.5
    if target_type in {"parser", "decoder", "archive"}:
        score += 1.5
    if "::" in api or re.search(r"[A-Za-z_][A-Za-z0-9_]*", api):
        score += 1.0
    return clamp_score(score)


def target_component_consumer_order_support(item: dict[str, Any]) -> float:
    target_type = str(item.get("target_type") or "").strip().lower()
    rationale = str(item.get("selection_rationale") or "").lower()
    bias = str(item.get("selection_bias_reason") or "").lower()
    signals = " ".join(str(x).lower() for x in (item.get("risk_signals") or []))
    score = 2.0
    if target_type in {"parser", "archive", "decoder"}:
        score += 2.0
    if any(tok in rationale for tok in ("runtime", "entrypoint", "stream", "state")):
        score += 2.0
    if any(tok in bias for tok in ("state", "parse", "decode", "deep")):
        score += 1.5
    if "state-machine" in signals or "parser-like" in signals:
        score += 1.5
    return clamp_score(score)


def target_score_breakdown(
    item: dict[str, Any],
    *,
    weights: dict[str, float],
    runtime_viability_rank_fn: Callable[[str], int],
) -> dict[str, Any]:
    coverage_gap = target_component_coverage_gap(item)
    complexity = target_component_complexity(item)
    api_relevance = target_component_api_relevance(
        item,
        runtime_viability_rank_fn=runtime_viability_rank_fn,
    )
    complexity_depth = complexity
    consumer_order_support = target_component_consumer_order_support(item)
    weighted_total = (
        coverage_gap * float(weights["coverage_gap"])
        + complexity * float(weights["complexity"])
        + api_relevance * float(weights["api_relevance"])
        + consumer_order_support * float(weights["consumer_order_support"])
    )
    return {
        "coverage_gap": round(coverage_gap, 4),
        "complexity": round(complexity, 4),
        "complexity_depth": round(complexity_depth, 4),
        "api_relevance": round(api_relevance, 4),
        "consumer_order_support": round(consumer_order_support, 4),
        "recent_yield_penalty": 0.0,
        "weights": {k: round(float(v), 4) for k, v in weights.items()},
        "weighted_total": round(weighted_total, 6),
    }


def runtime_penalty_from_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    if not feedback:
        return {"score_penalty": 0.0, "reason": "", "seed_feedback": {}}
    cold_start = bool(feedback.get("cold_start_failure") or False)
    seed_score = float(feedback.get("seed_score") or 0.0)
    early_units_30s = int(feedback.get("early_new_units_30s") or 0)
    penalty = 0.0
    reason = ""
    if cold_start and seed_score < 0.55 and early_units_30s <= 0:
        penalty = 1.5
        reason = "cold_start_low_yield"
    elif seed_score < 0.30:
        penalty = 0.8
        reason = "very_low_seed_score"
    return {"score_penalty": float(penalty), "reason": reason, "seed_feedback": dict(feedback)}
