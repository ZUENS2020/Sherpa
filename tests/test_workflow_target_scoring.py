from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_target_scoring as s


def test_target_score_breakdown_includes_expected_keys() -> None:
    weights = {
        "coverage_gap": 0.35,
        "complexity": 0.25,
        "api_relevance": 0.20,
        "consumer_order_support": 0.20,
    }
    item = {
        "coverage_gap": 6.0,
        "depth_score": 10,
        "target_type": "parser",
        "runtime_viability": "native_high",
        "api": "png::decode",
        "selection_rationale": "entrypoint stream parser state",
        "risk_signals": ["state-machine"],
    }
    out = s.target_score_breakdown(
        item,
        weights=weights,
        runtime_viability_rank_fn=lambda _: 3,
    )
    assert out["coverage_gap"] >= 0
    assert out["complexity_depth"] >= 0
    assert out["api_relevance"] >= 0
    assert "weighted_total" in out
    assert set(out["weights"].keys()) == set(weights.keys())


def test_runtime_penalty_from_feedback_cold_start() -> None:
    out = s.runtime_penalty_from_feedback(
        {
            "cold_start_failure": True,
            "seed_score": 0.4,
            "early_new_units_30s": 0,
        }
    )
    assert out["score_penalty"] == 1.5
    assert out["reason"] == "cold_start_low_yield"


def test_runtime_penalty_from_feedback_very_low_score() -> None:
    out = s.runtime_penalty_from_feedback(
        {
            "cold_start_failure": False,
            "seed_score": 0.2,
            "early_new_units_30s": 5,
        }
    )
    assert out["score_penalty"] == 0.8
    assert out["reason"] == "very_low_seed_score"
