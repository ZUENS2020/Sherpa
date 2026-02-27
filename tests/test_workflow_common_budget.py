from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import workflow_common


def test_parse_budget_value_preserves_zero():
    assert workflow_common.parse_budget_value(0, default=900) == 0
    assert workflow_common.parse_budget_value("0", default=900) == 0
    assert workflow_common.parse_budget_value(None, default=900) == 900


def test_enter_step_does_not_timeout_when_budget_is_unlimited():
    state = {"workflow_started_at": time.time() - 9999, "time_budget": 0, "step_count": 0, "max_steps": 10}
    out, stop = workflow_common.enter_step(state, "run")
    assert stop is False
    assert out["step_count"] == 1


def test_remaining_time_budget_is_large_when_unlimited():
    state = {"workflow_started_at": time.time() - 5000, "time_budget": 0}
    remaining = workflow_common.remaining_time_budget_sec(state)
    assert remaining > 1_000_000


def test_enter_step_stops_when_finite_budget_exceeded():
    state = {"workflow_started_at": time.time() - 3, "time_budget": 1, "step_count": 0, "max_steps": 10}
    out, stop = workflow_common.enter_step(state, "build")
    assert stop is True
    assert out["failed"] is True
    assert "time budget exceeded" in out["last_error"]
