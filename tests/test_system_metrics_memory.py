from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


if "langchain_openai" not in sys.modules:
    mod = types.ModuleType("langchain_openai")

    class _DummyChatOpenAI:  # pragma: no cover
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mod.ChatOpenAI = _DummyChatOpenAI
    sys.modules["langchain_openai"] = mod


import main as web_main


def test_system_status_exposes_memory_status(monkeypatch):
    monkeypatch.setattr(
        web_main,
        "_memory_status",
        lambda: {
            "process_rss_bytes": 123,
            "cgroup_current_bytes": 456,
            "cgroup_limit_bytes": 789,
            "cgroup_usage_ratio": 0.57,
            "oom_kill_count": 2,
            "pressure": "normal",
        },
    )

    data = web_main._system_status()

    assert data["memory"]["process_rss_bytes"] == 123
    assert data["memory"]["cgroup_current_bytes"] == 456
    assert data["memory"]["cgroup_limit_bytes"] == 789
    assert data["memory"]["cgroup_usage_ratio"] == 0.57
    assert data["memory"]["oom_kill_count"] == 2
    assert data["memory"]["pressure"] == "normal"


def test_metrics_payload_exposes_memory_metrics(monkeypatch):
    monkeypatch.setattr(
        web_main,
        "_memory_status",
        lambda: {
            "process_rss_bytes": 111,
            "cgroup_current_bytes": 222,
            "cgroup_limit_bytes": 333,
            "cgroup_usage_ratio": 0.666666,
            "oom_kill_count": 4,
            "pressure": "elevated",
        },
    )

    body = web_main._metrics_payload()

    assert "sherpa_process_resident_memory_bytes 111" in body
    assert "sherpa_cgroup_memory_current_bytes 222" in body
    assert "sherpa_cgroup_memory_limit_bytes 333" in body
    assert "sherpa_cgroup_memory_usage_ratio 0.666666" in body
    assert "sherpa_cgroup_memory_oom_kill_total 4" in body
