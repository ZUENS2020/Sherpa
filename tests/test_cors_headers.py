from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main as web_main


def test_api_cors_allows_any_origin_for_simple_and_preflight_requests(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("DATABASE_URL", "postgresql://sherpa:sherpa@127.0.0.1:55432/sherpa")
    monkeypatch.setattr(web_main, "_init_job_store", lambda: None)
    with TestClient(web_main.app) as client:
        get_resp = client.get("/api/config", headers={"Origin": "http://example-frontend.local"})
        preflight_resp = client.options(
            "/api/config",
            headers={
                "Origin": "http://example-frontend.local",
                "Access-Control-Request-Method": "PUT",
                "Access-Control-Request-Headers": "content-type,authorization",
            },
        )

    assert get_resp.status_code == 200
    assert get_resp.headers.get("access-control-allow-origin") == "*"

    assert preflight_resp.status_code == 200
    assert preflight_resp.headers.get("access-control-allow-origin") == "*"
    allow_methods = str(preflight_resp.headers.get("access-control-allow-methods") or "").upper()
    assert "PUT" in allow_methods
