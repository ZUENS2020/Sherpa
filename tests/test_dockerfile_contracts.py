from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_web_dockerfile_node_download_has_timeout_and_mirror_fallback() -> None:
    text = (ROOT / "docker" / "Dockerfile.web").read_text(encoding="utf-8")

    assert "ARG NODE_MIRROR_CANDIDATES=" in text
    assert 'for base in $NODE_MIRROR_CANDIDATES' in text
    assert "node_download_ok" in text
    assert "--connect-timeout 20" in text
    assert "--max-time 300" in text
    assert "--speed-time 60" in text
    assert "failed to download Node.js" in text
