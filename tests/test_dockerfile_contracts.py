from __future__ import annotations

from pathlib import Path
import re


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


def test_opencode_tarball_download_fails_fast_across_images() -> None:
    for dockerfile in (
        ROOT / "docker" / "Dockerfile.web",
        ROOT / "docker" / "Dockerfile.opencode",
    ):
        text = dockerfile.read_text(encoding="utf-8")
        candidates = re.search(r'ARG NPM_REGISTRY_CANDIDATES="([^"]+)"', text)

        assert candidates is not None
        registries = candidates.group(1).split()
        assert registries[0] == "https://registry.npmjs.org"
        assert "https://registry.npmmirror.com" in registries
        assert "--retry 2" in text
        assert "--max-time 90" in text
        assert "--retry-max-time 240" in text
        assert "--max-time 900" not in text
        assert "--retry-max-time 1800" not in text
