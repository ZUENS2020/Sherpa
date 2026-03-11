from __future__ import annotations

import sys
import types
from pathlib import Path

import yaml


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


def test_k8s_manifest_does_not_force_empty_opencode_docker_image():
    manifest_yaml = web_main._k8s_build_manifest(
        "job-test",
        {
            "job_id": "job-test",
            "repo_url": "https://github.com/madler/zlib.git",
            "model": "MiniMax-M2.5",
        },
    )
    manifest = yaml.safe_load(manifest_yaml)
    env_items = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    env_names = {item["name"] for item in env_items}

    assert "SHERPA_K8S_WORKER_PAYLOAD_B64" in env_names
    assert "OPENCODE_MODEL" in env_names
    assert "SHERPA_OPENCODE_DOCKER_IMAGE" not in env_names
