from __future__ import annotations

import sys
import types
from pathlib import Path

import yaml
import pytest


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


def test_k8s_configmap_does_not_set_opencode_docker_image():
    configmap = yaml.safe_load((ROOT / "k8s" / "base" / "configmap.yaml").read_text(encoding="utf-8"))
    data = configmap["data"]

    assert "SHERPA_OPENCODE_DOCKER_IMAGE" not in data


def test_k8s_configmap_defaults_tmpdir_to_container_tmp():
    configmap = yaml.safe_load((ROOT / "k8s" / "base" / "configmap.yaml").read_text(encoding="utf-8"))
    data = configmap["data"]

    assert data["TMPDIR"] == "/tmp"


def test_k8s_manifest_applies_default_worker_resources():
    manifest_yaml = web_main._k8s_build_manifest(
        "job-test",
        {
            "job_id": "job-test",
            "repo_url": "https://github.com/madler/zlib.git",
            "model": "MiniMax-M2.5",
        },
    )
    manifest = yaml.safe_load(manifest_yaml)
    resources = manifest["spec"]["template"]["spec"]["containers"][0]["resources"]

    assert resources["requests"]["cpu"] == "500m"
    assert resources["requests"]["memory"] == "512Mi"
    assert resources["limits"]["cpu"] == "2"
    assert resources["limits"]["memory"] == "2Gi"


def test_k8s_manifest_allows_worker_resource_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERPA_K8S_JOB_CPU_REQUEST", "250m")
    monkeypatch.setenv("SHERPA_K8S_JOB_CPU_LIMIT", "1")
    monkeypatch.setenv("SHERPA_K8S_JOB_MEMORY_REQUEST", "768Mi")
    monkeypatch.setenv("SHERPA_K8S_JOB_MEMORY_LIMIT", "1536Mi")

    manifest_yaml = web_main._k8s_build_manifest(
        "job-test",
        {
            "job_id": "job-test",
            "repo_url": "https://github.com/madler/zlib.git",
            "model": "MiniMax-M2.5",
        },
    )
    manifest = yaml.safe_load(manifest_yaml)
    resources = manifest["spec"]["template"]["spec"]["containers"][0]["resources"]

    assert resources["requests"]["cpu"] == "250m"
    assert resources["requests"]["memory"] == "768Mi"
    assert resources["limits"]["cpu"] == "1"
    assert resources["limits"]["memory"] == "1536Mi"


def test_k8s_manifest_applies_non_root_security_context():
    manifest_yaml = web_main._k8s_build_manifest(
        "job-test",
        {
            "job_id": "job-test",
            "repo_url": "https://github.com/madler/zlib.git",
            "model": "MiniMax-M2.5",
        },
    )
    manifest = yaml.safe_load(manifest_yaml)
    pod_sc = manifest["spec"]["template"]["spec"]["securityContext"]
    container_sc = manifest["spec"]["template"]["spec"]["containers"][0]["securityContext"]

    assert pod_sc["seccompProfile"]["type"] == "RuntimeDefault"
    assert pod_sc["fsGroup"] == 10001
    assert pod_sc["fsGroupChangePolicy"] == "OnRootMismatch"
    assert container_sc["runAsNonRoot"] is True
    assert container_sc["runAsUser"] == 10001
    assert container_sc["runAsGroup"] == 10001
    assert container_sc["allowPrivilegeEscalation"] is False
    assert container_sc["capabilities"]["drop"] == ["ALL"]


def test_k8s_manifest_initializes_runtime_volume_permissions():
    manifest_yaml = web_main._k8s_build_manifest(
        "job-test",
        {
            "job_id": "job-test",
            "repo_url": "https://github.com/madler/zlib.git",
            "model": "MiniMax-M2.5",
        },
    )
    manifest = yaml.safe_load(manifest_yaml)
    init_container = manifest["spec"]["template"]["spec"]["initContainers"][0]

    assert init_container["name"] == "runtime-permissions"
    assert init_container["securityContext"]["runAsUser"] == 0
    assert init_container["securityContext"]["allowPrivilegeEscalation"] is False
    mounts = {item["mountPath"] for item in init_container["volumeMounts"]}
    assert "/app/config" in mounts
    assert "/shared/tmp" in mounts
