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


def test_k8s_manifest_normalizes_opencode_model_value():
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
    env_map = {item["name"]: item["value"] for item in env_items if "value" in item}

    assert env_map["OPENCODE_MODEL"] == "minimax/MiniMax-M2.5"
    assert env_map["OPENAI_MODEL"] == "MiniMax-M2.5"


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
    assert resources["requests"]["memory"] == "4Gi"
    assert resources["limits"]["cpu"] == "2"
    assert resources["limits"]["memory"] == "64Gi"


def test_k8s_manifest_uses_optional_proxy_secret_env_from():
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
    env_from = manifest["spec"]["template"]["spec"]["containers"][0]["envFrom"]

    assert "SHERPA_NODE_IP" not in env_names
    assert "HTTP_PROXY" not in env_names
    assert "HTTPS_PROXY" not in env_names
    assert "ALL_PROXY" not in env_names
    assert {"secretRef": {"name": "sherpa-runtime-proxy", "optional": True}} in env_from


def test_k8s_manifest_allows_disabling_proxy_secret(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERPA_K8S_PROXY_SECRET_NAME", "")

    manifest_yaml = web_main._k8s_build_manifest(
        "job-test",
        {
            "job_id": "job-test",
            "repo_url": "https://github.com/madler/zlib.git",
            "model": "MiniMax-M2.5",
        },
    )
    manifest = yaml.safe_load(manifest_yaml)
    env_from = manifest["spec"]["template"]["spec"]["containers"][0]["envFrom"]

    assert {"secretRef": {"name": "sherpa-runtime-proxy", "optional": True}} not in env_from


def test_k8s_manifest_explicitly_injects_git_mirrors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(
        "SHERPA_GIT_MIRRORS",
        "https://ghfast.top/{url},https://ghproxy.net/{url},https://github.com",
    )

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
    env_map = {item["name"]: item["value"] for item in env_items if "value" in item}

    assert (
        env_map["SHERPA_GIT_MIRRORS"]
        == "https://ghfast.top/{url},https://ghproxy.net/{url},https://github.com"
    )


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


def test_k8s_manifest_build_stage_runs_worker_as_root_for_package_install():
    manifest_yaml = web_main._k8s_build_manifest(
        "job-test-build",
        {
            "job_id": "job-test-build",
            "repo_url": "https://github.com/madler/zlib.git",
            "model": "MiniMax-M2.5",
            "stop_after_step": "build",
        },
    )
    manifest = yaml.safe_load(manifest_yaml)
    container_sc = manifest["spec"]["template"]["spec"]["containers"][0]["securityContext"]

    assert container_sc["runAsNonRoot"] is False
    assert container_sc["runAsUser"] == 0
    assert container_sc["runAsGroup"] == 0
    assert container_sc["allowPrivilegeEscalation"] is False
    assert container_sc["capabilities"]["drop"] == ["ALL"]
    assert sorted(container_sc["capabilities"]["add"]) == ["SETGID", "SETUID"]


def test_k8s_manifest_build_stage_root_can_be_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SHERPA_K8S_BUILD_RUN_AS_ROOT", "0")
    manifest_yaml = web_main._k8s_build_manifest(
        "job-test-build-non-root",
        {
            "job_id": "job-test-build-non-root",
            "repo_url": "https://github.com/madler/zlib.git",
            "model": "MiniMax-M2.5",
            "stop_after_step": "build",
        },
    )
    manifest = yaml.safe_load(manifest_yaml)
    container_sc = manifest["spec"]["template"]["spec"]["containers"][0]["securityContext"]

    assert container_sc["runAsNonRoot"] is True
    assert container_sc["runAsUser"] == 10001
    assert container_sc["runAsGroup"] == 10001


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
    command = "\n".join(init_container["command"])
    assert "find \"$d\" -mindepth 1 -exec chown 10001:10001 {} +" in command
    assert "chmod 0777 \"$d\"" in command
    assert "find \"$d\" -mindepth 1 -exec chmod a+rwX {} +" in command
    assert "mkdir -p /shared/output/_k8s_jobs /shared/output/.opencode-home" in command
    assert "chown -R 10001:10001 /shared/output/_k8s_jobs /shared/output/.opencode-home" in command


def test_k8s_manifest_repairs_resume_repo_root_permissions():
    manifest_yaml = web_main._k8s_build_manifest(
        "job-test-build-resume",
        {
            "job_id": "job-test-build-resume",
            "repo_url": "https://github.com/libarchive/libarchive.git",
            "model": "MiniMax-M2.5",
            "stop_after_step": "build",
            "resume_repo_root": "/shared/output/libarchive-0dcd1388",
        },
    )
    manifest = yaml.safe_load(manifest_yaml)
    init_container = manifest["spec"]["template"]["spec"]["initContainers"][0]

    env_map = {item["name"]: item.get("value", "") for item in init_container.get("env", [])}
    assert env_map["SHERPA_RESUME_REPO_ROOT"] == "/shared/output/libarchive-0dcd1388"

    command = "\n".join(init_container["command"])
    assert "chown -R 10001:10001 \"${SHERPA_RESUME_REPO_ROOT}\"" in command
    assert "chmod -R a+rwX \"${SHERPA_RESUME_REPO_ROOT}\"" in command


def test_k8s_manifest_non_root_stage_can_reuse_resume_repo_root():
    manifest_yaml = web_main._k8s_build_manifest(
        "job-test-run-resume",
        {
            "job_id": "job-test-run-resume",
            "repo_url": "https://github.com/libarchive/libarchive.git",
            "model": "MiniMax-M2.5",
            "stop_after_step": "run",
            "resume_repo_root": "/shared/output/libarchive-0dcd1388",
        },
    )
    manifest = yaml.safe_load(manifest_yaml)
    container_sc = manifest["spec"]["template"]["spec"]["containers"][0]["securityContext"]
    init_container = manifest["spec"]["template"]["spec"]["initContainers"][0]

    assert container_sc["runAsNonRoot"] is True
    assert container_sc["runAsUser"] == 10001
    env_map = {item["name"]: item.get("value", "") for item in init_container.get("env", [])}
    assert env_map["SHERPA_RESUME_REPO_ROOT"] == "/shared/output/libarchive-0dcd1388"

    command = "\n".join(init_container["command"])
    assert "chown -R 10001:10001 \"${SHERPA_RESUME_REPO_ROOT}\"" in command
    assert "chmod -R a+rwX \"${SHERPA_RESUME_REPO_ROOT}\"" in command
