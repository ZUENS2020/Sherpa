from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_compose_web_uses_tmp_runtime_config_and_runtime_init():
    doc = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = doc["services"]
    web = services["sherpa-web"]

    assert "sherpa-runtime-init" in services
    assert web["environment"]["OPENCODE_CONFIG"] == "/tmp/sherpa-runtime/opencode.generated.json"
    assert web["environment"]["TMPDIR"] == "/tmp"
    assert web["depends_on"]["sherpa-runtime-init"]["condition"] == "service_completed_successfully"
    assert "/tmp" in web["tmpfs"]
    runtime_init_command = "\n".join(services["sherpa-runtime-init"]["entrypoint"])
    assert "find \"$$d\" -mindepth 1 -exec chown 10001:10001 {} +" in runtime_init_command
    assert "chmod 0777 \"$$d\"" in runtime_init_command
    assert "find \"$$d\" -mindepth 1 -exec chmod a+rwX {} +" in runtime_init_command


def test_compose_gateway_has_non_root_runtime_support():
    doc = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    gateway = doc["services"]["sherpa-gateway"]

    assert gateway["environment"]["CERT_DIR"] == "/tmp/nginx-certs"
    assert "NET_BIND_SERVICE" in gateway["cap_add"]
    assert "/var/cache/nginx" in gateway["tmpfs"]
    assert "/var/run" in gateway["tmpfs"]
    assert "/tmp" in gateway["tmpfs"]


def test_k8s_web_deployment_initializes_shared_volumes_for_non_root():
    doc = yaml.safe_load((ROOT / "k8s" / "base" / "web-deployment.yaml").read_text(encoding="utf-8"))
    init_container = doc["spec"]["template"]["spec"]["initContainers"][0]
    env_from = doc["spec"]["template"]["spec"]["containers"][0]["envFrom"]

    assert init_container["name"] == "runtime-permissions"
    assert init_container["securityContext"]["runAsUser"] == 0
    mounts = {item["mountPath"] for item in init_container["volumeMounts"]}
    assert "/app/config" in mounts
    assert "/shared/tmp" in mounts
    assert {"secretRef": {"name": "sherpa-runtime-proxy", "optional": True}} in env_from
    command = "\n".join(init_container["command"])
    assert "find \"$d\" -mindepth 1 -exec chown 10001:10001 {} +" in command
    assert "chmod 0777 \"$d\"" in command
    assert "find \"$d\" -mindepth 1 -exec chmod a+rwX {} +" in command
    assert "mkdir -p /shared/output/_k8s_jobs /shared/output/.opencode-home" in command
    assert "chown -R 10001:10001 /shared/output/_k8s_jobs /shared/output/.opencode-home" in command

def test_non_root_dockerfiles_define_user_and_non_root_home():
    gateway = (ROOT / "docker" / "Dockerfile.gateway").read_text(encoding="utf-8")
    fuzz = (ROOT / "docker" / "Dockerfile.fuzz").read_text(encoding="utf-8")
    fuzz_cpp = (ROOT / "docker" / "Dockerfile.fuzz-cpp").read_text(encoding="utf-8")
    fuzz_java = (ROOT / "docker" / "Dockerfile.fuzz-java").read_text(encoding="utf-8")
    opencode = (ROOT / "docker" / "Dockerfile.opencode").read_text(encoding="utf-8")

    assert "USER 101:101" in gateway
    assert "CERT_DIR=/tmp/nginx-certs" in gateway
    assert "USER 10001:10001" in fuzz
    assert "USER 10001:10001" in fuzz_cpp
    assert "USER 10001:10001" in fuzz_java
    assert "HOME=/home/fuzzer" in fuzz
    assert "HOME=/home/fuzzer" in fuzz_cpp
    assert "HOME=/home/fuzzer" in fuzz_java
    assert "USER 10001:10001" in opencode
    assert "HOME=/home/opencode" in opencode
