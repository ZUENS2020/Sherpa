from __future__ import annotations

import base64
import json
import os
import subprocess
import traceback
from pathlib import Path

from fuzz_relative_functions import fuzz_logic
from persistent_config import apply_config_to_env, load_config
from errors import SherpaError
from workflow_context_store import (
    context_dir_for_repo_root,
    read_context_docs,
    strip_meta,
)


def _decode_payload() -> dict:
    raw = (os.environ.get("SHERPA_K8S_WORKER_PAYLOAD_B64", "") or "").strip()
    if not raw:
        raise RuntimeError("SHERPA_K8S_WORKER_PAYLOAD_B64 is required")
    data = base64.b64decode(raw.encode("ascii"))
    payload = json.loads(data.decode("utf-8", errors="replace"))
    if not isinstance(payload, dict):
        raise RuntimeError("worker payload must be a JSON object")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_error(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text((text or "").strip() + "\n", encoding="utf-8")


def _parse_int_keep_zero(value: object, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return int(default)
        return int(text)
    return int(value)


def _opencode_defunct_threshold() -> int:
    raw = (os.environ.get("SHERPA_OPENCODE_DEFUNCT_THRESHOLD") or "3").strip()
    try:
        return max(0, min(int(raw), 200))
    except (ValueError, TypeError):
        return 3


def _count_opencode_defunct_processes() -> int:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "stat=,args="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    if int(proc.returncode or 0) != 0:
        return 0
    count = 0
    for raw_line in str(proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        stat = parts[0] if parts else ""
        cmd = parts[1] if len(parts) > 1 else ""
        cmd_l = cmd.lower()
        if "opencode" not in cmd_l:
            continue
        if "<defunct>" in cmd_l or stat.startswith("Z"):
            count += 1
    return count


def _reap_any_dead_children(max_rounds: int = 16) -> int:
    if os.name == "nt":
        return 0
    reaped = 0
    rounds = max(1, min(int(max_rounds), 256))
    for _ in range(rounds):
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        except OSError:
            break
        if pid == 0:
            break
        reaped += 1
    return reaped


def _merge_opencode_mcp_servers(companion_url: str) -> None:
    url = str(companion_url or "").strip()
    if not url:
        return
    current_raw = (os.environ.get("SHERPA_OPENCODE_MCP_SERVERS_JSON") or "").strip()
    payload: dict[str, object] = {}
    if current_raw:
        try:
            parsed = json.loads(current_raw)
            if isinstance(parsed, dict):
                payload = dict(parsed)
        except (json.JSONDecodeError, ValueError):
            payload = {}
    payload["promefuzz"] = {
        "type": "remote",
        "url": url,
        "enabled": True,
    }
    os.environ["SHERPA_OPENCODE_MCP_SERVERS_JSON"] = json.dumps(payload, ensure_ascii=False)
    os.environ["SHERPA_OPENCODE_MCP_URL"] = url


def _resolve_analysis_companion_url(payload: dict, job_id: str) -> str:
    enabled_raw = (os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_ENABLED", "1") or "").strip().lower()
    if enabled_raw not in {"1", "true", "yes", "on"}:
        return ""
    ready = payload.get("analysis_companion_ready")
    if ready is not None and not bool(ready):
        return ""
    explicit = str(payload.get("analysis_companion_url") or "").strip()
    if explicit:
        return explicit
    jid = str(job_id or "").strip()
    if not jid:
        return ""
    ns = (os.environ.get("SHERPA_K8S_NAMESPACE") or "sherpa-dev").strip() or "sherpa-dev"
    port_raw = (os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_PORT") or "18080").strip()
    path_raw = (os.environ.get("SHERPA_K8S_ANALYSIS_COMPANION_MCP_PATH") or "/mcp").strip()
    try:
        port = max(1, min(int(port_raw), 65535))
    except (ValueError, TypeError):
        port = 18080
    path = path_raw if path_raw.startswith("/") else f"/{path_raw}"
    svc = f"sherpa-promefuzz-{jid[:10]}"
    return f"http://{svc}.{ns}.svc.cluster.local:{port}{path}"


def main() -> int:
    payload = _decode_payload()
    job_id = str(payload.get("job_id") or "")
    if job_id:
        os.environ["SHERPA_JOB_ID"] = job_id
        os.environ["SHERPA_CURRENT_JOB_ID"] = job_id
    result_path = Path(str(payload.get("result_path") or f"/shared/output/_k8s_jobs/{job_id}/result.json")).expanduser()
    error_path = Path(str(payload.get("error_path") or f"/shared/output/_k8s_jobs/{job_id}/error.txt")).expanduser()

    print(f"[k8s-worker] start job_id={job_id} repo={payload.get('repo_url')}")
    try:
        opencode_defunct_count_before = _count_opencode_defunct_processes()
        opencode_defunct_threshold = _opencode_defunct_threshold()
        opencode_defunct_reaped = _reap_any_dead_children()
        opencode_defunct_count_after = _count_opencode_defunct_processes()
        print(
            "[k8s-worker] diagnostics "
            f"opencode_defunct_count_before={opencode_defunct_count_before} "
            f"opencode_defunct_reaped={opencode_defunct_reaped} "
            f"opencode_defunct_count_after={opencode_defunct_count_after} "
            f"threshold={opencode_defunct_threshold}"
        )
        if opencode_defunct_threshold > 0 and opencode_defunct_count_after > opencode_defunct_threshold:
            raise RuntimeError(
                "opencode defunct process count exceeded threshold: "
                f"{opencode_defunct_count_after}>{opencode_defunct_threshold}"
            )

        # Rebuild runtime OpenCode config inside the worker container.
        # The path is injected via OPENCODE_CONFIG, but the file itself is
        # container-local and must be generated after secrets/env are loaded.
        _merge_opencode_mcp_servers(_resolve_analysis_companion_url(payload, job_id))
        apply_config_to_env(load_config())

        # Native runtime baseline: never execute inner Docker in k8s worker.
        effective_docker_image = None
        context_dir = str(payload.get("context_dir") or "").strip()
        if not context_dir:
            context_dir = str(
                context_dir_for_repo_root(str(payload.get("resume_repo_root") or "").strip())
                or ""
            ).strip()
        control_doc, _workflow_doc = read_context_docs(
            context_dir or None,
            job_id=job_id,
        )
        control_ctx = strip_meta(control_doc)

        run_rss_limit_mb_override = str(control_ctx.get("run_rss_limit_mb_override") or "").strip()
        if run_rss_limit_mb_override:
            os.environ["SHERPA_RUN_RSS_LIMIT_MB"] = run_rss_limit_mb_override
        run_parallel_fuzzers_override = str(control_ctx.get("run_parallel_fuzzers_override") or "").strip()
        if run_parallel_fuzzers_override:
            os.environ["SHERPA_PARALLEL_FUZZERS"] = run_parallel_fuzzers_override
        run_unlimited_round_budget_sec = str(control_ctx.get("run_timeout_budget_sec_override") or "").strip()
        if not run_unlimited_round_budget_sec:
            run_unlimited_round_budget_sec = str(payload.get("run_unlimited_round_budget_sec") or "").strip()
        if run_unlimited_round_budget_sec:
            os.environ["SHERPA_RUN_UNLIMITED_ROUND_BUDGET_SEC"] = run_unlimited_round_budget_sec

        result = fuzz_logic(
            repo_url=str(payload.get("repo_url") or "").strip(),
            max_len=_parse_int_keep_zero(payload.get("max_len"), 0),
            time_budget=_parse_int_keep_zero(payload.get("time_budget"), 900),
            run_time_budget=_parse_int_keep_zero(payload.get("run_time_budget"), 900),
            coverage_loop_max_rounds=_parse_int_keep_zero(payload.get("coverage_loop_max_rounds"), 0),
            max_fix_rounds=_parse_int_keep_zero(payload.get("max_fix_rounds"), 0),
            same_error_max_retries=_parse_int_keep_zero(payload.get("same_error_max_retries"), 0),
            email=(str(payload.get("email") or "").strip() or None),
            docker_image=effective_docker_image,
            ai_key_path=(Path(str(payload.get("ai_key_path") or "")).expanduser() if payload.get("ai_key_path") else None),
            oss_fuzz_dir=(str(payload.get("oss_fuzz_dir") or "").strip() or None),
            model=(str(payload.get("model") or "").strip() or None),
            resume_from_step=(str(payload.get("resume_from_step") or "").strip() or None),
            resume_repo_root=(str(payload.get("resume_repo_root") or "").strip() or None),
            stop_after_step=(str(payload.get("stop_after_step") or "").strip() or None),
            context_dir=(context_dir or None),
        )
        out = {
            "ok": True,
            "job_id": job_id,
            "result": result,
        }
        _write_json(result_path, out)
        print(f"[k8s-worker] done job_id={job_id} result_path={result_path}")
        return 0
    except (SherpaError, ValueError, OSError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as e:
        tb = traceback.format_exc()
        msg = f"{e}\n{tb}"
        _write_error(error_path, msg)
        _write_json(
            result_path,
            {
                "ok": False,
                "job_id": job_id,
                "error": str(e),
            },
        )
        print(f"[k8s-worker] failed job_id={job_id}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
