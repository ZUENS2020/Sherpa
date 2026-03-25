from __future__ import annotations

import base64
import json
import os
import subprocess
import traceback
from pathlib import Path

from fuzz_relative_functions import fuzz_logic
from persistent_config import apply_config_to_env, load_config


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
    except Exception:
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
    except Exception:
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


def main() -> int:
    payload = _decode_payload()
    job_id = str(payload.get("job_id") or "")
    if job_id:
        os.environ["SHERPA_JOB_ID"] = job_id
    result_path = Path(str(payload.get("result_path") or f"/shared/output/_k8s_jobs/{job_id}/result.json")).expanduser()
    error_path = Path(str(payload.get("error_path") or f"/shared/output/_k8s_jobs/{job_id}/error.txt")).expanduser()

    print(f"[k8s-worker] start job_id={job_id} repo={payload.get('repo_url')}")
    try:
        opencode_defunct_count = _count_opencode_defunct_processes()
        opencode_defunct_threshold = _opencode_defunct_threshold()
        print(
            "[k8s-worker] diagnostics "
            f"opencode_defunct_count={opencode_defunct_count} "
            f"threshold={opencode_defunct_threshold}"
        )
        if opencode_defunct_threshold > 0 and opencode_defunct_count > opencode_defunct_threshold:
            raise RuntimeError(
                "opencode defunct process count exceeded threshold: "
                f"{opencode_defunct_count}>{opencode_defunct_threshold}"
            )

        # Rebuild runtime OpenCode config inside the worker container.
        # The path is injected via OPENCODE_CONFIG, but the file itself is
        # container-local and must be generated after secrets/env are loaded.
        apply_config_to_env(load_config())

        # Native runtime baseline: never execute inner Docker in k8s worker.
        effective_docker_image = None
        run_rss_limit_mb_override = str(payload.get("run_rss_limit_mb_override") or "").strip()
        if run_rss_limit_mb_override:
            os.environ["SHERPA_RUN_RSS_LIMIT_MB"] = run_rss_limit_mb_override
        run_parallel_fuzzers_override = str(payload.get("run_parallel_fuzzers_override") or "").strip()
        if run_parallel_fuzzers_override:
            os.environ["SHERPA_PARALLEL_FUZZERS"] = run_parallel_fuzzers_override
        run_unlimited_round_budget_sec = str(payload.get("run_unlimited_round_budget_sec") or "").strip()
        if run_unlimited_round_budget_sec:
            os.environ["SHERPA_RUN_UNLIMITED_ROUND_BUDGET_SEC"] = run_unlimited_round_budget_sec

        result = fuzz_logic(
            repo_url=str(payload.get("repo_url") or "").strip(),
            max_len=_parse_int_keep_zero(payload.get("max_len"), 0),
            time_budget=_parse_int_keep_zero(payload.get("time_budget"), 900),
            run_time_budget=_parse_int_keep_zero(payload.get("run_time_budget"), 900),
            email=(str(payload.get("email") or "").strip() or None),
            docker_image=effective_docker_image,
            ai_key_path=(Path(str(payload.get("ai_key_path") or "")).expanduser() if payload.get("ai_key_path") else None),
            oss_fuzz_dir=(str(payload.get("oss_fuzz_dir") or "").strip() or None),
            model=(str(payload.get("model") or "").strip() or None),
            resume_from_step=(str(payload.get("resume_from_step") or "").strip() or None),
            resume_repo_root=(str(payload.get("resume_repo_root") or "").strip() or None),
            stop_after_step=(str(payload.get("stop_after_step") or "").strip() or None),
            last_fuzzer=(str(payload.get("last_fuzzer") or "").strip() or None),
            last_crash_artifact=(str(payload.get("last_crash_artifact") or "").strip() or None),
            re_workspace_root=(str(payload.get("re_workspace_root") or "").strip() or None),
            coverage_loop_max_rounds=0,
            max_fix_rounds=0,
            same_error_max_retries=0,
            restart_to_plan_reason=(str(payload.get("restart_to_plan_reason") or "").strip() or None),
            restart_to_plan_stage=(str(payload.get("restart_to_plan_stage") or "").strip() or None),
            restart_to_plan_error_text=(str(payload.get("restart_to_plan_error_text") or "").strip() or None),
            restart_to_plan_report_path=(str(payload.get("restart_to_plan_report_path") or "").strip() or None),
        )
        out = {
            "ok": True,
            "job_id": job_id,
            "result": result,
        }
        _write_json(result_path, out)
        print(f"[k8s-worker] done job_id={job_id} result_path={result_path}")
        return 0
    except Exception as e:
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
