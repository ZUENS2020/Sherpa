from __future__ import annotations

import base64
import json
import os
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


def main() -> int:
    payload = _decode_payload()
    job_id = str(payload.get("job_id") or "")
    result_path = Path(str(payload.get("result_path") or f"/shared/output/_k8s_jobs/{job_id}/result.json")).expanduser()
    error_path = Path(str(payload.get("error_path") or f"/shared/output/_k8s_jobs/{job_id}/error.txt")).expanduser()

    print(f"[k8s-worker] start job_id={job_id} repo={payload.get('repo_url')}")
    try:
        # Rebuild runtime OpenCode config inside the worker container.
        # The path is injected via OPENCODE_CONFIG, but the file itself is
        # container-local and must be generated after secrets/env are loaded.
        apply_config_to_env(load_config())

        # Native runtime baseline: never execute inner Docker in k8s worker.
        effective_docker_image = None

        result = fuzz_logic(
            repo_url=str(payload.get("repo_url") or "").strip(),
            max_len=_parse_int_keep_zero(payload.get("max_len"), 1024),
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
            coverage_loop_max_rounds=_parse_int_keep_zero(payload.get("coverage_loop_max_rounds"), 3),
            max_fix_rounds=_parse_int_keep_zero(payload.get("max_fix_rounds"), 3),
            same_error_max_retries=_parse_int_keep_zero(payload.get("same_error_max_retries"), 1),
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
