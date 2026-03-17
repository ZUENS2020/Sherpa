from __future__ import annotations

import base64
import json
import os
import subprocess
import traceback
import tempfile
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
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _write_text_resilient(path, body)


def _write_error(path: Path, text: str) -> None:
    _write_text_resilient(path, (text or "").strip() + "\n")


def _write_text_resilient(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(body, encoding="utf-8")
        return
    except PermissionError:
        # Root-owned stale files may exist from previous runs; if directory is
        # writable, removing the old inode allows a clean write.
        try:
            if path.exists():
                path.unlink()
            path.write_text(body, encoding="utf-8")
            return
        except Exception:
            pass

    fallback = Path(tempfile.gettempdir()) / "sherpa-k8s-worker-write-fallback.log"
    try:
        fallback.write_text(body, encoding="utf-8")
    except Exception:
        pass
    print(f"[k8s-worker] warn: failed to write {path}; wrote fallback={fallback}")


def _append_git_safe_directory_env(path_value: str) -> None:
    key = "safe.directory"
    val = (path_value or "").strip()
    if not val:
        return
    try:
        idx = int((os.environ.get("GIT_CONFIG_COUNT") or "0").strip())
        if idx < 0:
            idx = 0
    except Exception:
        idx = 0
    os.environ[f"GIT_CONFIG_KEY_{idx}"] = key
    os.environ[f"GIT_CONFIG_VALUE_{idx}"] = val
    os.environ["GIT_CONFIG_COUNT"] = str(idx + 1)


def _configure_git_safe_directory(payload: dict) -> None:
    # Apply both env-based and git-config based safe.directory so git subprocess
    # calls remain robust even when previous stages created root-owned repos.
    values: list[str] = ["*", "/shared/output"]
    for k in ("resume_repo_root", "re_workspace_root"):
        v = str(payload.get(k) or "").strip()
        if v:
            values.append(v)

    raw_extra = (os.environ.get("SHERPA_GIT_SAFE_DIRECTORY_EXTRA") or "").strip()
    if raw_extra:
        values.extend([x.strip() for x in raw_extra.split(",") if x.strip()])

    unique_values: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        unique_values.append(v)

    for v in unique_values:
        _append_git_safe_directory_env(v)

    for v in unique_values:
        try:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", v],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception:
            continue


def _repair_shared_permissions(payload: dict | None = None) -> None:
    if os.geteuid() != 0:
        return
    raw_targets = (
        os.environ.get(
            "SHERPA_K8S_PERMISSION_REPAIR_TARGETS",
            "/shared/output/_k8s_jobs,/shared/output/.opencode-home",
        )
        or ""
    ).strip()
    targets = [x.strip() for x in raw_targets.split(",") if x.strip()]

    resume_repo_root = str((payload or {}).get("resume_repo_root") or "").strip()
    if resume_repo_root:
        targets.extend(
            [
                resume_repo_root,
                f"{resume_repo_root}/.git",
                f"{resume_repo_root}/.git/sherpa-opencode",
            ]
        )

    # Keep ordering stable while removing duplicates.
    deduped_targets: list[str] = []
    seen: set[str] = set()
    for t in targets:
        if t in seen:
            continue
        seen.add(t)
        deduped_targets.append(t)
    targets = deduped_targets

    if not targets:
        return
    script = ["set -eu"]
    for t in targets:
        script.append(f'if [ -e "{t}" ]; then')
        script.append(f'  chown -R 10001:10001 "{t}" || true')
        script.append(f'  chmod -R a+rwX "{t}" || true')
        script.append("fi")
    try:
        subprocess.run(
            ["sh", "-lc", "\n".join(script)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        pass


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
    _configure_git_safe_directory(payload)

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
        try:
            _write_error(error_path, msg)
        except Exception:
            pass
        try:
            _write_json(
                result_path,
                {
                    "ok": False,
                    "job_id": job_id,
                    "error": str(e),
                },
            )
        except Exception:
            pass
        print(f"[k8s-worker] failed job_id={job_id}: {e}")
        return 1
    finally:
        _repair_shared_permissions(payload)


if __name__ == "__main__":
    raise SystemExit(main())
