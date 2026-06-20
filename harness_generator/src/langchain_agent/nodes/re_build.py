"""Carved from workflow_graph.py - '_node_re_build' LangGraph node."""

from __future__ import annotations
from loguru import logger
import hashlib
import importlib
import json
import os
import re
import subprocess
import tempfile
import textwrap
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict, cast
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from persistent_config import load_config
import workflow_common as _wf_common
import workflow_observability as _wf_obs
import workflow_coverage_decision as _wf_coverage_decision
import workflow_normalization as _wf_norm
import workflow_target_scoring as _wf_target_scoring
import workflow_target_selection as _wf_target_selection
import workflow_summary as _wf_summary
import procedural_memory as _proc_mem
from coverage_replay import collect_per_input_frontier
from seed_families import seed_families_for_target as _seed_families_for_target
from workflow_context_store import (
    context_dir_for_repo_root,
    merge_result_into_contexts,
    read_context_docs,
    strip_meta,
    write_context_docs,
)
from fuzz_unharnessed_repo import (
    FuzzerRunResult,
    HarnessGeneratorError,
    NonOssFuzzHarnessGenerator,
    RepoSpec,
    extract_crash_stack_signature,
    snapshot_repo_text,
    write_patch_from_snapshot,
)

from wf_state import (
    FuzzWorkflowRuntimeState,
    _enter_step,
    _fmt_dt,
    _wf_log,
)
from workflow_helpers import (
    _apply_coverage_cc_wrapper_env,
    _extract_repair_top_trace,
    _inject_coverage_instrumentation,
    _re_restart_limit,
    _remaining_time_budget_sec,
    _write_repro_context,
)


def _node_re_build(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "re-build")
    if stop_now:
        return state

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> re-build")
    repo_root = gen.repo_root
    report_md = repo_root / "re_build_report.md"
    report_json = repo_root / "re_build_report.json"

    if not bool(state.get("crash_found")):
        out = {
            **state,
            "last_step": "re-build",
            "last_error": "",
            "re_build_done": True,
            "re_build_ok": False,
            "re_build_rc": 0,
            "message": "re-build skipped (no crash found)",
            "re_build_report_path": str(report_md),
            "re_build_json_path": str(report_json),
        }
        _wf_log(cast(dict[str, Any], out), f"<- re-build skip=no-crash dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    repo_url = str(state.get("repo_url") or "").strip()
    now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    payload: dict[str, Any] = {
        "timestamp": now_ts,
        "repo_url": repo_url,
        "fuzzer": str(state.get("last_fuzzer") or ""),
        "artifact": str(state.get("last_crash_artifact") or ""),
        "clone_repo_root": "",
        "clone_ok": False,
        "clone_rc": 1,
        "build_ok": False,
        "build_rc": 1,
        "error": "",
        "stdout_tail": "",
        "stderr_tail": "",
    }

    try:
        if not repo_url:
            raise HarnessGeneratorError("missing repo_url for re-build")

        repro_workspace = repo_root / ".repro_crash"
        repro_workspace.mkdir(parents=True, exist_ok=True)
        clone_root = repro_workspace / "workdir"
        if clone_root.exists():
            shutil.rmtree(clone_root, ignore_errors=True)

        # Reuse the same clone path as init so mirrors/proxy/retry behavior stays consistent.
        rem = _remaining_time_budget_sec(state, min_timeout=0)
        if rem <= 0:
            raise HarnessGeneratorError("re-build clone skipped: no remaining workflow budget")
        try:
            cloned_root = gen._clone_repo(RepoSpec(url=repo_url, workdir=clone_root))
        except Exception as clone_err:
            payload["clone_rc"] = 1
            payload["clone_ok"] = False
            payload["clone_repo_root"] = str(clone_root)
            payload["stderr_tail"] = str(clone_err)[-4000:]
            raise HarnessGeneratorError(f"re-build clone failed via init clone logic: {clone_err}")

        payload["clone_rc"] = 0
        payload["clone_ok"] = True
        payload["clone_repo_root"] = str(cloned_root)

        source_fuzz = repo_root / "fuzz"
        if not source_fuzz.is_dir():
            raise HarnessGeneratorError(f"run fuzz directory missing: {source_fuzz}")
        dest_fuzz = clone_root / "fuzz"
        if dest_fuzz.exists():
            shutil.rmtree(dest_fuzz, ignore_errors=True)
        shutil.copytree(
            source_fuzz,
            dest_fuzz,
            ignore=shutil.ignore_patterns(
                "build-work",   # CMake build dir (contains CMakeCache.txt with hardcoded paths)
                "CMakeFiles",   # CMake intermediate files
                "out",          # fuzzer output (corpus/crashes); re-run regenerates
                "__pycache__",
                "*.o",
                "*.a",
            ),
        )

        python_runner = "python3"
        try:
            python_runner = str(gen._python_runner() or "python3")
        except Exception:
            python_runner = "python3"

        build_cmd: list[str]
        build_cwd: Path
        if (clone_root / "fuzz" / "build.py").is_file():
            build_cmd = [python_runner, "build.py"]
            build_cwd = clone_root / "fuzz"
            # The run-stage rebuild produces the binary that is actually fuzzed;
            # instrument its library + harness build deterministically so the
            # fuzzed binary is not blind to the target library.
            _inject_coverage_instrumentation(str(clone_root / "fuzz" / "build.py"), state)
        elif (clone_root / "fuzz" / "build.sh").is_file():
            build_cmd = ["bash", "build.sh"]
            build_cwd = clone_root / "fuzz"
        else:
            raise HarnessGeneratorError("no fuzz/build.py or fuzz/build.sh found in cloned repo")

        rem = _remaining_time_budget_sec(state, min_timeout=15)
        build_timeout = max(30, min(rem, 1800))
        build_env = os.environ.copy()
        if hasattr(gen, "_compose_vcpkg_runtime_env"):
            try:
                build_env = gen._compose_vcpkg_runtime_env(build_env, repo_root=clone_root)  # type: ignore[attr-defined]
            except Exception:
                pass
        # Instrument the whole library (not just the harness) for libFuzzer.
        build_env = _apply_coverage_cc_wrapper_env(build_env, clone_root)
        if hasattr(gen, "_run_cmd"):
            rc, out, err = gen._run_cmd(  # type: ignore[attr-defined]
                build_cmd,
                cwd=build_cwd,
                env=build_env,
                timeout=build_timeout,
                idle_timeout=0,
            )
            payload["build_rc"] = int(rc)
            payload["build_ok"] = int(rc) == 0
            if int(rc) != 0:
                payload["stdout_tail"] = (out or "")[-4000:]
                payload["stderr_tail"] = (err or "")[-4000:]
                raise HarnessGeneratorError(f"re-build build failed (rc={rc})")
        else:
            build = subprocess.run(
                build_cmd,
                cwd=build_cwd,
                capture_output=True,
                text=True,
                timeout=build_timeout,
                env=build_env,
            )
            payload["build_rc"] = int(build.returncode)
            payload["build_ok"] = build.returncode == 0
            if build.returncode != 0:
                payload["stdout_tail"] = (build.stdout or "")[-4000:]
                payload["stderr_tail"] = (build.stderr or "")[-4000:]
                raise HarnessGeneratorError(f"re-build build failed (rc={build.returncode})")
    except Exception as e:
        payload["error"] = str(e)

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# Re-Build Report",
        "",
        f"- timestamp: {payload['timestamp']}",
        f"- repo_url: {payload['repo_url']}",
        f"- clone_ok: {payload['clone_ok']} (rc={payload['clone_rc']})",
        f"- build_ok: {payload['build_ok']} (rc={payload['build_rc']})",
        "",
    ]
    if payload["error"]:
        md_lines.extend(["## Error", "", str(payload["error"]), ""])
    if payload["stdout_tail"]:
        md_lines.extend(["## STDOUT (tail)", "", "```text", str(payload["stdout_tail"]), "```", ""])
    if payload["stderr_tail"]:
        md_lines.extend(["## STDERR (tail)", "", "```text", str(payload["stderr_tail"]), "```", ""])
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    re_build_ok = bool(payload["build_ok"])
    restart_reason = ""
    restart_error = ""
    restart_report = ""
    restart_stage = ""
    restart_count = int(state.get("restart_to_plan_count") or 0)
    if not re_build_ok:
        restart_reason = "re_build_failed"
        restart_stage = "re-build"
        restart_error = str(payload.get("error") or payload.get("stderr_tail") or payload.get("stdout_tail") or "")[:4096]
        restart_report = str(report_md)
        restart_count += 1
    restart_limit = _re_restart_limit()
    restart_exceeded = (not re_build_ok) and restart_count > restart_limit
    if re_build_ok:
        _write_repro_context(
            repo_root,
            repo_url=repo_url,
            last_fuzzer=str(state.get("last_fuzzer") or ""),
            last_crash_artifact=str(state.get("last_crash_artifact") or ""),
            crash_signature=str(state.get("crash_signature") or ""),
            re_workspace_root=str(payload.get("clone_repo_root") or ""),
        )

    out = {
        **state,
        "last_step": "re-build",
        "last_error": "" if re_build_ok else restart_error,
        "re_build_done": True,
        "re_build_ok": re_build_ok,
        "re_build_rc": int(payload["build_rc"]),
        "re_build_report_path": str(report_md),
        "re_build_json_path": str(report_json),
        "re_workspace_root": str(payload.get("clone_repo_root") or ""),
        "restart_to_plan": not re_build_ok,
        "restart_to_plan_reason": restart_reason,
        "restart_to_plan_stage": restart_stage,
        "restart_to_plan_error_text": restart_error,
        "restart_to_plan_report_path": restart_report,
        "restart_to_plan_count": restart_count,
        "failed": bool(state.get("failed")) or restart_exceeded,
        "run_terminal_reason": "re_restart_limit_exceeded" if restart_exceeded else str(state.get("run_terminal_reason") or ""),
        "message": "re-build validated" if re_build_ok else "re-build failed",
        "repair_mode": (not re_build_ok),
        "repair_origin_stage": "crash" if not re_build_ok else "",
        "repair_error_kind": "re_build_failed" if not re_build_ok else "",
        "repair_error_code": restart_reason if not re_build_ok else "",
        "repair_signature": str(state.get("crash_signature") or "")[:12] if not re_build_ok else "",
        "repair_stdout_tail": str(payload.get("stdout_tail") or "") if not re_build_ok else "",
        "repair_stderr_tail": str(payload.get("stderr_tail") or "") if not re_build_ok else "",
        "repair_attempt_index": (int(state.get("repair_attempt_index") or 0) + 1) if not re_build_ok else 0,
        "repair_strategy_force_change": False,
        "repair_error_digest": (
            {
                "error_code": restart_reason,
                "error_kind": "re_build_failed",
                "signature": str(state.get("crash_signature") or "")[:12],
                "failing_files": [],
                "symbols": [],
                "first_seen": int(time.time()),
                "latest_seen": int(time.time()),
                "top_trace": _extract_repair_top_trace(
                    restart_error,
                    str(payload.get("stdout_tail") or ""),
                    str(payload.get("stderr_tail") or ""),
                ),
            }
            if not re_build_ok
            else {}
        ),
        "repair_recent_attempts": (
            (list(state.get("repair_recent_attempts") or []) + [{
                "step": "re-build",
                "origin": "crash",
                "error_kind": "re_build_failed",
                "error_code": restart_reason,
                "signature": str(state.get("crash_signature") or "")[:12],
                "attempt_index": int(state.get("repair_attempt_index") or 0) + 1,
                "message": restart_error[:512],
            }])[-5:]
            if not re_build_ok
            else []
        ),
    }
    if restart_exceeded:
        out["last_error"] = f"re failed and restart-to-plan limit exceeded ({restart_limit})"
    _wf_log(
        cast(dict[str, Any], out),
        (
            "<- re-build "
            f"ok={re_build_ok} clone_rc={payload['clone_rc']} build_rc={payload['build_rc']} "
            f"dt={_fmt_dt(time.perf_counter()-t0)}"
        ),
    )
    return out
