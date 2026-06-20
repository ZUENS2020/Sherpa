"""Carved from workflow_graph.py - '_node_re_run' LangGraph node."""

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
    _re_restart_limit,
    _read_repro_context,
    _remaining_time_budget_sec,
    _write_repro_context,
)


def _node_re_run(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "re-run")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> re-run")
    _wf_log(cast(dict[str, Any], state), "re-run: reusing run-stage corpus from fuzz/corpus; no new seeds will be generated")

    repo_root = gen.repo_root
    report_md = repo_root / "re_run_report.md"
    report_json = repo_root / "re_run_report.json"
    last_fuzzer = str(state.get("last_fuzzer") or "").strip()
    last_artifact = str(state.get("last_crash_artifact") or "").strip()
    workspace_root = str(state.get("re_workspace_root") or "").strip() or str((repo_root / ".repro_crash" / "workdir"))
    artifact_path = Path(last_artifact) if last_artifact else None

    def _recover_artifact_path() -> tuple[str, Path | None]:
        recovered = last_artifact
        if not recovered:
            repro_doc = _read_repro_context(repo_root)
            if isinstance(repro_doc, dict):
                recovered = str(repro_doc.get("last_crash_artifact") or "").strip()
        if not recovered and (repo_root / "re_build_report.json").is_file():
            try:
                re_build_doc = json.loads((repo_root / "re_build_report.json").read_text(encoding="utf-8", errors="replace"))
                if isinstance(re_build_doc, dict):
                    recovered = str(re_build_doc.get("artifact") or "").strip()
            except Exception:
                pass
        if not recovered and (repo_root / "run_summary.json").is_file():
            try:
                summary_doc = json.loads((repo_root / "run_summary.json").read_text(encoding="utf-8", errors="replace"))
                if isinstance(summary_doc, dict):
                    recovered = str(summary_doc.get("last_crash_artifact") or "").strip()
            except Exception:
                pass
        if not recovered:
            artifacts_dir = repo_root / "fuzz" / "out" / "artifacts"
            if artifacts_dir.is_dir():
                candidates: list[Path] = []
                for p in artifacts_dir.iterdir():
                    if not p.is_file():
                        continue
                    name = p.name.lower()
                    if name.startswith("crash-") or "crash" in name:
                        candidates.append(p)
                if not candidates:
                    for p in artifacts_dir.iterdir():
                        if p.is_file():
                            candidates.append(p)
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    recovered = str(candidates[0])
        return recovered, (Path(recovered) if recovered else None)
    def _rebuild_workspace_from_init_clone() -> Path:
        repo_url = str(state.get("repo_url") or "").strip()
        if not repo_url:
            raise HarnessGeneratorError("missing repo_url for re-run workspace rebuild")
        repro_workspace = repo_root / ".repro_crash"
        repro_workspace.mkdir(parents=True, exist_ok=True)
        clone_root = repro_workspace / "workdir"
        if clone_root.exists():
            shutil.rmtree(clone_root, ignore_errors=True)

        rem = _remaining_time_budget_sec(state, min_timeout=15)
        if rem <= 0:
            raise HarnessGeneratorError("re-run workspace rebuild skipped: no remaining workflow budget")
        clone_result = gen._clone_repo(RepoSpec(url=repo_url, workdir=clone_root))
        clone_root = Path(clone_result).expanduser().resolve()
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
        elif (clone_root / "fuzz" / "build.sh").is_file():
            build_cmd = ["bash", "build.sh"]
            build_cwd = clone_root / "fuzz"
        else:
            raise HarnessGeneratorError("no fuzz/build.py or fuzz/build.sh found in re-run workspace rebuild")

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
            if int(rc) != 0:
                err_tail = ((err or "") + "\n" + (out or ""))[-1200:]
                raise HarnessGeneratorError(f"re-run workspace rebuild build failed (rc={rc}): {err_tail}")
        else:
            build = subprocess.run(
                build_cmd,
                cwd=build_cwd,
                capture_output=True,
                text=True,
                timeout=build_timeout,
                env=build_env,
            )
            if build.returncode != 0:
                err_tail = ((build.stderr or "") + "\n" + (build.stdout or ""))[-1200:]
                raise HarnessGeneratorError(f"re-run workspace rebuild build failed (rc={build.returncode}): {err_tail}")
        return clone_root

    def _guess_fuzzer_from_workspace(workdir: Path) -> str:
        out_dir = workdir / "fuzz" / "out"
        if not out_dir.is_dir():
            return ""
        candidates: list[Path] = []
        for p in out_dir.iterdir():
            if not p.is_file():
                continue
            name = p.name
            if name.startswith("."):
                continue
            if name.startswith(("crash-", "timeout-", "slow-unit-")):
                continue
            if name.endswith((".md", ".json", ".txt", ".log", ".patch", ".py")):
                continue
            if os.access(p, os.X_OK):
                candidates.append(p)
        if len(candidates) == 1:
            return candidates[0].name
        # Prefer common fuzzer naming if multiple binaries are present.
        named = [p for p in candidates if "fuzz" in p.name.lower()]
        if len(named) == 1:
            return named[0].name
        return ""

    now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    payload: dict[str, Any] = {
        "timestamp": now_ts,
        "fuzzer": last_fuzzer,
        "artifact": last_artifact,
        "workspace_root": workspace_root,
        "reproduce_ok": False,
        "reproduce_rc": 1,
        "error": "",
        "stdout_tail": "",
        "stderr_tail": "",
    }
    try:
        workdir = Path(workspace_root)
        if not last_fuzzer or not last_artifact or not str(state.get("re_workspace_root") or "").strip():
            repro_doc = _read_repro_context(repo_root)
            if isinstance(repro_doc, dict):
                if not last_fuzzer:
                    last_fuzzer = str(repro_doc.get("last_fuzzer") or "").strip()
                    payload["fuzzer"] = last_fuzzer
                if not last_artifact:
                    last_artifact = str(repro_doc.get("last_crash_artifact") or "").strip()
                    payload["artifact"] = last_artifact
                    if last_artifact:
                        artifact_path = Path(last_artifact)
                restored_workspace = str(repro_doc.get("re_workspace_root") or "").strip()
                if restored_workspace and not workdir.is_dir():
                    workspace_root = restored_workspace
                    payload["workspace_root"] = restored_workspace
                    workdir = Path(restored_workspace)
        if not workdir.is_dir():
            _wf_log(cast(dict[str, Any], state), f"re-run: workspace missing, attempting rebuild via init clone logic: {workdir}")
            workdir = _rebuild_workspace_from_init_clone()
            workspace_root = str(workdir)
            payload["workspace_root"] = workspace_root
            _write_repro_context(
                repo_root,
                repo_url=str(state.get("repo_url") or ""),
                re_workspace_root=workspace_root,
            )
        if (not last_fuzzer or not last_artifact) and (repo_root / "re_build_report.json").is_file():
            try:
                re_build_doc = json.loads((repo_root / "re_build_report.json").read_text(encoding="utf-8", errors="replace"))
                if isinstance(re_build_doc, dict):
                    if not last_fuzzer:
                        last_fuzzer = str(re_build_doc.get("fuzzer") or "").strip()
                        payload["fuzzer"] = last_fuzzer
                    if not last_artifact:
                        last_artifact = str(re_build_doc.get("artifact") or "").strip()
                        payload["artifact"] = last_artifact
                        if last_artifact:
                            artifact_path = Path(last_artifact)
            except Exception:
                pass
        if artifact_path is None or not artifact_path.is_file():
            recovered_artifact, recovered_path = _recover_artifact_path()
            if recovered_artifact:
                last_artifact = recovered_artifact
                artifact_path = recovered_path
                payload["artifact"] = recovered_artifact
        if not last_fuzzer:
            # Stage resume can occasionally lose last_fuzzer in state; recover from workspace.
            last_fuzzer = _guess_fuzzer_from_workspace(workdir)
            payload["fuzzer"] = last_fuzzer
        if not last_fuzzer:
            _wf_log(cast(dict[str, Any], state), "re-run: last_fuzzer missing, attempting workspace rebuild before failing")
            workdir = _rebuild_workspace_from_init_clone()
            workspace_root = str(workdir)
            payload["workspace_root"] = workspace_root
            last_fuzzer = _guess_fuzzer_from_workspace(workdir)
            payload["fuzzer"] = last_fuzzer
        if not last_fuzzer:
            raise HarnessGeneratorError("missing last_fuzzer for re-run after workspace rebuild")
        if artifact_path is None or not artifact_path.is_file():
            recovered_artifact, recovered_path = _recover_artifact_path()
            if recovered_artifact:
                last_artifact = recovered_artifact
                artifact_path = recovered_path
                payload["artifact"] = recovered_artifact
        if artifact_path is None or not artifact_path.is_file():
            raise HarnessGeneratorError(f"crash artifact not found: {last_artifact}")
        fuzzer_bin = workdir / "fuzz" / "out" / last_fuzzer
        if not fuzzer_bin.is_file():
            _wf_log(cast(dict[str, Any], state), f"re-run: fuzzer binary missing, attempting workspace rebuild: {fuzzer_bin}")
            workdir = _rebuild_workspace_from_init_clone()
            workspace_root = str(workdir)
            payload["workspace_root"] = workspace_root
            if not last_fuzzer:
                last_fuzzer = _guess_fuzzer_from_workspace(workdir)
                payload["fuzzer"] = last_fuzzer
            fuzzer_bin = workdir / "fuzz" / "out" / last_fuzzer
            if not fuzzer_bin.is_file():
                raise HarnessGeneratorError(f"re-run fuzzer binary not found after workspace rebuild: {fuzzer_bin}")
        rem = _remaining_time_budget_sec(state, min_timeout=15)
        repro_timeout = max(20, min(rem, 180))
        repro_env = os.environ.copy()
        if hasattr(gen, "_compose_vcpkg_runtime_env"):
            try:
                repro_env = gen._compose_vcpkg_runtime_env(repro_env, repo_root=workdir)  # type: ignore[attr-defined]
            except Exception:
                pass
        repro_cmd = [str(fuzzer_bin), "-runs=1", str(artifact_path)]
        if hasattr(gen, "_run_cmd"):
            rc, out, err = gen._run_cmd(  # type: ignore[attr-defined]
                repro_cmd,
                cwd=workdir,
                env=repro_env,
                timeout=repro_timeout,
                idle_timeout=0,
            )
            payload["reproduce_rc"] = int(rc)
            payload["reproduce_ok"] = int(rc) != 0
            payload["stdout_tail"] = (out or "")[-4000:]
            payload["stderr_tail"] = (err or "")[-4000:]
        else:
            repro = subprocess.run(
                repro_cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=repro_timeout,
                env=repro_env,
            )
            payload["reproduce_rc"] = int(repro.returncode)
            payload["reproduce_ok"] = repro.returncode != 0
            payload["stdout_tail"] = (repro.stdout or "")[-4000:]
            payload["stderr_tail"] = (repro.stderr or "")[-4000:]
        _write_repro_context(
            repo_root,
            repo_url=str(state.get("repo_url") or ""),
            last_fuzzer=last_fuzzer,
            last_crash_artifact=last_artifact,
            crash_signature=str(state.get("crash_signature") or ""),
            re_workspace_root=workspace_root,
        )
    except Exception as e:
        payload["error"] = str(e)

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_lines = [
        "# Re-Run Report",
        "",
        f"- timestamp: {payload['timestamp']}",
        f"- fuzzer: {payload['fuzzer']}",
        f"- artifact: {payload['artifact']}",
        f"- workspace_root: {payload['workspace_root']}",
        f"- reproduce_ok: {payload['reproduce_ok']} (rc={payload['reproduce_rc']})",
        "",
    ]
    if payload["error"]:
        md_lines.extend(["## Error", "", str(payload["error"]), ""])
    if payload["stdout_tail"]:
        md_lines.extend(["## STDOUT (tail)", "", "```text", str(payload["stdout_tail"]), "```", ""])
    if payload["stderr_tail"]:
        md_lines.extend(["## STDERR (tail)", "", "```text", str(payload["stderr_tail"]), "```", ""])
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    re_run_ok = bool(payload["reproduce_ok"])
    restart_reason = ""
    restart_error = ""
    restart_report = ""
    restart_stage = ""
    restart_count = int(state.get("restart_to_plan_count") or 0)
    if not re_run_ok:
        restart_reason = "re_run_failed"
        restart_stage = "re-run"
        restart_error = str(payload.get("error") or payload.get("stderr_tail") or payload.get("stdout_tail") or "")[:4096]
        restart_report = str(report_md)
        restart_count += 1
    restart_limit = _re_restart_limit()
    restart_exceeded = (not re_run_ok) and restart_count > restart_limit

    out = {
        **state,
        "last_step": "re-run",
        "last_error": "" if re_run_ok else restart_error,
        "re_run_done": True,
        "re_run_ok": re_run_ok,
        "re_run_rc": int(payload["reproduce_rc"]),
        "re_run_report_path": str(report_md),
        "re_run_json_path": str(report_json),
        "crash_repro_done": True,
        "crash_repro_ok": re_run_ok,
        "crash_repro_rc": int(payload["reproduce_rc"]),
        "crash_repro_report_path": str(report_md),
        "crash_repro_json_path": str(report_json),
        "restart_to_plan": not re_run_ok,
        "restart_to_plan_reason": restart_reason,
        "restart_to_plan_stage": restart_stage,
        "restart_to_plan_error_text": restart_error,
        "restart_to_plan_report_path": restart_report,
        "restart_to_plan_count": restart_count,
        "failed": bool(state.get("failed")) or restart_exceeded,
        "run_terminal_reason": "re_restart_limit_exceeded" if restart_exceeded else str(state.get("run_terminal_reason") or ""),
        "message": "re-run validated" if re_run_ok else "re-run failed",
        "repair_mode": (not re_run_ok),
        "repair_origin_stage": "crash" if not re_run_ok else "",
        "repair_error_kind": "re_run_failed" if not re_run_ok else "",
        "repair_error_code": restart_reason if not re_run_ok else "",
        "repair_signature": str(state.get("crash_signature") or "")[:12] if not re_run_ok else "",
        "repair_stdout_tail": str(payload.get("stdout_tail") or "") if not re_run_ok else "",
        "repair_stderr_tail": str(payload.get("stderr_tail") or "") if not re_run_ok else "",
        "repair_attempt_index": (int(state.get("repair_attempt_index") or 0) + 1) if not re_run_ok else 0,
        "repair_strategy_force_change": False,
        "repair_error_digest": (
            {
                "error_code": restart_reason,
                "error_kind": "re_run_failed",
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
            if not re_run_ok
            else {}
        ),
        "repair_recent_attempts": (
            (list(state.get("repair_recent_attempts") or []) + [{
                "step": "re-run",
                "origin": "crash",
                "error_kind": "re_run_failed",
                "error_code": restart_reason,
                "signature": str(state.get("crash_signature") or "")[:12],
                "attempt_index": int(state.get("repair_attempt_index") or 0) + 1,
                "message": restart_error[:512],
            }])[-5:]
            if not re_run_ok
            else []
        ),
    }
    if restart_exceeded:
        out["last_error"] = f"re failed and restart-to-plan limit exceeded ({restart_limit})"
    _wf_log(
        cast(dict[str, Any], out),
        (
            "<- re-run "
            f"ok={re_run_ok} rc={payload['reproduce_rc']} "
            f"dt={_fmt_dt(time.perf_counter()-t0)}"
        ),
    )
    return out
