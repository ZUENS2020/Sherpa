"""Carved from workflow_graph.py - '_node_build' LangGraph node."""

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
    _sha256_text,
    _wf_log,
)
from workflow_helpers import (
    _build_failure_recovery_advice,
    _build_repair_error_digest,
    _cache_successful_build_template,
    _classify_build_failure,
    _clear_error_markers_on_success,
    _effective_same_error_retry_limit,
    _execution_targets_min_required,
    _harness_index_path,
    _inject_coverage_instrumentation,
    _install_coverage_cc_wrapper,
    _load_execution_plan_doc,
    _load_harness_index_doc,
    _load_selected_targets_doc,
    _materialize_replay_binaries,
    _normalize_exec_target_token,
    _remaining_time_budget_sec,
    _repair_strategy_repeat_threshold,
    _replay_out_dir,
    _time_budget_exceeded_state,
    _try_hotfix_missing_decl,
    _workflow_target_state_from_execution_plan,
    _write_execution_plan_doc,
    _write_harness_index_doc,
)


def _node_build(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "build")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), f"-> build attempt={(int(state.get('build_attempts') or 0)+1)}")
    try:
        fuzz_dir = gen.repo_root / "fuzz"
        build_py = fuzz_dir / "build.py"
        build_sh = fuzz_dir / "build.sh"
        build_full_log_path = fuzz_dir / "build_full.log"

        def _tail(s: str, n: int = 120) -> str:
            lines = (s or "").replace("\r", "\n").splitlines()
            return "\n".join(lines[-n:]).strip()

        def _init_build_full_log() -> None:
            try:
                build_full_log_path.parent.mkdir(parents=True, exist_ok=True)
                header = (
                    "TianHeng build full log\n"
                    f"repo_root={gen.repo_root}\n"
                    f"generated_at={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
                    + "=" * 88
                    + "\n"
                )
                build_full_log_path.write_text(header, encoding="utf-8", errors="replace")
            except Exception:
                pass

        def _append_build_full_log(*, stage: str, cmd: list[str], cwd: Path, rc: int, out: str, err: str) -> None:
            try:
                lines = [
                    "",
                    "=" * 88,
                    f"stage={stage}",
                    f"cmd={' '.join(cmd)}",
                    f"cwd={cwd}",
                    f"rc={rc}",
                    "-" * 88,
                    "[stdout]",
                    out or "",
                    "-" * 88,
                    "[stderr]",
                    err or "",
                    "=" * 88,
                    "",
                ]
                with build_full_log_path.open("a", encoding="utf-8", errors="replace") as f:
                    f.write("\n".join(lines))
            except Exception:
                pass

        _init_build_full_log()

        def _build_py_supports_clean_flag(path: Path) -> bool:
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return False
            return "--clean" in txt

        def _env_bool(name: str, default: bool) -> bool:
            raw = (os.environ.get(name) or "").strip().lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "on"}

        def _read_declared_system_packages(dep_file: Path) -> set[str]:
            alias_map = {
                "z": "zlib",
                "bz2": "bzip2",
                "lzma": "liblzma",
                "xz": "liblzma",
                "ssl": "openssl",
                "crypto": "openssl",
                "libssl": "openssl",
                "libcrypto": "openssl",
                "xml2": "libxml2",
                "libxml": "libxml2",
            }
            if not dep_file.is_file():
                return set()
            declared: set[str] = set()
            try:
                for raw_line in dep_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = raw_line.split("#", 1)[0].strip().lower()
                    if not line:
                        continue
                    if re.fullmatch(r"[a-z0-9][a-z0-9+._-]*", line):
                        declared.add(alias_map.get(line, line))
            except Exception:
                return set()
            return declared

        def _detect_missing_optional_ports(stdout_text: str, stderr_text: str) -> list[str]:
            combined = ((stdout_text or "") + "\n" + (stderr_text or "")).lower()
            signal_to_port: list[tuple[list[str], str]] = [
                (["could not find zlib", "zlib_library", "zlib_include_dir"], "zlib"),
                (["could not find bzip2", "bzip2_libraries", "bzip2_include_dir"], "bzip2"),
                (["could not find liblzma", "liblzma_library", "liblzma_include_dir"], "liblzma"),
                (["could not find lz4", "lz4_library", "lz4_include_dir"], "lz4"),
                (["could not find zstd", "zstd_library", "zstd_include_dir"], "zstd"),
                (["could not find openssl", "openssl_crypto_library", "openssl_include_dir"], "openssl"),
                (["could not find libxml2", "libxml2_library", "libxml2_include_dir"], "libxml2"),
                (["could not find expat", "expat_library", "expat_include_dir"], "expat"),
            ]
            missing: list[str] = []
            for needles, port in signal_to_port:
                if any(n in combined for n in needles) and port not in missing:
                    missing.append(port)
            return missing

        def _list_static_libs_for_diagnostics() -> str:
            build_dir = gen.repo_root / "build"
            if not build_dir.exists():
                return f"(no build dir at {build_dir})"
            libs: list[str] = []
            try:
                for p in build_dir.rglob("*"):
                    if not p.is_file():
                        continue
                    if p.suffix.lower() in {".a", ".lib", ".so", ".dylib"}:
                        try:
                            libs.append(f"{p.relative_to(gen.repo_root)} ({p.stat().st_size} bytes)")
                        except Exception:
                            libs.append(str(p.relative_to(gen.repo_root)))
                    if len(libs) >= 80:
                        break
            except Exception as e:
                return f"(failed to list libs under build/: {e})"
            return "\n".join(libs) if libs else "(no static libs found under build/)"

        build_cmd_clean: list[str] | None = None
        build_cwd = fuzz_dir
        fallback_cmd: list[str] | None = None
        fallback_cwd: Path | None = None
        if build_py.is_file():
            build_cmd = [gen._python_runner(), "build.py"]
            fallback_cmd = [gen._python_runner(), "fuzz/build.py"]
            fallback_cwd = gen.repo_root
            # Deterministically instrument the library + harness build (does not
            # rely on the PATH cc-wrapper reaching `make` subprocesses).
            _inject_coverage_instrumentation(str(build_py), cast(dict[str, Any], state))
            if _build_py_supports_clean_flag(build_py):
                build_cmd_clean = list(build_cmd) + ["--clean"]
        elif build_sh.is_file():
            shell = "bash"
            if not getattr(gen, "docker_image", None):
                if shutil.which("bash") is None:
                    if shutil.which("sh") is not None:
                        shell = "sh"
                    else:
                        raise HarnessGeneratorError("build.sh exists but neither bash nor sh is available in PATH")
            try:
                mode = build_sh.stat().st_mode
                build_sh.chmod(mode | 0o111)
            except Exception:
                pass
            build_cmd = [shell, "build.sh"]
            fallback_cmd = [shell, "fuzz/build.sh"]
            fallback_cwd = gen.repo_root
        else:
            raise HarnessGeneratorError("Missing fuzz/build.py (agent must create fuzz/build.py)")

        build_env = os.environ.copy()
        # Install a compiler wrapper that automatically appends coverage
        # instrumentation flags to every clang/clang++ invocation.  This
        # approach is format-agnostic — it works regardless of how build.py
        # structures its compile/link commands — and does not conflict with
        # replay builds that already carry -fprofile-instr-generate.
        _cc_wrapper_dir = _install_coverage_cc_wrapper(gen.repo_root)
        build_env["PATH"] = f"{_cc_wrapper_dir}:{build_env.get('PATH', '')}"
        build_env["CC"] = "clang"
        build_env["CXX"] = "clang++"
        build_env.setdefault("CFLAGS", "-D_GNU_SOURCE")
        build_env.setdefault("CXXFLAGS", "-D_GNU_SOURCE")
        if getattr(gen, "docker_image", None):
            include_root = "/work"
            for stale_dir in (gen.repo_root / "fuzz" / "build", gen.repo_root / "build"):
                if stale_dir.exists():
                    try:
                        shutil.rmtree(stale_dir)
                    except Exception:
                        pass
        else:
            include_root = str(gen.repo_root)
        for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            prev = build_env.get(key, "").strip()
            build_env[key] = f"{include_root}:{prev}" if prev else include_root

        retries_raw = os.environ.get("SHERPA_WORKFLOW_BUILD_LOCAL_RETRIES", "2")
        try:
            max_local_attempts = int(retries_raw)
        except Exception:
            max_local_attempts = 2
        max_local_attempts = max(1, min(max_local_attempts, 5))
        retry_with_clean = _env_bool("SHERPA_WORKFLOW_BUILD_RETRY_WITH_CLEAN", True)
        retry_delay_s = 1.0

        attempts_used = 0
        final_rc = 1
        final_out = ""
        final_err = ""
        final_bins: list[Path] = []
        out_dir_mismatch_count = int(state.get("build_output_path_mismatch_count") or 0)
        root_level_bins: list[Path] = []
        soft_gate_threshold_raw = (os.environ.get("SHERPA_BUILD_OUT_PATH_MISMATCH_SOFT_RETRY_LIMIT") or "2").strip()
        try:
            out_path_soft_retry_limit = max(0, min(int(soft_gate_threshold_raw), 10))
        except Exception:
            out_path_soft_retry_limit = 2

        def _discover_root_level_fuzzer_bins() -> list[Path]:
            fuzz_dir = gen.repo_root / "fuzz"
            out_dir = fuzz_dir / "out"
            if not fuzz_dir.is_dir():
                return []
            out: list[Path] = []
            name_re = re.compile(r".*(?:_fuzz(?:er)?|fuzz(?:er)?|Fuzzer)$")
            for p in fuzz_dir.iterdir():
                if p == out_dir:
                    continue
                if not p.is_file():
                    continue
                is_exe = os.access(p, os.X_OK) or p.suffix.lower() == ".exe"
                if not is_exe:
                    continue
                stem = p.stem
                if name_re.match(p.name) or name_re.match(stem) or "fuzz" in p.name.lower():
                    out.append(p)
            return sorted(out)

        def _is_repo_root_cwd_issue(out: str, err: str) -> bool:
            combined = ((out or "") + "\n" + (err or "")).lower()
            return (
                ("no such file or directory" in combined and "fuzz/" in combined)
                or "can't open file '/work/fuzz/fuzz/" in combined
                or "can't open file 'fuzz/" in combined
            )

        for attempt in range(1, max_local_attempts + 1):
            build_cmd_timeout = _remaining_time_budget_sec(state, min_timeout=0)
            if build_cmd_timeout <= 0:
                return _time_budget_exceeded_state(state, step_name="build")
            _wf_log(cast(dict[str, Any], state), f"build cmd attempt {attempt}/{max_local_attempts} -> {' '.join(build_cmd)}")
            rc, out, err = gen._run_cmd(list(build_cmd), cwd=build_cwd, env=build_env, timeout=build_cmd_timeout)
            _append_build_full_log(stage=f"attempt-{attempt}/primary", cmd=list(build_cmd), cwd=build_cwd, rc=rc, out=out, err=err)
            attempts_used += 1

            # Fast-path hotfix for implicit-decl / undeclared-identifier errors.
            # Header-only libraries often need extern declarations that the
            # synthesize agent omits.  Patch harness sources in-place so the
            # next retry (or the fallback retries below) can succeed without
            # going through a full plan→synthesize cycle.
            if rc != 0:
                _hotfix_state: dict[str, Any] = dict(cast(dict[str, Any], state))
                _hotfix_state["last_error"] = (out or "") + "\n" + (err or "")
                _hotfix_state["build_stdout_tail"] = out or ""
                _hotfix_state["build_stderr_tail"] = err or ""
                if _try_hotfix_missing_decl(_hotfix_state, ""):
                    # Hotfix applied — retry the build immediately
                    _wf_log(cast(dict[str, Any], state), "build: harness hotfix applied; retrying")

            # Backward-compatibility shim: older generated scripts may hardcode "fuzz/..."
            # and therefore need repo-root cwd.
            if rc != 0 and fallback_cmd is not None and fallback_cwd is not None and _is_repo_root_cwd_issue(out, err):
                fallback_timeout = _remaining_time_budget_sec(state, min_timeout=0)
                if fallback_timeout <= 0:
                    return _time_budget_exceeded_state(state, step_name="build")
                _wf_log(
                    cast(dict[str, Any], state),
                    f"build retry from repo-root cwd -> {' '.join(fallback_cmd)}",
                )
                rc, out, err = gen._run_cmd(list(fallback_cmd), cwd=fallback_cwd, env=build_env, timeout=fallback_timeout)
                _append_build_full_log(stage=f"attempt-{attempt}/repo-root-fallback", cmd=list(fallback_cmd), cwd=fallback_cwd, rc=rc, out=out, err=err)
                attempts_used += 1

            if rc != 0 and retry_with_clean and build_cmd_clean is not None:
                combined = (out or "") + "\n" + (err or "")
                if not re.search(r"unrecognized arguments: --clean", combined, re.IGNORECASE):
                    clean_timeout = _remaining_time_budget_sec(state, min_timeout=0)
                    if clean_timeout <= 0:
                        return _time_budget_exceeded_state(state, step_name="build")
                    _wf_log(cast(dict[str, Any], state), "build failed; retrying once with --clean")
                    rc2, out2, err2 = gen._run_cmd(list(build_cmd_clean), cwd=build_cwd, env=build_env, timeout=clean_timeout)
                    _append_build_full_log(stage=f"attempt-{attempt}/clean-retry", cmd=list(build_cmd_clean), cwd=build_cwd, rc=rc2, out=out2, err=err2)
                    attempts_used += 1
                    combined2 = (out2 or "") + "\n" + (err2 or "")
                    if re.search(r"unrecognized arguments: --clean", combined2, re.IGNORECASE):
                        _wf_log(cast(dict[str, Any], state), "build.py rejected --clean; keeping original diagnostics")
                    else:
                        rc, out, err = rc2, out2, err2

            bins = gen._discover_fuzz_binaries() if rc == 0 else []
            final_rc, final_out, final_err, final_bins = rc, out, err, bins
            if rc == 0 and bins:
                break

            if attempt < max_local_attempts:
                reason = f"rc={rc}" if rc != 0 else "no fuzzer binaries generated"
                _wf_log(cast(dict[str, Any], state), f"build attempt {attempt} not ready ({reason}); retrying")
                time.sleep(retry_delay_s)

        if final_rc == 0 and not final_bins:
            root_level_bins = _discover_root_level_fuzzer_bins()
            if root_level_bins:
                out_dir_mismatch_count += 1
                mismatch_lines = "\n".join(
                    f"- {p.relative_to(gen.repo_root).as_posix()}" for p in root_level_bins[:20]
                )
                final_out = (
                    (final_out or "")
                    + "\n\n=== build output path mismatch detected ===\n"
                    + "build produced executable fuzzers outside fuzz/out:\n"
                    + mismatch_lines
                    + "\nExpected output directory: fuzz/out/\n"
                )
            libs_diag = _list_static_libs_for_diagnostics()
            if libs_diag:
                final_out = (final_out or "") + "\n\n=== build dir artifacts (static libs) ===\n" + libs_diag + "\n"

        attempts_total = int(state.get("build_attempts") or 0) + attempts_used
        next_state: FuzzWorkflowRuntimeState = {
            **state,
            "build_attempts": attempts_total,
            "build_rc": int(final_rc),
            "build_stdout_tail": _tail(final_out),
            "build_stderr_tail": _tail(final_err),
            "build_full_log_path": str(build_full_log_path),
            "harness_index_path": str(_harness_index_path(gen.repo_root)),
            "last_step": "build",
            "build_mode": str(state.get("build_mode") or ""),
            "build_target_source": str(state.get("build_target_source") or "external_scaffold"),
            "build_output_path_mismatch_count": out_dir_mismatch_count,
        }
        def _mark_build_repair_state(*, kind: str, code: str, sig: str = "") -> None:
            signature_short = sig[:12] if sig else str(next_state.get("build_error_signature_short") or "")
            attempt_index = int(state.get("repair_attempt_index") or 0) + 1
            same_signature_streak = int(next_state.get("same_build_error_repeats") or 0) + 1
            force_change = same_signature_streak >= _repair_strategy_repeat_threshold()
            next_state["repair_mode"] = True
            next_state["repair_origin_stage"] = "build"
            next_state["repair_error_kind"] = kind or "build_failure_generic"
            next_state["repair_error_code"] = code or ""
            next_state["repair_signature"] = signature_short
            next_state["repair_stdout_tail"] = str(next_state.get("build_stdout_tail") or "")
            next_state["repair_stderr_tail"] = str(next_state.get("build_stderr_tail") or "")
            next_state["repair_attempt_index"] = attempt_index
            next_state["repair_strategy_force_change"] = force_change
            if force_change:
                force_msg = (
                    " strategy_change_required: same build signature repeated; "
                    "next repair round must materially change target selection or harness/build strategy."
                )
                current_error = str(next_state.get("last_error") or "")
                if force_msg.strip() not in current_error:
                    next_state["last_error"] = (current_error + force_msg).strip()
            next_state["repair_error_digest"] = _build_repair_error_digest(
                repo_root=gen.repo_root,
                error_kind=kind or "build_failure_generic",
                error_code=code or "",
                signature=signature_short,
                error_text=str(next_state.get("last_error") or ""),
                stdout_tail=str(next_state.get("build_stdout_tail") or ""),
                stderr_tail=str(next_state.get("build_stderr_tail") or ""),
                prev_digest=dict(state.get("repair_error_digest") or {}),
            )
            recent = list(state.get("repair_recent_attempts") or [])
            recent.append(
                {
                    "step": "build",
                    "origin": "build",
                    "error_kind": kind or "build_failure_generic",
                    "error_code": code or "",
                    "signature": signature_short,
                    "attempt_index": attempt_index,
                    "force_strategy_change": force_change,
                    "message": str(next_state.get("last_error") or "")[:512],
                }
            )
            next_state["repair_recent_attempts"] = recent[-5:]
        build_error_kind, build_error_code = _classify_build_failure(
            str(next_state.get("last_error") or ""),
            str(next_state.get("build_stdout_tail") or ""),
            str(next_state.get("build_stderr_tail") or ""),
            build_rc=int(final_rc),
            has_fuzzer_binaries=bool(final_bins),
        )

        def _calc_build_error_signature() -> str:
            marker = "rc-fail" if final_rc != 0 else "no-fuzzers"
            blob = (
                marker
                + "\n"
                + _tail(final_out, n=220)
                + "\n"
                + _tail(final_err, n=220)
            )
            return _sha256_text(blob)

        prev_sig = str(state.get("build_error_signature") or "").strip()
        prev_repeats = int(state.get("same_build_error_repeats") or 0)
        max_same_repeats = _effective_same_error_retry_limit(state)

        if final_rc != 0:
            sig = _calc_build_error_signature()
            next_state["build_error_signature_short"] = sig[:12]
            repeats = (prev_repeats + 1) if (prev_sig and prev_sig == sig) else 0
            next_state["build_error_signature"] = sig
            next_state["build_error_signature_before"] = prev_sig
            next_state["build_error_signature_after"] = sig
            next_state["same_build_error_repeats"] = repeats
            next_state["build_error_kind"] = build_error_kind
            next_state["build_error_code"] = build_error_code
            advice = _build_failure_recovery_advice(build_error_kind, build_error_code)
            if max_same_repeats > 0 and repeats >= max_same_repeats:
                repeated_err = (
                    "build failed with the same error signature repeatedly "
                    f"(repeats={repeats + 1}, threshold={max_same_repeats + 1})"
                )
                next_state["failed"] = False
                next_state["last_error"] = repeated_err
                next_state["message"] = "build failed repeatedly (same error)"
                next_state["restart_to_plan"] = build_error_kind == "infra"
                next_state["restart_to_plan_reason"] = "build_same_error_repeated" if build_error_kind == "infra" else ""
                next_state["restart_to_plan_stage"] = "build" if build_error_kind == "infra" else ""
                next_state["restart_to_plan_error_text"] = repeated_err if build_error_kind == "infra" else ""
                _mark_build_repair_state(
                    kind=str(next_state.get("build_error_kind") or build_error_kind or "build_failure_generic"),
                    code=str(next_state.get("build_error_code") or build_error_code or ""),
                    sig=sig,
                )
                _wf_log(
                    cast(dict[str, Any], next_state),
                    "<- build stop same-error "
                    f"repeats={repeats+1} "
                    f"signature_before={prev_sig[:12] if prev_sig else '-'} "
                    f"signature_after={sig[:12]} "
                    f"same_error_max_retries={max_same_repeats}",
                )
                return next_state
            next_state["last_error"] = f"build failed rc={final_rc} after {attempts_used} command run(s)"
            if advice:
                next_state["last_error"] += f"\nrecovery: {advice}"
            next_state["message"] = "build failed"
            next_state["restart_to_plan"] = build_error_kind == "infra"
            next_state["restart_to_plan_reason"] = "build_failed" if build_error_kind == "infra" else ""
            next_state["restart_to_plan_stage"] = "build" if build_error_kind == "infra" else ""
            next_state["restart_to_plan_error_text"] = str(next_state["last_error"]) if build_error_kind == "infra" else ""
            _mark_build_repair_state(
                kind=str(next_state.get("build_error_kind") or build_error_kind or "build_failure_generic"),
                code=str(next_state.get("build_error_code") or build_error_code or ""),
                sig=sig,
            )
            _wf_log(
                cast(dict[str, Any], next_state),
                "<- build fail "
                f"rc={final_rc} "
                f"signature_before={prev_sig[:12] if prev_sig else '-'} "
                f"signature_after={sig[:12]} "
                f"same_error_count={repeats} "
                f"same_error_max_retries={max_same_repeats} "
                f"dt={_fmt_dt(time.perf_counter()-t0)}",
            )
            return next_state

        if not final_bins:
            sig = _calc_build_error_signature()
            next_state["build_error_signature_short"] = sig[:12]
            repeats = (prev_repeats + 1) if (prev_sig and prev_sig == sig) else 0
            next_state["build_error_signature"] = sig
            next_state["build_error_signature_before"] = prev_sig
            next_state["build_error_signature_after"] = sig
            next_state["same_build_error_repeats"] = repeats
            next_state["build_error_kind"] = build_error_kind
            next_state["build_error_code"] = build_error_code
            if max_same_repeats > 0 and repeats >= max_same_repeats:
                repeated_err = (
                    "build produced no fuzzers with the same diagnostics repeatedly "
                    f"(repeats={repeats + 1}, threshold={max_same_repeats + 1})"
                )
                next_state["failed"] = False
                next_state["last_error"] = repeated_err
                next_state["message"] = "build failed repeatedly (no fuzzers)"
                next_state["restart_to_plan"] = build_error_kind == "infra"
                next_state["restart_to_plan_reason"] = "build_no_fuzzer_repeated" if build_error_kind == "infra" else ""
                next_state["restart_to_plan_stage"] = "build" if build_error_kind == "infra" else ""
                next_state["restart_to_plan_error_text"] = repeated_err if build_error_kind == "infra" else ""
                _mark_build_repair_state(
                    kind=build_error_kind or "build_failure_generic",
                    code=build_error_code,
                    sig=sig,
                )
                _wf_log(
                    cast(dict[str, Any], next_state),
                    "<- build stop same-no-fuzzer "
                    f"repeats={repeats+1} "
                    f"signature_before={prev_sig[:12] if prev_sig else '-'} "
                    f"signature_after={sig[:12]} "
                    f"same_error_max_retries={max_same_repeats}",
                )
                return next_state
            if root_level_bins and out_dir_mismatch_count <= out_path_soft_retry_limit:
                root_listing = ", ".join(p.name for p in root_level_bins[:8])
                next_state["last_error"] = (
                    "Build output path mismatch: executable fuzzers exist under fuzz/ root "
                    f"({root_listing}) but none under fuzz/out/ after {attempts_used} command run(s)."
                )
                next_state["build_error_kind"] = "source"
                next_state["build_error_code"] = "build_output_path_mismatch"
            else:
                next_state["last_error"] = f"No fuzzer binaries found under fuzz/out/ after {attempts_used} command run(s)"
            next_state["message"] = "build produced no fuzzers"
            next_state["restart_to_plan"] = build_error_kind == "infra"
            next_state["restart_to_plan_reason"] = "build_no_fuzzers" if build_error_kind == "infra" else ""
            next_state["restart_to_plan_stage"] = "build" if build_error_kind == "infra" else ""
            next_state["restart_to_plan_error_text"] = str(next_state["last_error"]) if build_error_kind == "infra" else ""
            _mark_build_repair_state(kind=build_error_kind or "build_failure_generic", code=build_error_code, sig=sig)
            _wf_log(
                cast(dict[str, Any], next_state),
                "<- build fail no-fuzzers "
                f"signature_before={prev_sig[:12] if prev_sig else '-'} "
                f"signature_after={sig[:12]} "
                f"same_error_count={repeats} "
                f"same_error_max_retries={max_same_repeats} "
                f"dt={_fmt_dt(time.perf_counter()-t0)}",
            )
            return next_state

        selected_targets_doc = _load_selected_targets_doc(gen.repo_root)
        if selected_targets_doc:
            _, execution_plan_doc = _write_execution_plan_doc(gen.repo_root, selected_targets_doc)
        else:
            execution_plan_doc = _load_execution_plan_doc(gen.repo_root)
        execution_targets = [
            item for item in list(execution_plan_doc.get("execution_targets") or [])
            if isinstance(item, dict)
        ]
        boundary_target_state = _workflow_target_state_from_execution_plan(
            gen.repo_root,
            execution_plan_doc,
        )
        next_state.update(boundary_target_state)
        if selected_targets_doc:
            try:
                _, harness_index_doc = _write_harness_index_doc(
                    gen.repo_root,
                    execution_plan_doc=execution_plan_doc,
                )
            except Exception:
                harness_index_doc = _load_harness_index_doc(gen.repo_root)
        else:
            harness_index_doc = _load_harness_index_doc(gen.repo_root)
            if not harness_index_doc:
                try:
                    _, harness_index_doc = _write_harness_index_doc(
                        gen.repo_root,
                        execution_plan_doc=execution_plan_doc,
                    )
                except Exception:
                    harness_index_doc = {}
        mapping_by_target: dict[str, dict[str, Any]] = {}
        for row in list(harness_index_doc.get("mappings") or []):
            if not isinstance(row, dict):
                continue
            key = str(row.get("target_name") or "").strip()
            if key and key not in mapping_by_target:
                mapping_by_target[key] = row
        built_names = {p.name for p in final_bins}
        built_stems = {Path(name).stem for name in built_names}
        built_norm_tokens = {
            token
            for token in (
                _normalize_exec_target_token(name)
                for name in (list(built_names) + list(built_stems))
            )
            if token
        }
        target_build_matrix: list[dict[str, Any]] = []
        built_execution_targets = 0
        for item in execution_targets:
            target_name = str(item.get("target_name") or "").strip()
            expected = str(item.get("expected_fuzzer_name") or item.get("target_name") or "").strip()
            mapping = mapping_by_target.get(target_name)
            source_path = str(mapping.get("source_path") or "").strip() if isinstance(mapping, dict) else ""
            source_stem = Path(source_path).stem if source_path else ""
            raw_candidates = {
                expected,
                f"{expected}_fuzz" if expected else "",
                f"{expected}_fuzzer" if expected else "",
                target_name,
                f"{target_name}_fuzz" if target_name else "",
                f"{target_name}_fuzzer" if target_name else "",
                source_stem,
            }
            raw_candidates = {c for c in raw_candidates if c}
            norm_candidates = {
                _normalize_exec_target_token(c)
                for c in (list(raw_candidates) + [Path(c).stem for c in raw_candidates])
            }
            norm_candidates = {c for c in norm_candidates if c}
            matched = bool(
                (raw_candidates & built_names)
                or ({Path(c).stem for c in raw_candidates} & built_stems)
                or (norm_candidates & built_norm_tokens)
            )
            has_source = bool(source_path)
            if matched and has_source:
                built_execution_targets += 1
            target_build_matrix.append(
                {
                    "target_name": target_name,
                    "expected_fuzzer_name": expected,
                    "must_run": bool(item.get("must_run") or False),
                    "source_path": source_path,
                    "has_source": has_source,
                    "built": bool(matched),
                }
            )
        min_required_built = int(execution_plan_doc.get("min_required_built_targets") or _execution_targets_min_required())
        if execution_targets and len(execution_targets) > 1 and built_execution_targets < min_required_built:
            missing_targets = [
                str(item.get("target_name") or item.get("expected_fuzzer_name") or "")
                for item in target_build_matrix
                if not (bool(item.get("built")) and bool(item.get("has_source")))
            ]
            next_state["last_error"] = (
                "partial_build_undercoverage: built "
                f"{built_execution_targets}/{len(execution_targets)} execution targets "
                f"(required>={min_required_built}); missing={','.join([x for x in missing_targets if x]) or 'unknown'}"
            )
            next_state["message"] = "build undercoverage: execution target gate not met"
            next_state["build_error_kind"] = "source"
            next_state["build_error_code"] = "partial_build_undercoverage"
            next_state["restart_to_plan"] = False
            next_state["restart_to_plan_reason"] = ""
            next_state["restart_to_plan_stage"] = ""
            next_state["restart_to_plan_error_text"] = ""
            next_state["restart_to_plan_report_path"] = ""
            _mark_build_repair_state(kind="source", code="partial_build_undercoverage", sig=str(next_state.get("build_error_signature_short") or ""))
            if isinstance(next_state.get("repair_error_digest"), dict):
                next_state["repair_error_digest"]["oracle"] = "undercoverage_gate"
            next_state["build_gate_reason"] = "partial_build_undercoverage"
            next_state["built_targets"] = sorted(built_names)
            next_state["missing_targets"] = missing_targets
            next_state["target_build_matrix"] = target_build_matrix
            _wf_log(
                cast(dict[str, Any], next_state),
                "<- build gate partial_build_undercoverage "
                f"built={built_execution_targets}/{len(execution_targets)} required={min_required_built} "
                f"dt={_fmt_dt(time.perf_counter()-t0)}",
            )
            return next_state

        next_state["build_error_signature"] = ""
        next_state["build_error_signature_before"] = prev_sig
        next_state["build_error_signature_after"] = ""
        next_state["build_error_signature_short"] = ""
        next_state["same_build_error_repeats"] = 0
        next_state["build_error_kind"] = ""
        next_state["build_error_code"] = ""
        next_state["fix_build_attempts"] = 0
        next_state["fix_build_noop_streak"] = 0
        next_state["fix_build_terminal_reason"] = ""
        next_state["fix_build_last_diff_paths"] = []
        next_state["fix_action_type"] = ""
        next_state["fix_effect"] = ""
        next_state["last_error"] = ""
        next_state["repair_mode"] = False
        next_state["repair_origin_stage"] = ""
        next_state["repair_error_kind"] = ""
        next_state["repair_error_code"] = ""
        next_state["repair_signature"] = ""
        next_state["repair_stdout_tail"] = ""
        next_state["repair_stderr_tail"] = ""
        next_state["repair_recent_attempts"] = []
        next_state["repair_error_digest"] = {}
        next_state["repair_attempt_index"] = 0
        next_state["repair_strategy_force_change"] = False

        enforce_declared_optional_deps = _env_bool("SHERPA_BUILD_ENFORCE_DECLARED_OPTIONAL_DEPS", True)
        if enforce_declared_optional_deps:
            dep_file = fuzz_dir / "system_packages.txt"
            declared_ports = _read_declared_system_packages(dep_file)
            missing_ports = [
                p for p in _detect_missing_optional_ports(final_out, final_err)
                if p not in declared_ports
            ]
            if missing_ports:
                next_state["last_error"] = (
                    "build succeeded with missing optional libraries but "
                    "fuzz/system_packages.txt does not declare required vcpkg ports: "
                    + ", ".join(missing_ports)
                )
                next_state["message"] = "build missing declared optional deps"
                next_state["build_error_kind"] = "source"
                next_state["build_error_code"] = "missing_system_packages_declared"
                next_state["restart_to_plan"] = False
                next_state["restart_to_plan_reason"] = ""
                next_state["restart_to_plan_stage"] = ""
                next_state["restart_to_plan_error_text"] = ""
                next_state["restart_to_plan_report_path"] = ""
                _mark_build_repair_state(kind="source", code="missing_system_packages_declared", sig=str(next_state.get("build_error_signature_short") or ""))
                if isinstance(next_state.get("repair_error_digest"), dict):
                    next_state["repair_error_digest"]["oracle"] = "undercoverage_gate"
                _wf_log(
                    cast(dict[str, Any], next_state),
                    "<- build gate missing-optional-deps "
                    f"ports={','.join(missing_ports)} dt={_fmt_dt(time.perf_counter()-t0)}",
                )
                return next_state

        cache_path = _cache_successful_build_template(
            gen.repo_root,
            binaries=final_bins,
            target_build_matrix=target_build_matrix,
        )
        replay_bins = _materialize_replay_binaries(gen.repo_root, final_bins)
        if cache_path:
            next_state["build_template_cache_path"] = cache_path
        next_state["built_targets"] = sorted(built_names)
        next_state["missing_targets"] = [
            str(item.get("target_name") or item.get("expected_fuzzer_name") or "")
            for item in target_build_matrix
            if not bool(item.get("built"))
        ]
        next_state["target_build_matrix"] = target_build_matrix
        next_state["build_gate_reason"] = "ok"
        next_state["coverage_replay_binary_dir"] = str(_replay_out_dir(gen.repo_root))
        next_state["coverage_replay_binary_count"] = int(len(replay_bins))
        next_state["message"] = f"built ({len(final_bins)} fuzzers)"
        next_state = _clear_error_markers_on_success(next_state)
        _wf_log(cast(dict[str, Any], next_state), f"<- build ok fuzzers={len(final_bins)} dt={_fmt_dt(time.perf_counter()-t0)}")
        return next_state
    except Exception as e:
        out = {
            **state,
            "last_step": "build",
            "last_error": str(e),
            "message": "build failed",
            "failed": True,
            "build_error_kind": "unknown",
            "build_error_code": "build_node_exception",
            "restart_to_plan": True,
            "restart_to_plan_reason": "build_node_exception",
            "restart_to_plan_stage": "build",
            "restart_to_plan_error_text": str(e),
            "repair_mode": True,
            "repair_origin_stage": "build",
            "repair_error_kind": "build_failure_generic",
            "repair_error_code": "build_node_exception",
            "repair_signature": "",
            "repair_stdout_tail": str(state.get("build_stdout_tail") or ""),
            "repair_stderr_tail": str(state.get("build_stderr_tail") or ""),
            "repair_recent_attempts": list(state.get("repair_recent_attempts") or []),
        }
        if "build_full_log_path" in locals():
            out["build_full_log_path"] = str(build_full_log_path)
        _wf_log(cast(dict[str, Any], out), f"<- build err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
