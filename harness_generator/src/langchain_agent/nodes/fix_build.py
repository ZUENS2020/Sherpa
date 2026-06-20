"""Carved from workflow_graph.py - '_node_fix_build' LangGraph node."""

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
    _attach_prompt_render_status,
    _build_file_targeted_fix_lines,
    _classify_build_failure,
    _contains_cjk_text,
    _effective_max_fix_rounds,
    _extract_json_object,
    _fix_build_context_history_limit,
    _fix_build_context_max_chars,
    _fix_build_feedback_history_limit,
    _fix_build_keep_recent_errors,
    _fix_build_max_noop_streak,
    _fix_build_ruleset,
    _fix_build_stderr_max_chars,
    _fix_build_stdout_max_chars,
    _llm_or_none,
    _load_build_runtime_facts_doc,
    _load_build_strategy_doc,
    _load_repo_understanding_doc,
    _opencode_cli_retries,
    _remaining_time_budget_sec,
    _render_opencode_prompt_safe,
    _splice_sources_list,
    _summarize_build_error,
    _try_hotfix_missing_decl,
)


def _node_fix_build(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "fix_build")
    if stop_now:
        return state

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> fix_build")
    fix_attempts = int(state.get("fix_build_attempts") or 0) + 1
    state = cast(FuzzWorkflowRuntimeState, {**state, "fix_build_attempts": fix_attempts})

    last_error = (state.get("last_error") or "").strip()
    stdout_tail = (state.get("build_stdout_tail") or "").strip()
    stderr_tail = (state.get("build_stderr_tail") or "").strip()
    build_error_kind = (state.get("build_error_kind") or "").strip().lower()
    build_error_code = (state.get("build_error_code") or "").strip().lower()
    repo_root = str(gen.repo_root)
    diag_text = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
    prev_noop_streak = int(state.get("fix_build_noop_streak") or 0)
    history = list(state.get("fix_build_attempt_history") or [])
    rule_hits = list(state.get("fix_build_rule_hits") or [])
    max_noop_streak = _fix_build_max_noop_streak()
    max_fix_attempts = _effective_max_fix_rounds(state)
    history_limit = _fix_build_feedback_history_limit()
    context_history_limit = _fix_build_context_history_limit()
    context_max_chars = _fix_build_context_max_chars()
    error_sig = (state.get("build_error_signature_short") or "").strip()
    if not error_sig:
        error_sig = _sha256_text("\n".join([last_error, stdout_tail, stderr_tail]))[:12]

    if max_fix_attempts > 0 and fix_attempts > max_fix_attempts:
        out = {
            **state,
            "last_step": "fix_build",
            "failed": False,
            "fix_build_terminal_reason": "fix_build_max_attempts_exceeded",
            "last_error": f"fix_build max attempts exceeded ({max_fix_attempts}); restart from plan",
            "message": "fix_build max attempts exceeded; restarting from plan",
            "restart_to_plan": True,
            "restart_to_plan_reason": "fix_build_max_attempts_exceeded",
            "restart_to_plan_stage": "fix_build",
            "restart_to_plan_error_text": str(last_error or "").strip(),
            "fix_action_type": "none",
            "fix_effect": "stalled",
        }
        _wf_log(cast(dict[str, Any], out), f"<- fix_build stop=max-attempts limit={max_fix_attempts}")
        return out

    def _append_attempt(outcome: str, *, rejection_reason: str = "", rule_hit: str = "", changed_paths_count: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
        updated_rule_hits = list(rule_hits)
        if rule_hit and rule_hit not in updated_rule_hits:
            updated_rule_hits.append(rule_hit)
        row = {
            "attempt_index": fix_attempts,
            "build_error_kind": build_error_kind or "unknown",
            "build_error_code": build_error_code or "unknown",
            "classified_signature": error_sig,
            "changed_paths_count": int(changed_paths_count),
            "outcome": outcome,
            "rejection_reason": rejection_reason,
            "rule_hit": rule_hit,
        }
        updated_history = history + [row]
        if len(updated_history) > history_limit:
            updated_history = updated_history[-history_limit:]
        return updated_history, updated_rule_hits

    def _fix_build_quick_check_timeout_sec() -> int:
        raw = (os.environ.get("SHERPA_FIX_BUILD_QUICK_CHECK_TIMEOUT_SEC") or "0").strip()
        try:
            return max(0, min(int(raw), 300))
        except Exception:
            return 0

    def _run_fix_build_quick_probe() -> tuple[bool, dict[str, Any]]:
        def _tail_local(s: str, n: int = 120) -> str:
            lines = (s or "").replace("\r", "\n").splitlines()
            return "\n".join(lines[-n:]).strip()

        if not hasattr(gen, "_run_cmd"):
            return False, {"reason": "unsupported_generator"}

        fuzz_dir = gen.repo_root / "fuzz"
        build_py = fuzz_dir / "build.py"
        build_sh = fuzz_dir / "build.sh"
        if not build_py.is_file() and not build_sh.is_file():
            return False, {"reason": "missing_build_script"}

        quick_timeout = _fix_build_quick_check_timeout_sec()
        if quick_timeout <= 0:
            return False, {"reason": "disabled"}
        remaining = _remaining_time_budget_sec(state, min_timeout=0)
        if remaining <= 0:
            return False, {"reason": "no_budget"}
        timeout = min(remaining, quick_timeout)

        build_cwd = fuzz_dir
        if build_py.is_file():
            if hasattr(gen, "_python_runner"):
                cmd = [gen._python_runner(), "build.py"]
            else:
                py = shutil.which("python3") or shutil.which("python") or "python"
                cmd = [py, "build.py"]
        else:
            shell = "bash"
            if not getattr(gen, "docker_image", None):
                if shutil.which("bash") is None and shutil.which("sh") is not None:
                    shell = "sh"
            cmd = [shell, "build.sh"]

        build_env = os.environ.copy()
        if getattr(gen, "docker_image", None):
            include_root = "/work"
            build_env.setdefault("CC", "clang")
            build_env.setdefault("CXX", "clang++")
            build_env.setdefault("CFLAGS", "-D_GNU_SOURCE")
            build_env.setdefault("CXXFLAGS", "-D_GNU_SOURCE")
        else:
            include_root = str(gen.repo_root)
        for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            prev = build_env.get(key, "").strip()
            build_env[key] = f"{include_root}:{prev}" if prev else include_root

        rc, out, err = gen._run_cmd(list(cmd), cwd=build_cwd, env=build_env, timeout=timeout)
        bins = gen._discover_fuzz_binaries() if rc == 0 else []
        marker = "rc-fail" if rc != 0 else ("ok" if bins else "no-fuzzers")
        signature = _sha256_text(marker + "\n" + _tail_local(out, n=200) + "\n" + _tail_local(err, n=200))
        kind, code = _classify_build_failure(
            "",
            _tail_local(out, n=200),
            _tail_local(err, n=200),
            build_rc=int(rc),
            has_fuzzer_binaries=bool(bins),
        )
        return True, {
            "rc": int(rc),
            "has_bins": bool(bins),
            "stdout_tail": _tail_local(out, n=200),
            "stderr_tail": _tail_local(err, n=200),
            "signature": signature,
            "kind": kind,
            "code": code,
            "cmd": " ".join(cmd),
            "timeout": timeout,
        }

    def _requires_env_rebuild(changed_paths: list[str] | None = None) -> bool:
        normalized = {
            str(p or "").strip().replace("\\", "/")
            for p in (changed_paths or [])
            if str(p or "").strip()
        }
        return "fuzz/system_packages.txt" in normalized

    def _success_out(message: str, *, outcome: str, rule_hit: str = "", changed_paths_count: int = 1, last_diff_paths: list[str] | None = None) -> FuzzWorkflowRuntimeState:
        updated_history, updated_rule_hits = _append_attempt(
            outcome,
            rule_hit=rule_hit,
            changed_paths_count=changed_paths_count,
        )
        out = cast(
            FuzzWorkflowRuntimeState,
            {
                **state,
                "last_step": "fix_build",
                "last_error": "",
                "codex_hint": "",
                "message": message,
                "fix_build_noop_streak": 0,
                "fix_build_attempt_history": updated_history,
                "fix_build_rule_hits": updated_rule_hits,
                "fix_build_terminal_reason": "",
                "fix_build_last_diff_paths": list(last_diff_paths or []),
                "fix_action_type": "rule" if rule_hit else "opencode",
                "fix_effect": "advanced",
            },
        )
        if _requires_env_rebuild(last_diff_paths):
            out["message"] = f"{message} (requires env rebuild)"
            out["fix_effect"] = "requires_env_rebuild"
            out["fix_build_terminal_reason"] = "requires_env_rebuild"
            return out
        probe_ran, probe = _run_fix_build_quick_probe()
        if probe_ran:
            probe_rc = int(probe.get("rc") or 1)
            probe_has_bins = bool(probe.get("has_bins"))
            probe_sig = str(probe.get("signature") or "")
            prev_sig_full = str(state.get("build_error_signature") or "")
            same_signature = bool(probe_sig and prev_sig_full and probe_sig == prev_sig_full)
            _wf_log(
                cast(dict[str, Any], state),
                "fix_build: quick-check "
                f"cmd={probe.get('cmd')} timeout={probe.get('timeout')}s rc={probe_rc} has_bins={probe_has_bins}",
            )
            if probe_rc == 0 and probe_has_bins:
                out["message"] = f"{message} (quick-check passed)"
                out["fix_effect"] = "advanced"
                return out

            next_noop_streak = (prev_noop_streak + 1) if same_signature else 0
            out["fix_build_noop_streak"] = next_noop_streak
            out["build_rc"] = probe_rc
            out["build_stdout_tail"] = str(probe.get("stdout_tail") or "")
            out["build_stderr_tail"] = str(probe.get("stderr_tail") or "")
            out["build_error_signature_before"] = prev_sig_full
            out["build_error_signature_after"] = probe_sig
            out["build_error_signature"] = probe_sig
            out["build_error_signature_short"] = probe_sig[:12]
            out["build_error_kind"] = str(probe.get("kind") or "")
            out["build_error_code"] = str(probe.get("code") or "")
            out["same_build_error_repeats"] = (int(state.get("same_build_error_repeats") or 0) + 1) if same_signature else 0
            out["last_error"] = (
                f"fix_build quick-check failed rc={probe_rc} "
                f"(same_signature={'yes' if same_signature else 'no'})"
            )
            out["message"] = "fix_build changed files but quick-check failed"
            out["fix_effect"] = "stalled" if same_signature else "advanced"
            if same_signature and next_noop_streak >= max_noop_streak:
                out["failed"] = False
                out["fix_build_terminal_reason"] = "fix_build_noop_streak_exceeded"
                out["last_error"] = f"fix_build no-op streak exceeded ({max_noop_streak}); restart from plan"
                out["message"] = "fix_build no-op streak exceeded; restarting from plan"
                out["restart_to_plan"] = True
                out["restart_to_plan_reason"] = "fix_build_noop_streak_exceeded"
                out["restart_to_plan_stage"] = "fix_build"
                out["restart_to_plan_error_text"] = str(out.get("last_error") or "")
        else:
            _wf_log(cast(dict[str, Any], state), f"fix_build: quick-check skipped ({probe.get('reason')})")
        return out

    def _detect_non_source_build_blocker(diag: str) -> str:
        checks: list[tuple[str, list[str]]] = [
            (
                "docker_daemon_unavailable",
                [
                    "cannot connect to the docker daemon",
                    "is the docker daemon running",
                    "lookup sherpa-docker",
                    "permission denied while trying to connect to the docker daemon",
                ],
            ),
            (
                "registry_or_network_unavailable",
                [
                    "tls handshake timeout",
                    "temporary failure in name resolution",
                    "failed to resolve source metadata",
                    "dial tcp",
                    "no such host",
                ],
            ),
            (
                "resource_exhausted",
                [
                    "no space left on device",
                    "cannot allocate memory",
                    "out of memory",
                    "killed",
                ],
            ),
            (
                "build_command_timeout",
                [
                    "[timeout] process exceeded limit and was killed",
                    "process exceeded limit and was killed",
                ],
            ),
        ]
        for reason, needles in checks:
            if any(n in diag for n in needles):
                return reason
        return ""

    stop_on_infra_raw = (os.environ.get("SHERPA_WORKFLOW_STOP_ON_INFRA_BUILD_ERROR") or "").strip().lower()
    stop_on_infra = stop_on_infra_raw in {"1", "true", "yes", "on"}
    non_source_reason = ""
    if build_error_kind == "infra":
        non_source_reason = build_error_code or _detect_non_source_build_blocker(diag_text) or "infra_build_failure"
    else:
        non_source_reason = _detect_non_source_build_blocker(diag_text)
    if stop_on_infra and non_source_reason:
        updated_history, updated_rule_hits = _append_attempt(
            "infra_blocked",
            rejection_reason=non_source_reason,
            changed_paths_count=0,
        )
        out = {
            **state,
            "last_step": "fix_build",
            "failed": False,
            "build_error_kind": "infra",
            "build_error_code": non_source_reason,
            "fix_build_terminal_reason": "fix_build_infra_blocked",
            "fix_build_attempt_history": updated_history,
            "fix_build_rule_hits": updated_rule_hits,
            "last_error": f"non-source build blocker detected: {non_source_reason}; restart from plan",
            "message": "fix_build skipped (environment/infrastructure issue), restarting from plan",
            "restart_to_plan": True,
            "restart_to_plan_reason": f"infra:{non_source_reason}",
            "restart_to_plan_stage": "fix_build",
            "restart_to_plan_error_text": str(last_error or "").strip(),
            "fix_action_type": "none",
            "fix_effect": "stalled",
        }
        _wf_log(cast(dict[str, Any], out), f"<- fix_build stop=non-source reason={non_source_reason}")
        return out

    build_log_file = ""
    raw_build_log_path = (state.get("build_full_log_path") or "").strip()
    if raw_build_log_path:
        p = Path(raw_build_log_path)
        if p.is_file():
            try:
                build_log_file = str(p.resolve().relative_to(gen.repo_root.resolve())).replace("\\", "/")
            except Exception:
                build_log_file = p.name
    if not build_log_file:
        default_log = gen.repo_root / "fuzz" / "build_full.log"
        if default_log.is_file():
            build_log_file = "fuzz/build_full.log"

    def _is_fix_build_allowed_path(rel_path: str) -> bool:
        rel = rel_path.strip().replace("\\", "/")
        if not rel:
            return False
        if rel == "done":
            return True
        return rel.startswith("fuzz/")

    def _collect_fix_step_hashes() -> dict[str, str]:
        repo_root = gen.repo_root
        out: dict[str, str] = {}
        skip_prefixes = (
            ".git/",
            "fuzz/out/",
            "fuzz/corpus/",
            "fuzz/build/",
        )
        skip_names = {"fuzz/build_full.log", "done"}
        for current_root, dirnames, filenames in os.walk(repo_root, topdown=True):
            try:
                root_rel = str(Path(current_root).relative_to(repo_root)).replace("\\", "/")
            except Exception:
                continue
            if root_rel == ".":
                root_rel = ""

            keep_dirs: list[str] = []
            for d in dirnames:
                rel_dir = f"{root_rel}/{d}" if root_rel else d
                rel_dir = rel_dir.replace("\\", "/")
                rel_prefix = f"{rel_dir}/"
                if rel_dir == ".git" or any(rel_prefix.startswith(pref) for pref in skip_prefixes):
                    continue
                keep_dirs.append(d)
            dirnames[:] = keep_dirs

            for name in filenames:
                rel = f"{root_rel}/{name}" if root_rel else name
                rel = rel.replace("\\", "/")
                if rel in skip_names:
                    continue
                if any(rel.startswith(pref) for pref in skip_prefixes):
                    continue
                path = repo_root / rel
                try:
                    if path.stat().st_size > 5_000_000:
                        continue
                    data = path.read_bytes()
                except Exception:
                    continue
                out[rel] = hashlib.sha256(data).hexdigest()
        return out

    def _collect_fix_relevant_hashes() -> dict[str, str]:
        fuzz_dir = gen.repo_root / "fuzz"
        if not fuzz_dir.is_dir():
            return {}
        out: dict[str, str] = {}
        skip_prefixes = ("fuzz/out/", "fuzz/corpus/", "fuzz/build/")
        skip_names = {"fuzz/build_full.log"}
        for p in fuzz_dir.rglob("*"):
            if not p.is_file():
                continue
            try:
                rel = str(p.relative_to(gen.repo_root)).replace("\\", "/")
            except Exception:
                continue
            if rel in skip_names:
                continue
            if any(rel.startswith(pref) for pref in skip_prefixes):
                continue
            try:
                data = p.read_bytes()
            except Exception:
                continue
            if len(data) > 5_000_000:
                continue
            out[rel] = hashlib.sha256(data).hexdigest()
        return out

    baseline_fix_hashes = _collect_fix_relevant_hashes()
    baseline_step_hashes = _collect_fix_step_hashes()

    # Fast-path hotfixes (minimal, no refactor):
    # 1) libstdc++/libc++ ABI mismatch from injected "-stdlib=libc++"
    # 2) libFuzzer main conflict when target sources define main()
    # 3) linking with `-lz` while the static library is only available by file path.
    def _repo_has_c_cpp_main() -> bool:
        exts = {".c", ".cc", ".cpp", ".cxx"}
        try:
            checked = 0
            for p in gen.repo_root.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                checked += 1
                if checked > 200:
                    break
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if re.search(r"\bint\s+main\s*\(", txt):
                    return True
        except Exception:
            return False
        return False

    def _inject_define_into_flag_list(text: str, define_flag: str) -> tuple[str, bool]:
        if define_flag in text:
            return text, False
        lines = text.splitlines()
        changed = False
        in_flags = False
        for i, line in enumerate(lines):
            if not in_flags and re.search(r"^\s*(?:CXXFLAGS|flags)\s*=\s*\[", line):
                in_flags = True
                continue
            if not in_flags:
                continue
            if re.search(r"^\s*\]", line):
                indent_match = re.match(r"^(\s*)", line)
                indent = indent_match.group(1) if indent_match else "    "
                lines.insert(i, f'{indent}"{define_flag}",')
                changed = True
                break
        if changed:
            return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), True
        # Fallback for common command pattern in generated build.py
        replaced = text.replace(
            " + [harness_cpp, VULNERABLE_CPP] + ",
            f" + ['{define_flag}', harness_cpp, VULNERABLE_CPP] + ",
        )
        if replaced != text:
            return replaced, True
        return text, False

    def _try_hotfix_stdlib_mismatch_and_main_conflict() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        abi_mismatch = any(
            token in diag
            for token in [
                "undefined reference to `std::__cxx11",
                "undefined reference to `std::",
                "vtable for std::",
                "libclang_rt.fuzzer",
            ]
        )
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        has_libcpp_flag = "-stdlib=libc++" in text
        multiple_main = ("multiple definition of `main'" in diag) or ("multiple definition of main" in diag)

        if not (abi_mismatch or has_libcpp_flag or multiple_main):
            return False

        changed = False
        # Avoid libc++/libstdc++ mismatch with clang/libFuzzer runtime in our base image.
        if has_libcpp_flag:
            text2 = text
            # Remove simple flag-list entries like:
            #   "-stdlib=libc++",
            #   '-stdlib=libc++',
            text2 = re.sub(r'^[ \t]*["\']-stdlib=libc\+\+["\'],?[ \t]*\n?', "", text2, flags=re.MULTILINE)
            # Remove conditional list entries like:
            #   ("-stdlib=libc++" if "clang" in cxx else ""),
            # without leaving broken syntax.
            text2 = re.sub(
                r'^[ \t]*\(\s*["\']-stdlib=libc\+\+["\']\s*if\s+.*?\s+else\s+["\']{0,1}["\']{0,1}\s*\)\s*,?[ \t]*\n?',
                "",
                text2,
                flags=re.MULTILINE,
            )
            # Repair previously broken malformed artifact:
            #   ( if "clang" in cxx else ""),
            text2 = re.sub(
                r'^[ \t]*\(\s*if\s+.*?\s+else\s+["\']{0,1}["\']{0,1}\s*\)\s*,?[ \t]*\n?',
                "",
                text2,
                flags=re.MULTILINE,
            )
            if text2 != text:
                text = text2
                changed = True

        # If sources define main(), rename it away from libFuzzer's main symbol.
        need_main_rename = multiple_main or _repo_has_c_cpp_main()
        if need_main_rename and "-Dmain=vuln_main" not in text:
            text, injected = _inject_define_into_flag_list(text, "-Dmain=vuln_main")
            changed = changed or injected

        # Keep legacy libFuzzer macro hotfix for compatibility with existing build.py patterns/tests.
        if multiple_main and "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION" not in text:
            text, injected = _inject_define_into_flag_list(text, "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION")
            changed = changed or injected

        if not changed:
            return False

        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(
                cast(dict[str, Any], state),
                "fix_build: applied local hotfix for stdlib mismatch/main conflict",
            )
            return True
        except Exception:
            return False

    def _try_hotfix_libfuzzer_main_conflict() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if "multiple definition of `main'" not in diag and "multiple definition of main" not in diag:
            return False

        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False

        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        define_flag = "-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION"
        if define_flag in text:
            return False

        lines = text.splitlines()
        changed = False
        in_flags = False
        for i, line in enumerate(lines):
            if not in_flags and re.search(r"^\s*flags\s*=\s*\[", line):
                in_flags = True
                continue
            if not in_flags:
                continue
            if "-fsanitize=fuzzer" in line:
                indent_match = re.match(r"^(\s*)", line)
                indent = indent_match.group(1) if indent_match else "        "
                lines.insert(i + 1, f"{indent}'{define_flag}',")
                changed = True
                break
            if re.search(r"^\s*\]", line):
                lines.insert(i, f"        '{define_flag}',")
                changed = True
                break

        if not changed:
            replaced = text.replace(
                "cmd = [cxx] + flags + [source_path, harness_path, '-o', output_path]",
                "cmd = [cxx, '-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION'] + flags + [source_path, harness_path, '-o', output_path]",
            )
            if replaced != text:
                text = replaced
                changed = True
            else:
                return False
        else:
            text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for libfuzzer main conflict")
            return True
        except Exception:
            return False

    def _try_hotfix_missing_lz() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if "cannot find -lz" not in diag and "undefined reference to `gz" not in diag and "undefined reference to `inflate" not in diag:
            return False

        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False

        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        changed = False
        if "import glob" not in text:
            if "import os" in text:
                text = text.replace("import os", "import os\nimport glob", 1)
                changed = True
            elif "import subprocess" in text:
                text = text.replace("import subprocess", "import os\nimport glob\nimport subprocess", 1)
                changed = True

        # Strengthen search path first.
        if "-L' + os.path.join(build_dir, 'lib')" not in text:
            text2 = text.replace(
                "lib_path = ['-L' + build_dir]",
                "lib_path = ['-L' + build_dir, '-L' + os.path.join(build_dir, 'lib')]",
            )
            if text2 != text:
                text = text2
                changed = True

        # Prefer explicit static archive path to avoid flaky '-lz' resolution in container builds.
        if "zlib_link_arg = '-lz'" not in text:
            marker = "libs = ['-lz']"
            if marker in text:
                inject = (
                    "zlib_link_arg = '-lz'\n"
                    "    zlib_candidates = [\n"
                    "        os.path.join(build_dir, 'libz.a'),\n"
                    "        os.path.join(build_dir, 'lib', 'libz.a'),\n"
                    "    ]\n"
                    "    for p in glob.glob(os.path.join(build_dir, '**', 'libz.a'), recursive=True):\n"
                    "        if p not in zlib_candidates:\n"
                    "            zlib_candidates.append(p)\n"
                    "    for p in zlib_candidates:\n"
                    "        if os.path.exists(p):\n"
                    "            zlib_link_arg = p\n"
                    "            break\n"
                    "    libs = [zlib_link_arg]"
                )
                text = text.replace(marker, inject, 1)
                changed = True

        # Generic fallback for scripts that embed '-lz' directly in command arrays.
        replaced = re.sub(r"(['\"])\\-lz\\1", "zlib_link_arg", text)
        if replaced != text:
            if "zlib_link_arg = '-lz'" not in replaced:
                # Keep insertion local and simple for ad-hoc scripts.
                if "def build_target(" in replaced:
                    replaced = replaced.replace(
                        "def build_target(",
                        "zlib_link_arg = '-lz'\n\n\ndef build_target(",
                        1,
                    )
                else:
                    replaced = "zlib_link_arg = '-lz'\n" + replaced
            text = replaced
            changed = True

        if not changed:
            return False

        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for missing -lz")
            return True
        except Exception:
            return False

    def _try_hotfix_collapsed_include_flags() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        # High-frequency generation issue: '-I/a -I/b' produced as one argv token.
        has_file_signal = bool(re.search(r"['\"][^'\"]*-I[^'\"]+\s+-I[^'\"]*['\"]", text))
        has_diag_signal = ("no such file or directory" in diag and " -i/" in diag)
        if not (has_file_signal or has_diag_signal):
            return False

        def _split_token(tok: str) -> str:
            parts = [x for x in tok.strip().split() if x]
            if len(parts) <= 1 or not all(x.startswith("-I") for x in parts):
                return tok
            return ", ".join(f"'{x}'" for x in parts)

        changed = False

        def _repl_single(m: re.Match[str]) -> str:
            nonlocal changed
            inner = m.group(1)
            out = _split_token(inner)
            if out != inner:
                changed = True
                return out
            return m.group(0)

        # Single-quoted combined include flags.
        text2 = re.sub(r"'([^']*-I[^']+\s+-I[^']*)'", lambda m: _repl_single(m), text)
        # Double-quoted combined include flags.
        text3 = re.sub(r"\"([^\"]*-I[^\"]+\s+-I[^\"]*)\"", lambda m: _repl_single(m), text2)
        if text3 != text:
            text = text3
        if not changed:
            return False
        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for collapsed include flags")
            return True
        except Exception:
            return False

    def _try_hotfix_compiler_fuzzer_flag_mismatch() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        has_diag_signal = ("-fsanitize=" in diag and "fuzzer" in diag and "unrecognized argument" in diag)
        has_file_signal = ("gcc" in text and "-fsanitize=fuzzer" in text)
        if not (has_diag_signal or has_file_signal):
            return False
        text2 = text.replace("'gcc'", "'clang'").replace('"gcc"', '"clang"')
        text2 = text2.replace("'g++'", "'clang++'").replace('"g++"', '"clang++"')
        if text2 == text:
            return False
        try:
            build_py.write_text(text2, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for compiler_fuzzer_flag_mismatch")
            return True
        except Exception:
            return False

    def _try_hotfix_missing_llvmfuzzer_entrypoint() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        has_diag_signal = "undefined reference to `llvmfuzzertestoneinput'" in diag
        has_file_signal = build_error_code == "missing_llvmfuzzer_entrypoint"
        if not (has_diag_signal or has_file_signal):
            return False

        fuzz_dir = gen.repo_root / "fuzz"
        cpp_exts = {".cc", ".cpp", ".cxx"}
        entry_pat = re.compile(r"(?m)^(\s*)int\s+LLVMFuzzerTestOneInput\s*\(")
        extern_entry_pat = re.compile(r'(?m)^\s*extern\s+"C"\s+int\s+LLVMFuzzerTestOneInput\s*\(')
        for src in fuzz_dir.rglob("*"):
            if not src.is_file() or src.suffix.lower() not in cpp_exts:
                continue
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "LLVMFuzzerTestOneInput" not in text:
                continue
            if extern_entry_pat.search(text):
                continue
            if not entry_pat.search(text):
                continue
            text2 = entry_pat.sub(r'\1extern "C" int LLVMFuzzerTestOneInput(', text, count=1)
            if text2 == text:
                continue
            try:
                src.write_text(text2, encoding="utf-8", errors="replace")
                _wf_log(
                    cast(dict[str, Any], state),
                    f"fix_build: applied local hotfix for missing_llvmfuzzer_entrypoint in {src.relative_to(gen.repo_root)}",
                )
                return True
            except Exception:
                continue

        build_py = fuzz_dir / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        changed = False
        # Fallback for scripts that compile C harnesses with clang++ and rely on
        # libFuzzer's C linkage entrypoint.
        if "'clang++'" in text and ".c" in text:
            text2 = text.replace("'clang++'", "'clang'").replace('"clang++"', '"clang"')
            if text2 != text:
                text = text2
                changed = True
        if changed and "'-x'" not in text and '"-x"' not in text and "flags = [" in text:
            text2 = text.replace("flags = [", "flags = ['-x', 'c', ", 1)
            if text2 != text:
                text = text2
        if not changed:
            return False
        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for missing_llvmfuzzer_entrypoint")
            return True
        except Exception:
            return False

    def _try_hotfix_cxx_for_c_source_mismatch() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if "treating 'c' input as 'c++'" not in diag and "treated as c++" not in diag:
            return False
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        text2 = text.replace("'clang++'", "'clang'").replace('"clang++"', '"clang"')
        if text2 == text:
            return False
        try:
            build_py.write_text(text2, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for cxx_for_c_source_mismatch")
            return True
        except Exception:
            return False

    def _try_hotfix_c_compiler_for_cpp_source_mismatch() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if (
            "invalid argument '-std=c++" not in diag
            and "this file requires compiler and library support for the iso c++" not in diag
            and "unknown type name 'namespace'" not in diag
        ):
            return False
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        has_cpp_signal = any(x in text for x in [".cc", ".cpp", ".cxx", "-std=c++"])
        if not has_cpp_signal:
            return False
        text2 = re.sub(r"(['\"])clang\1", r"\1clang++\1", text)
        text2 = re.sub(r"(['\"])gcc\1", r"\1g++\1", text2)
        if text2 == text:
            return False
        try:
            build_py.write_text(text2, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for c_compiler_for_cpp_source_mismatch")
            return True
        except Exception:
            return False

    def _try_hotfix_missing_symbol_include() -> bool:
        diag_raw = last_error + "\n" + stdout_tail + "\n" + stderr_tail
        if "undeclared identifier" not in diag_raw.lower():
            return False

        symbol_rules: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"^archive_entry_"), "#include <archive_entry.h>"),
            (re.compile(r"^archive_(read|write|format|filter|error|version|match|util|string)_"), "#include <archive.h>"),
        ]
        include_edits: dict[Path, set[str]] = {}
        for m in re.finditer(
            r"(?m)^(?P<file>[^:\n]+(?:\.cc|\.cpp|\.cxx|\.c)):\d+:\d+:\s+error:\s+use of undeclared identifier '(?P<sym>[A-Za-z_][A-Za-z0-9_]*)'",
            diag_raw,
        ):
            raw_file = str(m.group("file")).strip()
            sym = str(m.group("sym")).strip()
            if not raw_file or not sym:
                continue
            src = Path(raw_file)
            if not src.is_absolute():
                src = gen.repo_root / src
            if not src.is_file():
                continue
            include_line = ""
            for pat, inc in symbol_rules:
                if pat.search(sym):
                    include_line = inc
                    break
            if not include_line:
                continue
            include_edits.setdefault(src, set()).add(include_line)

        if not include_edits:
            return False

        for src, include_lines in include_edits.items():
            try:
                text = src.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            lines = text.splitlines()
            insert_at = 0
            for i, line in enumerate(lines):
                if line.lstrip().startswith("#include"):
                    insert_at = i + 1
            to_insert = [inc for inc in sorted(include_lines) if inc not in text]
            if not to_insert:
                continue
            for inc in to_insert:
                lines.insert(insert_at, inc)
                insert_at += 1
            new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
            if new_text == text:
                continue
            try:
                src.write_text(new_text, encoding="utf-8", errors="replace")
                _wf_log(cast(dict[str, Any], state), f"fix_build: applied local hotfix for missing include(s) in {src}")
                return True
            except Exception:
                continue
        return False

    def _try_hotfix_missing_system_packages() -> bool:
        alias_map = {
            "z": "zlib",
            "libz": "zlib",
            "libz-dev": "zlib",
            "zlib-dev": "zlib",
            "zlib1g": "zlib",
            "zlib1g-dev": "zlib",
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
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        if "cannot find -lz" in diag or "undefined reference to `gz" in diag or "undefined reference to `inflate" in diag:
            # Prefer dedicated link-fix rule for zlib linker failures.
            return False
        pkg_signals: list[tuple[list[str], str]] = [
            (["zlib.h", "could not find zlib", "cannot find -lz"], "zlib"),
            (["bzlib.h", "could not find bzip2"], "bzip2"),
            (["lzma.h", "could not find liblzma"], "liblzma"),
            (["zstd.h", "could not find zstd", "one of the modules 'libzstd'"], "zstd"),
            (["lz4.h", "could not find lz4"], "lz4"),
            (["openssl/", "could not find openssl"], "openssl"),
            (["expat.h", "could not find expat"], "expat"),
            (["libxml/parser.h", "could not find libxml2"], "libxml2"),
        ]
        need_pkgs: list[str] = []
        for needles, pkg in pkg_signals:
            if any(n in diag for n in needles):
                need_pkgs.append(pkg)
        if not need_pkgs:
            return False
        dep_file = gen.repo_root / "fuzz" / "system_packages.txt"
        existing: list[str] = []
        if dep_file.is_file():
            try:
                for line in dep_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    token = line.split("#", 1)[0].strip().lower()
                    if not token or not re.fullmatch(r"[a-z0-9][a-z0-9+._-]*", token):
                        continue
                    existing.append(alias_map.get(token, token))
            except Exception:
                return False
        merged = sorted(set(existing) | set(need_pkgs))
        if merged == sorted(set(existing)):
            return False
        dep_file.parent.mkdir(parents=True, exist_ok=True)
        body = (
            "# Auto-maintained by fix_build hotfix rules.\n"
            "# Package names are vcpkg ports (not apt package names).\n"
            + "\n".join(merged)
            + "\n"
        )
        try:
            dep_file.write_text(body, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), f"fix_build: declared system packages in {dep_file}")
            return True
        except Exception:
            return False
    def _try_hotfix_fuzz_out_path_mismatch() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        has_diag_signal = ("no fuzzer binaries found under fuzz/out" in diag or "build produced no fuzzers" in diag)
        has_file_signal = ('out_dir="fuzz/out"' in text)
        if not (has_diag_signal or has_file_signal):
            return False
        changed = False
        if 'out_dir="fuzz/out"' in text:
            text = text.replace('out_dir="fuzz/out"', 'out_dir="out"')
            changed = True
        if "os.path.abspath(out_dir)" not in text and "def build_all(" in text and "os.makedirs(out_dir" in text:
            text = text.replace("os.makedirs(out_dir, exist_ok=True)", "abs_out_dir = os.path.abspath(out_dir)\n    os.makedirs(abs_out_dir, exist_ok=True)")
            text = text.replace("compile_target(name, target_info, out_dir, cc)", "compile_target(name, target_info, abs_out_dir, cc)")
            changed = True
        if not changed:
            return False
        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for fuzz_out_path_mismatch")
            return True
        except Exception:
            return False

    def _try_hotfix_source_build_dir_collision() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        collision_signals = [
            "build/version",
            "build/cmake",
            "cmakelists.txt: could not find requested file",
            "include(cmake/checkfileoffsetbits.cmake)",
        ]
        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        uses_repo_build = (
            "BUILD_DIR = REPO_ROOT / \"build\"" in text
            or "BUILD_DIR=REPO_ROOT / \"build\"" in text
            or "BUILD_DIR = REPO_ROOT/'build'" in text
        )
        destructive_clean = ("shutil.rmtree(BUILD_DIR" in text or "rm -rf \"$BUILD_DIR\"" in text)
        if not ((any(sig in diag for sig in collision_signals) or uses_repo_build) and uses_repo_build and destructive_clean):
            return False

        new_text = text
        changed = False
        if "BUILD_DIR = REPO_ROOT / \"build\"" in new_text:
            new_text = new_text.replace(
                "BUILD_DIR = REPO_ROOT / \"build\"",
                "BUILD_DIR = REPO_ROOT / \"fuzz\" / \"build-work\"",
            )
            changed = True
        if "BUILD_DIR=REPO_ROOT / \"build\"" in new_text:
            new_text = new_text.replace(
                "BUILD_DIR=REPO_ROOT / \"build\"",
                "BUILD_DIR=REPO_ROOT / \"fuzz\" / \"build-work\"",
            )
            changed = True
        if "BUILD_DIR = REPO_ROOT/'build'" in new_text:
            new_text = new_text.replace(
                "BUILD_DIR = REPO_ROOT/'build'",
                "BUILD_DIR = REPO_ROOT/'fuzz'/'build-work'",
            )
            changed = True

        if new_text == text or not changed:
            return False
        try:
            build_py.write_text(new_text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for source_build_dir_collision")
            return True
        except Exception:
            return False

    def _try_hotfix_missing_cmake_archive_target() -> bool:
        diag = (last_error + "\n" + stdout_tail + "\n" + stderr_tail).lower()
        target_miss_signals = [
            "no rule to make target 'archive'",
            'no rule to make target "archive"',
            "unknown target archive",
        ]
        if not any(sig in diag for sig in target_miss_signals):
            return False

        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        changed = False
        new_text = text
        replacements = [
            ("'--target', 'archive'", "'--target', 'all'"),
            ('"--target", "archive"', '"--target", "all"'),
            ("'--target','archive'", "'--target','all'"),
            ('"--target","archive"', '"--target","all"'),
        ]
        for old, new in replacements:
            if old in new_text:
                new_text = new_text.replace(old, new)
                changed = True
        if changed and new_text != text:
            try:
                build_py.write_text(new_text, encoding="utf-8", errors="replace")
                _wf_log(cast(dict[str, Any], state), "fix_build: replaced cmake --target archive with --target all")
            except Exception:
                return False

        pkg_signals: list[tuple[list[str], str]] = [
            (['could not find zlib', 'zlib_library', 'zlib_include_dir'], 'zlib'),
            (['could not find bzip2', 'bzip2_libraries', 'bzip2_include_dir'], 'bzip2'),
            (['could not find liblzma', 'liblzma_library', 'liblzma_include_dir'], 'liblzma'),
            (['could not find lz4', 'lz4_library', 'lz4_include_dir'], 'lz4'),
            (['could not find zstd', "one of the modules 'libzstd'", 'zstd_library'], 'zstd'),
            (['could not find openssl', 'openssl_crypto_library', 'openssl_include_dir'], 'openssl'),
            (['could not find expat', 'expat_library', 'expat_include_dir'], 'expat'),
            (['could not find libxml2', 'libxml2_library', 'libxml2_include_dir'], 'libxml2'),
        ]
        need_pkgs: list[str] = []
        for needles, pkg in pkg_signals:
            if any(n in diag for n in needles):
                need_pkgs.append(pkg)
        if need_pkgs:
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
            dep_file = gen.repo_root / 'fuzz' / 'system_packages.txt'
            existing: list[str] = []
            if dep_file.is_file():
                try:
                    for line in dep_file.read_text(encoding='utf-8', errors='replace').splitlines():
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        token = line.split("#", 1)[0].strip().lower()
                        if not token or not re.fullmatch(r"[a-z0-9][a-z0-9+._-]*", token):
                            continue
                        existing.append(alias_map.get(token, token))
                except Exception:
                    existing = []
            merged = sorted(set(existing) | set(need_pkgs))
            if merged != sorted(set(existing)):
                dep_file.parent.mkdir(parents=True, exist_ok=True)
                body = (
                    '# Auto-maintained by fix_build hotfix rules.\n'
                    '# Package names are vcpkg ports (not apt package names).\n'
                    + '\n'.join(merged)
                    + '\n'
                )
                try:
                    dep_file.write_text(body, encoding='utf-8', errors='replace')
                    _wf_log(cast(dict[str, Any], state), f'fix_build: declared system packages in {dep_file}')
                    changed = True
                except Exception:
                    pass

        return changed

    def _try_hotfix_missing_decl() -> bool:
        """Detect implicit-function-declaration and undeclared-identifier errors
        in harness source files and add extern declarations or missing includes.

        Header-only / single-file libraries often expose internal symbols that
        the synthesize agent fails to declare.  This hotfix scans the build
        diagnostics for ``implicit declaration`` or ``undeclared identifier``
        messages, extracts the symbol name, and inserts the appropriate
        ``extern`` declaration into the offending harness source.
        """
        diag_raw = last_error + "\n" + stdout_tail + "\n" + stderr_tail
        diag_lower = diag_raw.lower()
        if "implicit declaration of function" not in diag_lower and "undeclared identifier" not in diag_lower:
            return False

        # Parse symbol and source file from clang diagnostic lines like:
        #   harness.c:42:5: error: call to undeclared function 'foo'
        #   harness.c:42:14: error: use of undeclared identifier 'BAR'
        import re as _re_local
        _changes = 0
        _diag_lines = diag_raw.splitlines()
        for _dl in _diag_lines:
            # Clang: "<file>:<line>:<col>: error: ... 'symbol'"
            _m = _re_local.search(
                r"(?P<file>[^\s:]+\.(?:c|cc|cpp|cxx)):\d+:\d+:\s+(?:error|fatal error):\s+.*?(?:undeclared identifier|implicit declaration of function).*?(?:'|\")(?P<sym>[A-Za-z_][A-Za-z0-9_]*)(?:'|\")",
                _dl,
            )
            if not _m:
                continue
            _src_name = _m.group("file")
            _symbol = _m.group("sym")
            _src_path = Path(_src_name)
            if not _src_path.is_absolute():
                _src_path = gen.repo_root / _src_path
            if not _src_path.is_file() or _src_path.parent.name != "fuzz":
                continue  # only patch harness sources in fuzz/

            try:
                _src_text = _src_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if f"extern " in _src_text and _symbol in _src_text:
                continue  # already has extern

            # Also check for well-known header hints
            _stb_header = 'stb' in _src_name.lower() or 'stb' in _symbol.lower()
            _include_line = ""
            if _stb_header:
                # stb single-file libs often need CGLTF_IMPLEMENTATION or STB_* before include
                pass  # fall through to extern declaration

            # Insert extern declaration after the last #include line
            _lines = _src_text.splitlines()
            _insert_at = 0
            for _j, _line in enumerate(_lines):
                if _line.lstrip().startswith("#include") or _line.lstrip().startswith("#define"):
                    _insert_at = _j + 1
            # Also after any #ifdef / #ifndef guards
            if _insert_at == 0:
                for _j, _line in enumerate(_lines):
                    if _line.lstrip().startswith("#if"):
                        _insert_at = _j + 1

            _decl = f"extern void {_symbol}(void);  /* sherpa-hotfix: forward declaration for header-only lib */"
            _lines.insert(_insert_at, _decl)
            _new_text = "\n".join(_lines) + ("\n" if _src_text.endswith("\n") else "")
            if _new_text == _src_text:
                continue

            try:
                _src_path.write_text(_new_text, encoding="utf-8", errors="replace")
                _changes += 1
                rel = str(_src_path.relative_to(gen.repo_root))
                _wf_log(cast(dict[str, Any], state), f"fix_build: added extern {_symbol}() to {rel}")
            except Exception:
                continue

        return _changes > 0

    def _try_hotfix_batch_internal_symbol_discovery() -> bool:
        """Resolve multiple `undefined reference` errors in one pass.

        When the linker reports >=2 distinct missing symbols, scan the repo's
        likely internal-library source directories (apps/lib, src/lib, etc.)
        for files that define those symbols, then append them all to the
        SOURCES list in fuzz/build.py in a single edit. This shortcuts the
        otherwise per-round whack-a-mole loop where the LLM adds one .c file
        per build attempt.
        """
        if (os.environ.get("SHERPA_FIX_BUILD_BATCH_DISCOVERY") or "1").strip().lower() in {"0", "false", "no", "off"}:
            return False

        diag = (last_error or "") + "\n" + (stdout_tail or "") + "\n" + (stderr_tail or "")
        import re as _re_local
        syms = sorted(set(_re_local.findall(r"undefined reference to [`'\"]([A-Za-z_][A-Za-z0-9_]*)[`'\"]", diag)))
        if len(syms) < 2:
            return False

        repo = gen.repo_root
        candidate_rels = ("apps/lib", "apps", "src/lib", "lib", "src/internal", "core", "src")
        cand_dirs: list[Path] = []
        for rel in candidate_rels:
            d = repo / rel
            if not d.is_dir():
                continue
            try:
                c_count = sum(1 for _ in d.rglob("*.c"))
            except (OSError, PermissionError):
                continue
            if c_count >= 3:
                cand_dirs.append(d)
        if not cand_dirs:
            return False

        _SIMD_SUFFIX_SET = (
            "_neon", "_sse", "_sse2", "_sse3", "_sse4", "_sse41", "_sse42",
            "_ssse3", "_msa", "_avx", "_avx2", "_avx512", "_altivec", "_vsx",
        )

        def _is_skip(p: Path) -> bool:
            parts_lower = [x.lower() for x in p.parts]
            if any(seg in {"test", "tests", "demo", "demos", "examples", "example",
                            "build", "_build", "cmakefiles", "_deps", "_install"} for seg in parts_lower):
                return True
            stem_lower = p.stem.lower()
            if any(stem_lower.endswith(s) for s in _SIMD_SUFFIX_SET):
                return True
            return False

        sym_to_src: dict[str, Path] = {}
        for d in cand_dirs:
            try:
                c_files = list(d.rglob("*.c"))
            except (OSError, PermissionError):
                continue
            for c in c_files:
                if _is_skip(c):
                    continue
                missing = [s for s in syms if s not in sym_to_src]
                if not missing:
                    break
                try:
                    text = c.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                for sym in missing:
                    if _re_local.search(rf"(?m)^[A-Za-z_][\w\s\*]*\b{_re_local.escape(sym)}\s*\(", text):
                        sym_to_src[sym] = c
        if not sym_to_src:
            return False

        build_py = repo / "fuzz" / "build.py"
        if not build_py.is_file():
            return False
        try:
            build_text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        new_rel_paths = sorted({str(p.relative_to(repo)) for p in sym_to_src.values()})
        new_text, ok = _splice_sources_list(build_text, new_rel_paths)
        if not ok or new_text == build_text:
            return False

        try:
            build_py.write_text(new_text, encoding="utf-8", errors="replace")
        except Exception:
            return False

        added_count = sum(1 for p in new_rel_paths if p not in build_text)
        _wf_log(
            cast(dict[str, Any], state),
            f"fix_build: batch-discovered {added_count} sources for {len(sym_to_src)}/{len(syms)} undef symbols",
        )
        return True

    if _fix_build_ruleset() == "extended":
        if _try_hotfix_compiler_fuzzer_flag_mismatch():
            out = _success_out(
                "local hotfix for compiler_fuzzer_flag_mismatch applied",
                outcome="rule_fixed",
                rule_hit="compiler_fuzzer_flag_mismatch",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_cxx_for_c_source_mismatch():
            out = _success_out(
                "local hotfix for cxx_for_c_source_mismatch applied",
                outcome="rule_fixed",
                rule_hit="cxx_for_c_source_mismatch",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_collapsed_include_flags():
            out = _success_out(
                "local hotfix for collapsed include flags applied",
                outcome="rule_fixed",
                rule_hit="collapsed_include_flags",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_missing_llvmfuzzer_entrypoint():
            out = _success_out(
                "local hotfix for missing_llvmfuzzer_entrypoint applied",
                outcome="rule_fixed",
                rule_hit="missing_llvmfuzzer_entrypoint",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_fuzz_out_path_mismatch():
            out = _success_out(
                "local hotfix for fuzz_out_path_mismatch applied",
                outcome="rule_fixed",
                rule_hit="fuzz_out_path_mismatch",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_source_build_dir_collision():
            out = _success_out(
                "local hotfix for source_build_dir_collision applied",
                outcome="rule_fixed",
                rule_hit="source_build_dir_collision",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_missing_cmake_archive_target():
            out = _success_out(
                "local hotfix for missing_cmake_archive_target applied",
                outcome="rule_fixed",
                rule_hit="missing_cmake_archive_target",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_c_compiler_for_cpp_source_mismatch():
            out = _success_out(
                "local hotfix for c_compiler_for_cpp_source_mismatch applied",
                outcome="rule_fixed",
                rule_hit="c_compiler_for_cpp_source_mismatch",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_missing_symbol_include():
            out = _success_out(
                "local hotfix for missing symbol include applied",
                outcome="rule_fixed",
                # Keep legacy rule name for compatibility with existing dashboards/tests.
                rule_hit="archive_entry_missing_include",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_missing_system_packages():
            out = _success_out(
                "local hotfix for missing system package declarations applied",
                outcome="rule_fixed",
                rule_hit="missing_system_packages_declared",
                last_diff_paths=["fuzz/system_packages.txt"],
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_batch_internal_symbol_discovery():
            out = _success_out(
                "local hotfix for batch internal symbol discovery applied",
                outcome="rule_fixed",
                rule_hit="batch_internal_symbol_discovery",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        if _try_hotfix_missing_decl():
            out = _success_out(
                "local hotfix for missing declarations in harness source applied",
                outcome="rule_fixed",
                rule_hit="missing_harness_decl",
            )
            _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

    if _try_hotfix_stdlib_mismatch_and_main_conflict():
        out = _success_out(
            "local hotfix for stdlib mismatch/main conflict applied",
            outcome="rule_fixed",
            rule_hit="stdlib_mismatch_or_abi",
        )
        _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    if _try_hotfix_libfuzzer_main_conflict():
        out = _success_out(
            "local hotfix for libfuzzer main conflict applied",
            outcome="rule_fixed",
            rule_hit="main_symbol_conflict",
        )
        _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    if _try_hotfix_missing_lz():
        out = _success_out(
            "local hotfix for -lz applied",
            outcome="rule_fixed",
            rule_hit="missing_zlib_link_flag",
        )
        _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    stdout_for_summary = re.sub(r"(?m)^\[\s*\d+%]\s+Built target\s+.*$", "", stdout_tail).strip()
    summary = _summarize_build_error(last_error, stdout_for_summary, stderr_tail)
    recent_history = history[-history_limit:] if history else []
    build_strategy_doc = _load_build_strategy_doc(gen.repo_root)
    build_runtime_facts_doc = _load_build_runtime_facts_doc(gen.repo_root)
    repo_understanding_doc = _load_repo_understanding_doc(gen.repo_root)

    # Ask an LLM to draft an *OpenCode instruction* tailored to the diagnostics.
    llm = _llm_or_none()
    codex_hint = (state.get("codex_hint") or "").strip()
    prompt_render_issue = ""

    targeted_fix_lines = _build_file_targeted_fix_lines(gen.repo_root, last_error, stdout_tail, stderr_tail)
    if not codex_hint:
        if llm is not None:
            coordinator_prompt = (
                "You are coordinating OpenCode to fix a fuzz harness build.\n"
                "Given the build diagnostics, produce a short instruction for OpenCode.\n\n"
                "Requirements for your output:\n"
                "- Output JSON only: {\"codex_hint\": \"...\"}\n"
                "- codex_hint must be 1-10 lines, concrete and minimal.\n"
                "- codex_hint must be in English only.\n"
                "- Extract concrete failing file paths from diagnostics when possible.\n"
                "- Include at least one line in the form: `Read and fix <path>[:line]` when file evidence exists.\n"
                "- Tell OpenCode to only change files under fuzz/.\n"
                "- Any change outside fuzz/ (except ./done sentinel) is rejected.\n"
                + (f"- Tell OpenCode to read full build logs from `{build_log_file}` before editing.\n" if build_log_file else "")
                + "- IMPORTANT: Tell OpenCode to NOT run any commands — only edit files.\n"
                "- Acceptance: `(cd fuzz && python build.py)` succeeds and leaves at least one executable in fuzz/out/.\n\n"
                f"repo_root={repo_root}\n"
                + f"error_type={summary['error_type']}\n"
                + (f"build_log_file={build_log_file}\n" if build_log_file else "")
                + (f"last_error={last_error}\n" if last_error else "")
                + ("\n=== STDOUT (tail) ===\n" + stdout_tail + "\n" if stdout_tail else "")
                + ("\n=== STDERR (tail) ===\n" + stderr_tail + "\n" if stderr_tail else "")
                + "\n=== STRUCTURED EVIDENCE ===\n" + summary["evidence"] + "\n"
                + "\nReturn JSON only."
            )
            try:
                resp = llm.invoke(coordinator_prompt)
                text = getattr(resp, "content", None) or str(resp)
                obj = _extract_json_object(text) or {}
                codex_hint = str(obj.get("codex_hint") or "").strip()
            except Exception:
                codex_hint = ""
            if codex_hint and _contains_cjk_text(codex_hint):
                codex_hint = ""

        if not codex_hint:
            codex_hint = (
                (f"First read `{build_log_file}` for the complete build logs, then apply the minimal fix.\n" if build_log_file else "")
                +
                "Fix the fuzz build so that running `(cd fuzz && python build.py)` succeeds and leaves at least one executable fuzzer under fuzz/out/.\n"
                "Keep the scaffold grounded in `fuzz/repo_understanding.json`; repair that file first if the current build path is underspecified.\n"
                "Only modify files under fuzz/. Any change outside fuzz/ (except ./done sentinel) will be rejected.\n"
                "Do not use `-stdlib=libc++` in this environment.\n"
                "If target sources define `main`, add a compile define such as `-Dmain=vuln_main` to avoid libFuzzer main conflicts.\n"
                "If include/link flags are wrong, fix them from fuzz/build.py or fuzz harness code only.\n"
                "Do not invoke repository-provided fuzz targets or guessed `--target ...fuzzer` build commands.\n"
                "Always build the repository library/objects and link the generated harness externally.\n"
                "Do not refactor production code or edit upstream source files."
            )
        if targeted_fix_lines:
            codex_hint = (codex_hint.strip() + "\n" + "\n".join(targeted_fix_lines)).strip()
        if build_error_code == "build_strategy_mismatch":
            codex_hint = (
                codex_hint.strip()
                + "\nThe current scaffold incorrectly depends on a repository fuzz target. Rewrite fuzz/build.py to avoid any repo fuzz target invocation and use external harness linking only."
            )
        if build_error_code == "missing_fuzzer_main":
            codex_hint = (
                codex_hint.strip()
                + "\nThe current scaffold is missing a fuzzer main strategy. Add `-fsanitize=fuzzer` or explicitly compile a repo-provided main source as a normal source input."
            )
        if build_error_code == "insufficient_repo_understanding":
            codex_hint = (
                codex_hint.strip()
                + "\nThe current scaffold lacks grounded repository understanding. Repair `fuzz/repo_understanding.json` first with concrete build facts and evidence, then make `fuzz/build.py` match it."
            )
        if build_error_code == "non_public_api_usage":
            codex_hint = (
                codex_hint.strip()
                + "\nDiagnostics indicate non-public/internal API usage in harness code. Replace offending symbols with public/stable APIs first."
                + "\nIf no public API exists, declare `api_surface_exception` in `fuzz/repo_understanding.json` with non-empty `reason` and `evidence` (and optional `approved_symbols`)."
            )
        if recent_history and any("noop" in str(x.get("outcome") or "") for x in recent_history):
            codex_hint = (
                codex_hint.strip()
                + "\nPrevious attempts were no-op; this attempt MUST produce at least one meaningful change under fuzz/."
            )

    # Build an error-heavy, noise-reduced context for fix_build.
    def _tail_lines(text: str, n: int = 120) -> str:
        lines = str(text or "").replace("\r", "\n").splitlines()
        return "\n".join(lines[-n:]).strip()

    def _tail_chars(text: str, max_chars: int) -> str:
        s = str(text or "").strip()
        if max_chars <= 0 or len(s) <= max_chars:
            return s
        return s[-max_chars:]

    def _denoise_build_stdout(text: str) -> str:
        lines = str(text or "").replace("\r", "\n").splitlines()
        if not lines:
            return ""
        noise_patterns = [
            re.compile(r"^\[\s*\d+%]\s+Built target\s+", re.IGNORECASE),
            re.compile(r"^--\s*Configuring done\b", re.IGNORECASE),
            re.compile(r"^--\s*Generating done\b", re.IGNORECASE),
            re.compile(r"^--\s*Build files have been written to:\b", re.IGNORECASE),
            re.compile(r"^\s*done\s*$", re.IGNORECASE),
        ]
        kept: list[str] = []
        for ln in lines:
            raw = ln.rstrip("\n")
            if any(p.search(raw.strip()) for p in noise_patterns):
                continue
            kept.append(raw)
        return "\n".join(kept).strip()

    def _dedupe_stderr_blocks(text: str, keep_recent: int) -> str:
        raw = str(text or "").replace("\r", "\n").strip()
        if not raw:
            return ""
        blocks = [b.strip() for b in re.split(r"\n{2,}", raw) if b.strip()]
        if not blocks:
            return raw

        seen_first: dict[str, int] = {}
        seen_latest: dict[str, int] = {}
        for i, block in enumerate(blocks):
            first_line = next((ln.strip().lower() for ln in block.splitlines() if ln.strip()), "")
            key = first_line[:220] or block[:220].lower()
            if key not in seen_first:
                seen_first[key] = i
            seen_latest[key] = i

        selected: list[int] = []
        for key, first_idx in seen_first.items():
            selected.append(first_idx)
            last_idx = seen_latest.get(key, first_idx)
            if last_idx != first_idx:
                selected.append(last_idx)
        selected = sorted(set(selected))
        if keep_recent > 0 and len(selected) > keep_recent * 2:
            selected = selected[-(keep_recent * 2) :]
        return "\n\n".join(blocks[i] for i in selected).strip()

    explicit_actions = [
        line.strip()
        for line in codex_hint.splitlines()
        if line.strip().lower().startswith("read and fix ")
    ]
    if not explicit_actions:
        explicit_actions = [line.strip() for line in targeted_fix_lines if line.strip()]

    previous_failed_attempts: list[dict[str, Any]] = []
    for row in recent_history[-context_history_limit:]:
        outcome = str(row.get("outcome") or "").strip()
        if not outcome:
            continue
        previous_failed_attempts.append(
            {
                "attempt": int(row.get("attempt_index") or 0),
                "outcome": outcome,
                "changed_paths_count": int(row.get("changed_paths_count") or 0),
                "signature": str(row.get("classified_signature") or "").strip(),
                "error_code": str(row.get("build_error_code") or "").strip(),
                "reason": str(row.get("rejection_reason") or "").strip(),
            }
        )

    file_refs: dict[str, Any] = {}
    if build_log_file:
        file_refs["build_log_file"] = build_log_file
    if isinstance(build_strategy_doc, dict):
        file_refs["fuzz/build_strategy.json"] = {
            "build_system": str(build_strategy_doc.get("build_system") or ""),
            "build_mode": str(build_strategy_doc.get("build_mode") or ""),
            "fuzzer_entry_strategy": str(build_strategy_doc.get("fuzzer_entry_strategy") or ""),
            "library_targets": list(build_strategy_doc.get("library_targets") or [])[:5],
        }
    if isinstance(build_runtime_facts_doc, dict):
        file_refs["fuzz/build_runtime_facts.json"] = {
            "build_system": str(build_runtime_facts_doc.get("build_system") or ""),
            "build_mode": str(build_runtime_facts_doc.get("build_mode") or ""),
            "required_outputs": list(build_runtime_facts_doc.get("required_outputs") or [])[:5],
        }
    if isinstance(repo_understanding_doc, dict):
        file_refs["fuzz/repo_understanding.json"] = {
            "build_system": str(repo_understanding_doc.get("build_system") or ""),
            "chosen_target_api": str(repo_understanding_doc.get("chosen_target_api") or ""),
            "fuzzer_entry_strategy": str(repo_understanding_doc.get("fuzzer_entry_strategy") or ""),
        }

    stderr_text = _tail_chars(
        _dedupe_stderr_blocks(stderr_tail, keep_recent=_fix_build_keep_recent_errors()),
        _fix_build_stderr_max_chars(),
    )
    stdout_text = _tail_chars(_denoise_build_stdout(stdout_tail), _fix_build_stdout_max_chars())

    p0_blocks: list[str] = []
    if stderr_text:
        p0_blocks.append("=== build stderr diagnostics ===\n" + stderr_text)

    p1_blocks: list[str] = ["=== structured_error ===\n" + json.dumps(summary, ensure_ascii=False, indent=2)]
    if last_error:
        p1_blocks.append("=== last_error ===\n" + _tail_lines(last_error, n=80))

    p2_blocks: list[str] = []
    if previous_failed_attempts:
        p2_blocks.append(
            "=== previous_failed_attempts ===\n" + json.dumps(previous_failed_attempts, ensure_ascii=False, indent=2)
        )
    p3_blocks: list[str] = []
    if stdout_text:
        p3_blocks.append("=== build stdout relevant ===\n" + stdout_text)
    p4_blocks: list[str] = []
    if explicit_actions:
        p4_blocks.append("=== targeted_file_actions ===\n" + "\n".join(f"- {line}" for line in explicit_actions))
    if file_refs:
        p4_blocks.append("=== context_file_refs ===\n" + json.dumps(file_refs, ensure_ascii=False, indent=2))

    mandatory = p0_blocks + p1_blocks
    optional = p2_blocks + p3_blocks + p4_blocks

    packed: list[str] = []
    current_len = 0
    for block in mandatory:
        b = str(block or "").strip()
        if not b:
            continue
        sep = 2 if packed else 0
        packed.append(b)
        current_len += len(b) + sep
    for block in optional:
        b = str(block or "").strip()
        if not b:
            continue
        sep = 2 if packed else 0
        if current_len + len(b) + sep > context_max_chars:
            continue
        packed.append(b)
        current_len += len(b) + sep
    context = "\n\n".join(packed).strip()
    if len(context) > context_max_chars:
        context = context[:context_max_chars]

    prompt, render_issue = _render_opencode_prompt_safe(
        "fix_build_execute",
        fallback_name="plan_repair_build_with_hint",
        codex_hint=codex_hint.strip(),
        build_log_file=build_log_file or "fuzz/build_full.log",
        hint=codex_hint.strip(),
        fallback_hint=codex_hint.strip(),
    )
    if render_issue:
        prompt_render_issue = str(render_issue)
        _wf_log(cast(dict[str, Any], state), f"fix_build: prompt render degraded -> {render_issue}")

    try:
        _wf_log(cast(dict[str, Any], state), f"fix_build: running opencode (hint_lines={len(codex_hint.splitlines())})")
        gen.patcher.run_codex_command(
            prompt,
            additional_context=context or None,
            stage_skill="fix_build",
            timeout=_remaining_time_budget_sec(state),
            max_attempts=1,
            max_cli_retries=_opencode_cli_retries(),
        )
        post_step_hashes = _collect_fix_step_hashes()
        changed_paths = sorted(
            p
            for p in (set(baseline_step_hashes.keys()) | set(post_step_hashes.keys()))
            if baseline_step_hashes.get(p) != post_step_hashes.get(p)
        )
        effective_changed_paths = [p for p in changed_paths if str(p).strip().replace("\\", "/") != "done"]
        changed_paths_count = len(effective_changed_paths)
        llm_outcome = "llm_fixed" if changed_paths_count > 0 else "llm_noop"
        updated_history, updated_rule_hits = _append_attempt(
            llm_outcome,
            changed_paths_count=changed_paths_count,
        )
        next_noop_streak = prev_noop_streak + 1 if changed_paths_count == 0 else 0
        message = "opencode fixed build" if changed_paths_count > 0 else "opencode returned without code changes"
        last_error_text = "" if changed_paths_count > 0 else (last_error or "fix_build produced no file changes")
        fix_effect = "advanced" if changed_paths_count > 0 else "stalled"
        out = {
            **state,
            "last_step": "fix_build",
            "last_error": last_error_text,
            "codex_hint": "",
            "message": message,
            "fix_build_noop_streak": next_noop_streak,
            "fix_build_attempt_history": updated_history,
            "fix_build_rule_hits": updated_rule_hits,
            "fix_build_terminal_reason": "",
            "fix_build_last_diff_paths": effective_changed_paths,
            "fix_action_type": "opencode",
            "fix_effect": fix_effect,
        }
        out = _attach_prompt_render_status(out, issue=prompt_render_issue)
        if changed_paths_count == 0 and next_noop_streak >= max_noop_streak:
            out["failed"] = False
            out["fix_build_terminal_reason"] = "fix_build_noop_streak_exceeded"
            out["last_error"] = f"fix_build no-op streak exceeded ({max_noop_streak}); restart from plan"
            out["message"] = "fix_build no-op streak exceeded; restarting from plan"
            out["restart_to_plan"] = True
            out["restart_to_plan_reason"] = "fix_build_noop_streak_exceeded"
            out["restart_to_plan_stage"] = "fix_build"
            out["restart_to_plan_error_text"] = str(out.get("last_error") or "")
        if changed_paths_count > 0 and _requires_env_rebuild(effective_changed_paths):
            out["message"] = "opencode fixed build (requires env rebuild)"
            out["fix_effect"] = "requires_env_rebuild"
            out["fix_build_terminal_reason"] = "requires_env_rebuild"
        _wf_log(cast(dict[str, Any], out), f"<- fix_build ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        updated_history, updated_rule_hits = _append_attempt(
            "exception",
            rejection_reason=str(e),
            changed_paths_count=0,
        )
        out = {
            **state,
            "last_step": "fix_build",
            "last_error": str(e),
            "message": "opencode fix_build failed",
            "fix_build_attempt_history": updated_history,
            "fix_build_rule_hits": updated_rule_hits,
            "fix_build_last_diff_paths": [],
            "fix_action_type": "opencode",
            "fix_effect": "regressed",
        }
        out = _attach_prompt_render_status(out, issue=prompt_render_issue or str(e))
        _wf_log(cast(dict[str, Any], out), f"<- fix_build err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
