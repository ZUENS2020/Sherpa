from __future__ import annotations

import hashlib
import json
import os
import re
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from persistent_config import load_config

import workflow_common as _wf_common
import workflow_summary as _wf_summary

from fuzz_unharnessed_repo import (
    FuzzerRunResult,
    HarnessGeneratorError,
    NonOssFuzzHarnessGenerator,
    RepoSpec,
    snapshot_repo_text,
    write_patch_from_snapshot,
)


class FuzzWorkflowState(TypedDict, total=False):
    repo_url: str
    email: Optional[str]
    time_budget: int
    run_time_budget: int
    max_len: int
    docker_image: Optional[str]
    ai_key_path: str
    workflow_started_at: float
    resume_from_step: str
    resume_repo_root: str

    step_count: int
    max_steps: int
    last_step: str
    last_error: str
    build_rc: int
    build_stdout_tail: str
    build_stderr_tail: str
    build_full_log_path: str
    build_error_signature: str
    same_build_error_repeats: int
    build_error_kind: str
    build_error_code: str
    build_attempts: int
    fix_build_attempts: int
    codex_hint: str
    failed: bool
    repo_root: str
    run_rc: int
    crash_evidence: str
    run_error_kind: str
    run_details: list[dict[str, Any]]
    run_batch_plan: list[dict[str, Any]]
    last_crash_artifact: str
    last_fuzzer: str
    crash_signature: str
    same_crash_repeats: int
    crash_fix_attempts: int
    next: str
    fix_patch_path: str
    fix_patch_files: list[str]
    fix_patch_bytes: int
    summary_path: str
    summary_json_path: str
    plan_fix_on_crash: bool
    plan_max_fix_rounds: int


class FuzzWorkflowRuntimeState(FuzzWorkflowState, total=False):
    generator: NonOssFuzzHarnessGenerator
    crash_found: bool
    message: str


def _wf_log(state: dict[str, Any] | None, msg: str) -> None:
    _wf_common.wf_log(state, msg)


def _fmt_dt(seconds: float) -> str:
    return _wf_common.fmt_dt(seconds)


def _calc_parallel_batch_budget(
    *,
    pending_count: int,
    max_parallel: int,
    remaining_for_run: int,
    configured_run_time_budget: int,
) -> tuple[int, int, int]:
    rounds_left = (pending_count + max_parallel - 1) // max_parallel
    round_budget = min(configured_run_time_budget, max(1, remaining_for_run // max(1, rounds_left)))
    hard_timeout = min(max(60, round_budget + 120), max(60, remaining_for_run + 30))
    return rounds_left, round_budget, hard_timeout


def _llm_or_none() -> ChatOpenAI | None:
    openai_key = os.environ.get("OPENAI_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    cfg = None
    if not (openai_key or openrouter_key):
        try:
            cfg = load_config()
            openai_key = cfg.openai_api_key or ""
            openrouter_key = cfg.openrouter_api_key or ""
        except Exception:
            cfg = None

    key = (openai_key or openrouter_key or "").strip()
    if not key:
        return None

    if openai_key and openai_key.strip():
        model = (
            os.environ.get("OPENAI_MODEL")
            or os.environ.get("OPENCODE_MODEL")
            or "deepseek-reasoner"
        ).strip()
        base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
        if not base_url and cfg is not None:
            base_url = (cfg.openai_base_url or "").strip()
    else:
        model = (os.environ.get("OPENROUTER_MODEL") or "").strip()
        base_url = (os.environ.get("OPENROUTER_BASE_URL") or "").strip()
        if cfg is not None:
            if not model:
                model = (cfg.openrouter_model or "").strip()
            if not base_url:
                base_url = (cfg.openrouter_base_url or "").strip()
        if not model:
            model = "anthropic/claude-3.5-sonnet"
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"

    # NOTE: langchain_openai.ChatOpenAI signature has changed across versions.
    # Build kwargs dynamically to avoid type-checker false positives.
    params: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": 600,
        "timeout": 30,
        "openai_api_key": key.strip(),
        "openai_api_base": base_url,
    }
    return ChatOpenAI(**params)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    return _wf_common.extract_json_object(text)


def _sha256_text(text: str) -> str:
    return _wf_common.sha256_text(text)


def _validate_targets_json(repo_root: Path) -> tuple[bool, str]:
    return _wf_common.validate_targets_json(repo_root)


def _summarize_build_error(last_error: str, stdout_tail: str, stderr_tail: str) -> dict[str, str]:
    return _wf_common.summarize_build_error(last_error, stdout_tail, stderr_tail)


def _classify_build_failure(
    last_error: str,
    stdout_tail: str,
    stderr_tail: str,
    *,
    build_rc: int,
    has_fuzzer_binaries: bool,
) -> tuple[str, str]:
    return _wf_common.classify_build_failure(
        last_error,
        stdout_tail,
        stderr_tail,
        build_rc=build_rc,
        has_fuzzer_binaries=has_fuzzer_binaries,
    )


def _build_failure_recovery_advice(error_kind: str, error_code: str) -> str:
    return _wf_common.build_failure_recovery_advice(error_kind, error_code)


def _collect_key_artifact_hashes(repo_root: Path) -> dict[str, str]:
    return _wf_common.collect_key_artifact_hashes(repo_root)


def _has_codex_key() -> bool:
    return _wf_common.has_codex_key()


def _slug_from_repo_url(repo_url: str) -> str:
    return _wf_common.slug_from_repo_url(repo_url)


def _alloc_output_workdir(repo_url: str) -> Path | None:
    return _wf_common.alloc_output_workdir(repo_url)


def _enter_step(state: FuzzWorkflowRuntimeState, step_name: str) -> tuple[FuzzWorkflowRuntimeState, bool]:
    out, stop = _wf_common.enter_step(cast(dict[str, Any], state), step_name)
    return cast(FuzzWorkflowRuntimeState, out), stop


def _remaining_time_budget_sec(state: FuzzWorkflowRuntimeState, *, min_timeout: int = 5) -> int:
    return _wf_common.remaining_time_budget_sec(cast(dict[str, Any], state), min_timeout=min_timeout)


def _time_budget_exceeded_state(state: FuzzWorkflowRuntimeState, *, step_name: str) -> FuzzWorkflowRuntimeState:
    return cast(FuzzWorkflowRuntimeState, _wf_common.time_budget_exceeded_state(cast(dict[str, Any], state), step_name=step_name))


def _make_plan_hint(repo_root: Path) -> str:
    return _wf_common.make_plan_hint(repo_root)


def _derive_plan_policy(repo_root: Path) -> tuple[bool, int]:
    return _wf_common.derive_plan_policy(repo_root)


def _load_opencode_prompt_templates() -> dict[str, str]:
    return _wf_common.load_opencode_prompt_templates()


def _render_opencode_prompt(name: str, **kwargs: object) -> str:
    return _wf_common.render_opencode_prompt(name, **kwargs)


@dataclass(frozen=True)
class FuzzWorkflowInput:
    repo_url: str
    email: Optional[str]
    time_budget: int
    run_time_budget: int
    max_len: int
    docker_image: Optional[str]
    ai_key_path: Path
    model: Optional[str] = None
    resume_from_step: Optional[str] = None
    resume_repo_root: Optional[Path] = None


def _node_init(state: FuzzWorkflowState) -> FuzzWorkflowRuntimeState:
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> init")
    repo_url = (state.get("repo_url") or "").strip()
    if not repo_url:
        raise ValueError("repo_url is required")

    ai_key_path = Path(state.get("ai_key_path") or "").expanduser().resolve()
    if not ai_key_path:
        raise ValueError("ai_key_path is required")

    time_budget = int(state.get("time_budget") or 900)
    run_time_budget = int(state.get("run_time_budget") or time_budget or 900)
    max_len = int(state.get("max_len") or 1024)
    docker_image = (state.get("docker_image") or "").strip()
    if not docker_image:
        raise ValueError("Docker execution is mandatory; docker_image is required")
    codex_cli = (os.environ.get("SHERPA_CODEX_CLI") or os.environ.get("CODEX_CLI") or "opencode").strip()

    raw_resume_repo_root = (state.get("resume_repo_root") or "").strip()
    workdir: Path | None = None
    if raw_resume_repo_root:
        candidate = Path(raw_resume_repo_root).expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            workdir = candidate
    if workdir is None:
        workdir = _alloc_output_workdir(repo_url)
    generator = NonOssFuzzHarnessGenerator(
        repo_spec=RepoSpec(url=repo_url, workdir=workdir),
        ai_key_path=ai_key_path,
        max_len=max_len,
        time_budget_per_target=run_time_budget,
        docker_image=docker_image,
        codex_cli=codex_cli,
    )

    out = cast(
        FuzzWorkflowRuntimeState,
        {
            **state,
            "generator": generator,
            "crash_found": False,
            "message": "initialized",
            "step_count": int(state.get("step_count") or 0),
            "max_steps": int(state.get("max_steps") or 10),
            "last_step": "init",
            "last_error": "",
            "build_rc": 0,
            "build_stdout_tail": "",
            "build_stderr_tail": "",
            "build_full_log_path": "",
            "build_error_signature": "",
            "same_build_error_repeats": 0,
            "build_error_kind": "",
            "build_error_code": "",
            "build_attempts": int(state.get("build_attempts") or 0),
            "fix_build_attempts": int(state.get("fix_build_attempts") or 0),
            "codex_hint": "",
            "failed": False,
            "repo_root": str(generator.repo_root),
            "run_rc": 0,
            "crash_evidence": "none",
            "run_error_kind": "",
            "last_crash_artifact": "",
            "last_fuzzer": "",
            "crash_signature": "",
            "same_crash_repeats": 0,
            "crash_fix_attempts": int(state.get("crash_fix_attempts") or 0),
            "plan_fix_on_crash": True,
            "plan_max_fix_rounds": 1,
        },
    )
    _wf_log(cast(dict[str, Any], out), f"<- init ok repo_root={out.get('repo_root')} dt={_fmt_dt(time.perf_counter()-t0)}")
    return out


def _node_plan(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "plan")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> plan")
    hint = (state.get("codex_hint") or "").strip()
    if not _has_codex_key():
        out = {
            **state,
            "last_step": "plan",
            "last_error": "Missing OPENAI_API_KEY for planning",
            "message": "plan failed",
        }
        _wf_log(cast(dict[str, Any], out), f"<- plan err=missing-key dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    try:
        if hint:
            prompt = _render_opencode_prompt("plan_with_hint", hint=hint)
            gen.patcher.run_codex_command(
                prompt,
                timeout=_remaining_time_budget_sec(state),
                max_attempts=1,
                max_cli_retries=1,
            )
        else:
            gen._pass_plan_targets(timeout=_remaining_time_budget_sec(state))

        strict_targets = (os.environ.get("SHERPA_PLAN_STRICT_TARGETS_SCHEMA", "1").strip().lower() in {"1", "true", "yes", "on"})
        ok_targets, targets_err = _validate_targets_json(gen.repo_root)
        if strict_targets and not ok_targets:
            _wf_log(cast(dict[str, Any], state), f"plan: targets.json schema invalid -> {targets_err}; retrying once")
            prompt = _render_opencode_prompt("plan_fix_targets_schema", schema_error=targets_err)
            gen.patcher.run_codex_command(
                prompt,
                timeout=_remaining_time_budget_sec(state),
                max_attempts=1,
                max_cli_retries=1,
            )
            ok_targets, targets_err = _validate_targets_json(gen.repo_root)
            if not ok_targets:
                out = {
                    **state,
                    "last_step": "plan",
                    "last_error": f"targets schema validation failed: {targets_err}",
                    "message": "plan failed",
                }
                _wf_log(cast(dict[str, Any], out), f"<- plan err=targets-schema dt={_fmt_dt(time.perf_counter()-t0)}")
                return out

        fix_on_crash, max_fix_rounds = _derive_plan_policy(gen.repo_root)
        out = {
            **state,
            "last_step": "plan",
            "last_error": "",
            "codex_hint": _make_plan_hint(gen.repo_root),
            "plan_fix_on_crash": fix_on_crash,
            "plan_max_fix_rounds": max_fix_rounds,
            "message": "planned",
        }
        _wf_log(cast(dict[str, Any], out), f"<- plan ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {**state, "last_step": "plan", "last_error": str(e), "message": "plan failed"}
        _wf_log(cast(dict[str, Any], out), f"<- plan err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_synthesize(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "synthesize")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> synthesize")
    hint = (state.get("codex_hint") or "").strip()

    def _has_min_synthesis_outputs() -> bool:
        fuzz_dir = gen.repo_root / "fuzz"
        build_py = fuzz_dir / "build.py"
        if not build_py.is_file():
            return False
        try:
            for p in fuzz_dir.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".java"} and "fuzz" in p.stem.lower():
                    return True
        except Exception:
            return False
        return False

    if not _has_codex_key():
        out = {
            **state,
            "last_step": "synthesize",
            "last_error": "Missing OPENAI_API_KEY for synthesis",
            "message": "synthesize failed",
        }
        _wf_log(cast(dict[str, Any], out), f"<- synthesize err=missing-key dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    try:
        if hint:
            prompt = _render_opencode_prompt("synthesize_with_hint", hint=hint)
            # Provide context from plan/targets if present.
            plan = (gen.repo_root / "fuzz" / "PLAN.md")
            targets = (gen.repo_root / "fuzz" / "targets.json")
            ctx = ""
            try:
                if plan.is_file():
                    ctx += "=== fuzz/PLAN.md ===\n" + plan.read_text(encoding="utf-8", errors="replace") + "\n\n"
                if targets.is_file():
                    ctx += "=== fuzz/targets.json ===\n" + targets.read_text(encoding="utf-8", errors="replace") + "\n"
            except Exception:
                pass
            gen.patcher.run_codex_command(
                prompt,
                additional_context=ctx or None,
                timeout=_remaining_time_budget_sec(state),
                max_attempts=1,
                max_cli_retries=1,
            )
            # Guardrail: if hint-mode synthesis produced nothing useful, retry once
            # with the canonical full synthesis prompt.
            if not _has_min_synthesis_outputs():
                _wf_log(
                    cast(dict[str, Any], state),
                    "synthesize: missing build.py/harness after hint-mode; retrying full synthesize",
                )
                gen._pass_synthesize_harness(timeout=_remaining_time_budget_sec(state))
        else:
            gen._pass_synthesize_harness(timeout=_remaining_time_budget_sec(state))

        if not _has_min_synthesis_outputs():
            raise HarnessGeneratorError("synthesize incomplete: missing fuzz/build.py or harness source")
        out = {**state, "last_step": "synthesize", "last_error": "", "codex_hint": "", "message": "synthesized"}
        _wf_log(cast(dict[str, Any], out), f"<- synthesize ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {**state, "last_step": "synthesize", "last_error": str(e), "message": "synthesize failed"}
        _wf_log(cast(dict[str, Any], out), f"<- synthesize err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


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
                    "Sherpa build full log\n"
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
        if getattr(gen, "docker_image", None):
            include_root = "/work"
            build_env.setdefault("CC", "clang")
            build_env.setdefault("CXX", "clang++")
            build_env.setdefault("CFLAGS", "-D_GNU_SOURCE")
            build_env.setdefault("CXXFLAGS", "-D_GNU_SOURCE")
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
            "last_step": "build",
        }
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
        max_same_repeats_raw = os.environ.get("SHERPA_WORKFLOW_MAX_SAME_BUILD_ERROR_REPEATS", "2")
        try:
            max_same_repeats = max(0, min(int(max_same_repeats_raw), 10))
        except Exception:
            max_same_repeats = 2

        if final_rc != 0:
            sig = _calc_build_error_signature()
            repeats = (prev_repeats + 1) if (prev_sig and prev_sig == sig) else 0
            next_state["build_error_signature"] = sig
            next_state["same_build_error_repeats"] = repeats
            next_state["build_error_kind"] = build_error_kind
            next_state["build_error_code"] = build_error_code
            advice = _build_failure_recovery_advice(build_error_kind, build_error_code)
            if repeats >= max_same_repeats:
                next_state["failed"] = True
                next_state["last_error"] = (
                    "build failed with the same error signature repeatedly "
                    f"(repeats={repeats + 1}, threshold={max_same_repeats + 1})"
                )
                next_state["message"] = "build failed repeatedly (same error)"
                _wf_log(cast(dict[str, Any], next_state), f"<- build stop same-error repeats={repeats+1}")
                return next_state
            next_state["last_error"] = f"build failed rc={final_rc} after {attempts_used} command run(s)"
            if advice:
                next_state["last_error"] += f"\nrecovery: {advice}"
            next_state["message"] = "build failed"
            _wf_log(cast(dict[str, Any], next_state), f"<- build fail rc={final_rc} dt={_fmt_dt(time.perf_counter()-t0)}")
            return next_state

        if not final_bins:
            sig = _calc_build_error_signature()
            repeats = (prev_repeats + 1) if (prev_sig and prev_sig == sig) else 0
            next_state["build_error_signature"] = sig
            next_state["same_build_error_repeats"] = repeats
            next_state["build_error_kind"] = build_error_kind
            next_state["build_error_code"] = build_error_code
            if repeats >= max_same_repeats:
                next_state["failed"] = True
                next_state["last_error"] = (
                    "build produced no fuzzers with the same diagnostics repeatedly "
                    f"(repeats={repeats + 1}, threshold={max_same_repeats + 1})"
                )
                next_state["message"] = "build failed repeatedly (no fuzzers)"
                _wf_log(cast(dict[str, Any], next_state), f"<- build stop same-error repeats={repeats+1}")
                return next_state
            next_state["last_error"] = f"No fuzzer binaries found under fuzz/out/ after {attempts_used} command run(s)"
            next_state["message"] = "build produced no fuzzers"
            _wf_log(cast(dict[str, Any], next_state), f"<- build fail no-fuzzers dt={_fmt_dt(time.perf_counter()-t0)}")
            return next_state

        next_state["build_error_signature"] = ""
        next_state["same_build_error_repeats"] = 0
        next_state["build_error_kind"] = ""
        next_state["build_error_code"] = ""
        next_state["fix_build_attempts"] = 0
        next_state["last_error"] = ""
        next_state["message"] = f"built ({len(final_bins)} fuzzers)"
        _wf_log(cast(dict[str, Any], next_state), f"<- build ok fuzzers={len(final_bins)} dt={_fmt_dt(time.perf_counter()-t0)}")
        return next_state
    except Exception as e:
        out = {
            **state,
            "last_step": "build",
            "last_error": str(e),
            "message": "build failed",
            "build_error_kind": "unknown",
            "build_error_code": "build_node_exception",
        }
        if "build_full_log_path" in locals():
            out["build_full_log_path"] = str(build_full_log_path)
        _wf_log(cast(dict[str, Any], out), f"<- build err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


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
    stop_on_infra = stop_on_infra_raw not in {"0", "false", "no", "off"}
    non_source_reason = ""
    if build_error_kind == "infra":
        non_source_reason = build_error_code or _detect_non_source_build_blocker(diag_text) or "infra_build_failure"
    else:
        non_source_reason = _detect_non_source_build_blocker(diag_text)
    if stop_on_infra and non_source_reason:
        out = {
            **state,
            "last_step": "fix_build",
            "failed": True,
            "build_error_kind": "infra",
            "build_error_code": non_source_reason,
            "last_error": f"non-source build blocker detected: {non_source_reason}",
            "message": "fix_build skipped (environment/infrastructure issue)",
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
            text2 = re.sub(r'^[ \t]*["\']-stdlib=libc\+\+["\'],?\s*\n?', "", text, flags=re.MULTILINE)
            text2 = text2.replace('"-stdlib=libc++",', "").replace('"-stdlib=libc++"', "")
            text2 = text2.replace("'-stdlib=libc++',", "").replace("'-stdlib=libc++'", "")
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
        if "cannot find -lz" not in diag:
            return False

        build_py = gen.repo_root / "fuzz" / "build.py"
        if not build_py.is_file():
            return False

        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        changed = False
        if "-L' + os.path.join(build_dir, 'lib')" not in text:
            text = text.replace(
                "lib_path = ['-L' + build_dir]",
                "lib_path = ['-L' + build_dir, '-L' + os.path.join(build_dir, 'lib')]",
            )
            changed = True

        if "lib_patterns = [" not in text and "libs = ['-lz']" in text:
            inject = (
                "libs = ['-lz']\n"
                "    # Hotfix: resolve zlib by full path when -lz is not discoverable\n"
                "    lib_patterns = [os.path.join(build_dir, 'libz.*'),\n"
                "                    os.path.join(build_dir, 'lib', 'libz.*'),\n"
                "                    os.path.join(build_dir, '**', 'libz.*')]\n"
                "    for pattern in lib_patterns:\n"
                "        matches = glob.glob(pattern, recursive=True)\n"
                "        if matches:\n"
                "            libs = [matches[0]]\n"
                "            lib_path = []\n"
                "            break"
            )
            text = text.replace("libs = ['-lz']", inject)
            changed = True

        if not changed:
            return False

        try:
            build_py.write_text(text, encoding="utf-8", errors="replace")
            _wf_log(cast(dict[str, Any], state), "fix_build: applied local hotfix for missing -lz")
            return True
        except Exception:
            return False

    if _try_hotfix_stdlib_mismatch_and_main_conflict():
        out = {
            **state,
            "last_step": "fix_build",
            "last_error": "",
            "codex_hint": "",
            "message": "local hotfix for stdlib mismatch/main conflict applied",
        }
        _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    if _try_hotfix_libfuzzer_main_conflict():
        out = {**state, "last_step": "fix_build", "last_error": "", "codex_hint": "", "message": "local hotfix for libfuzzer main conflict applied"}
        _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    if _try_hotfix_missing_lz():
        out = {**state, "last_step": "fix_build", "last_error": "", "codex_hint": "", "message": "local hotfix for -lz applied"}
        _wf_log(cast(dict[str, Any], out), f"<- fix_build hotfix ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    summary = _summarize_build_error(last_error, stdout_tail, stderr_tail)

    # Ask an LLM to draft an *OpenCode instruction* tailored to the diagnostics.
    llm = _llm_or_none()
    codex_hint = (state.get("codex_hint") or "").strip()

    if not codex_hint:
        if llm is not None:
            coordinator_prompt = (
                "You are coordinating OpenCode to fix a fuzz harness build.\n"
                "Given the build diagnostics, produce a short instruction for OpenCode.\n\n"
                "Requirements for your output:\n"
                "- Output JSON only: {\"codex_hint\": \"...\"}\n"
                "- codex_hint must be 1-10 lines, concrete and minimal.\n"
                "- Tell OpenCode to only change fuzz/ and minimal build glue.\n"
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

        if not codex_hint:
            codex_hint = (
                (f"First read `{build_log_file}` for the complete build logs, then apply the minimal fix.\n" if build_log_file else "")
                +
                "Fix the fuzz build so that running `(cd fuzz && python build.py)` succeeds and leaves at least one executable fuzzer under fuzz/out/.\n"
                "Only modify files under fuzz/ and the minimal build glue required.\n"
                "Do not use `-stdlib=libc++` in this environment.\n"
                "If target sources define `main`, add a compile define such as `-Dmain=vuln_main` to avoid libFuzzer main conflicts.\n"
                "If the harness source is wrong or missing includes/links, fix it. If build.py uses wrong target names or paths, correct it.\n"
                "Do not refactor production code."
            )

    # Now call OpenCode with a purpose-built prompt including diagnostics.
    context_parts: list[str] = []
    if build_log_file:
        context_parts.append("=== full build log file ===\n" + build_log_file)
    context_parts.append("=== structured_error ===\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    if last_error:
        context_parts.append("=== last_error ===\n" + last_error)
    if stdout_tail:
        context_parts.append("=== build stdout (tail) ===\n" + stdout_tail)
    if stderr_tail:
        context_parts.append("=== build stderr (tail) ===\n" + stderr_tail)
    context = "\n\n".join(context_parts)

    prompt = _render_opencode_prompt(
        "fix_build_execute",
        codex_hint=codex_hint.strip(),
        build_log_file=build_log_file or "fuzz/build_full.log",
    )

    try:
        _wf_log(cast(dict[str, Any], state), f"fix_build: running opencode (hint_lines={len(codex_hint.splitlines())})")
        gen.patcher.run_codex_command(
            prompt,
            additional_context=context or None,
            timeout=_remaining_time_budget_sec(state),
            max_attempts=1,
            max_cli_retries=1,
        )
        post_fix_hashes = _collect_fix_relevant_hashes()
        if post_fix_hashes == baseline_fix_hashes:
            out = {
                **state,
                "last_step": "fix_build",
                "last_error": "opencode fix_build made no relevant file changes",
                "message": "opencode fix_build no-op",
            }
            _wf_log(cast(dict[str, Any], out), f"<- fix_build err=no-op dt={_fmt_dt(time.perf_counter()-t0)}")
            return out
        out = {**state, "last_step": "fix_build", "last_error": "", "codex_hint": "", "message": "opencode fixed build"}
        _wf_log(cast(dict[str, Any], out), f"<- fix_build ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {**state, "last_step": "fix_build", "last_error": str(e), "message": "opencode fix_build failed"}
        _wf_log(cast(dict[str, Any], out), f"<- fix_build err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_run(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "run")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> run")
    try:
        # If we've already seen crashes in a previous round, archive old artifacts so
        # new crashes are detectable.
        fix_attempts = int(state.get("crash_fix_attempts") or 0)
        if fix_attempts:
            try:
                art_dir = gen.fuzz_out_dir / "artifacts"
                if art_dir.is_dir():
                    archive = art_dir / f"old-{fix_attempts}"
                    archive.mkdir(exist_ok=True)
                    for p in art_dir.glob("*"):
                        if p.is_file():
                            p.rename(archive / p.name)
            except Exception:
                pass

        bins = gen._discover_fuzz_binaries()
        if not bins:
            raise HarnessGeneratorError("No fuzzer binaries found under fuzz/out/")

        crash_found = False
        last_artifact = ""
        last_fuzzer = ""
        run_rc = 0
        crash_evidence = "none"
        run_error_kind = ""
        run_last_error = ""
        run_details: list[dict[str, Any]] = []
        run_batch_plan: list[dict[str, Any]] = []
        configured_run_time_budget = max(1, int(state.get("run_time_budget") or state.get("time_budget") or 900))
        prev_crash_sig = str(state.get("crash_signature") or "").strip()
        prev_crash_repeats = int(state.get("same_crash_repeats") or 0)
        max_same_crash_repeats_raw = os.environ.get("SHERPA_WORKFLOW_MAX_SAME_CRASH_REPEATS", "1")
        try:
            max_same_crash_repeats = max(0, min(int(max_same_crash_repeats_raw), 10))
        except Exception:
            max_same_crash_repeats = 1
        max_parallel_raw = os.environ.get("SHERPA_PARALLEL_FUZZERS", "2")
        try:
            max_parallel = max(1, min(int(max_parallel_raw), 16))
        except Exception:
            max_parallel = 2
        if len(bins) <= 1:
            max_parallel = 1

        _wf_log(cast(dict[str, Any], state), f"run: fuzzers={len(bins)} parallel={max_parallel}")

        def _calc_crash_signature(fuzzer_name: str, artifact_path: str) -> str:
            parts: list[str] = [f"fuzzer={fuzzer_name}", f"artifact={artifact_path}"]
            crash_info = gen.repo_root / "crash_info.md"
            crash_analysis = gen.repo_root / "crash_analysis.md"
            for p in (crash_info, crash_analysis):
                if not p.is_file():
                    continue
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                tail = "\n".join(txt.splitlines()[-400:])
                parts.append(f"== {p.name} ==\n{tail}")
            return _sha256_text("\n\n".join(parts))

        # Seed generation uses OpenCode and shared repo context; keep it serial.
        prev_seed_timeout = getattr(gen, "seed_generation_timeout_sec", None)
        try:
            for bin_path in bins:
                remaining_for_seed = _remaining_time_budget_sec(state, min_timeout=0)
                if remaining_for_seed <= 0:
                    return _time_budget_exceeded_state(state, step_name="run")
                setattr(gen, "seed_generation_timeout_sec", max(1, remaining_for_seed))
                fuzzer_name = bin_path.name
                try:
                    gen._pass_generate_seeds(fuzzer_name)
                except Exception as e:
                    # Seed generation is best-effort; do not block fuzzing.
                    print(f"[warn] seed generation skipped ({fuzzer_name}): {e}")
        finally:
            setattr(gen, "seed_generation_timeout_sec", prev_seed_timeout)

        run_results: dict[str, FuzzerRunResult] = {}
        run_exec_errors: dict[str, str] = {}

        def _run_one(bin_path: Path) -> tuple[str, FuzzerRunResult]:
            return bin_path.name, gen._run_fuzzer(bin_path)

        # Execute fuzzers in parallel batches and cap each batch to remaining total budget.
        pending_bins = list(bins)
        prev_run_budget = getattr(gen, "current_run_time_budget_sec", None)
        prev_run_hard_timeout = getattr(gen, "current_run_hard_timeout_sec", None)
        try:
            while pending_bins:
                remaining_for_run = _remaining_time_budget_sec(state, min_timeout=0)
                if remaining_for_run <= 0:
                    if not run_last_error:
                        run_last_error = "time budget exceeded during run phase"
                    if not run_error_kind:
                        run_error_kind = "workflow_time_budget_exceeded"
                    for skipped in pending_bins:
                        run_exec_errors[skipped.name] = "skipped: workflow total time budget exhausted before execution"
                    pending_bins = []
                    break

                rounds_left, round_budget, hard_timeout = _calc_parallel_batch_budget(
                    pending_count=len(pending_bins),
                    max_parallel=max_parallel,
                    remaining_for_run=remaining_for_run,
                    configured_run_time_budget=configured_run_time_budget,
                )
                setattr(gen, "current_run_time_budget_sec", round_budget)
                setattr(gen, "current_run_hard_timeout_sec", hard_timeout)

                batch = pending_bins[:max_parallel]
                pending_bins = pending_bins[max_parallel:]
                run_batch_plan.append(
                    {
                        "round": len(run_batch_plan) + 1,
                        "batch_size": len(batch),
                        "pending_before": len(batch) + len(pending_bins),
                        "rounds_left": rounds_left,
                        "remaining_total_budget_sec": remaining_for_run,
                        "round_budget_sec": round_budget,
                        "hard_timeout_sec": hard_timeout,
                    }
                )
                _wf_log(
                    cast(dict[str, Any], state),
                    (
                        "run batch: "
                        f"size={len(batch)} round_budget={round_budget}s hard_timeout={hard_timeout}s "
                        f"remaining_total={remaining_for_run}s"
                    ),
                )

                if len(batch) <= 1:
                    for bin_path in batch:
                        try:
                            name, run = _run_one(bin_path)
                            run_results[name] = run
                        except Exception as e:
                            run_exec_errors[bin_path.name] = str(e)
                else:
                    with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                        futures = {pool.submit(_run_one, bin_path): bin_path for bin_path in batch}
                        for fut in as_completed(futures):
                            bin_path = futures[fut]
                            try:
                                name, run = fut.result()
                                run_results[name] = run
                            except Exception as e:
                                run_exec_errors[bin_path.name] = str(e)
        finally:
            setattr(gen, "current_run_time_budget_sec", prev_run_budget)
            setattr(gen, "current_run_hard_timeout_sec", prev_run_hard_timeout)

        first_nonzero_rc = 0
        crash_candidates: list[tuple[str, Path, FuzzerRunResult]] = []

        for bin_path in bins:
            fuzzer_name = bin_path.name
            exec_err = run_exec_errors.get(fuzzer_name, "")
            if exec_err:
                if not run_last_error:
                    run_last_error = f"fuzzer run crashed for {fuzzer_name}: {exec_err}"
                if not run_error_kind:
                    run_error_kind = "run_exception"
                if first_nonzero_rc == 0:
                    first_nonzero_rc = 1
                run_details.append(
                    {
                        "fuzzer": fuzzer_name,
                        "rc": 1,
                        "crash_found": False,
                        "crash_evidence": "none",
                        "run_error_kind": "run_exception",
                        "error": exec_err,
                        "new_artifacts": [],
                        "first_artifact": "",
                        "final_cov": 0,
                        "final_ft": 0,
                        "final_iteration": 0,
                        "final_execs_per_sec": 0,
                        "final_rss_mb": 0,
                        "final_corpus_files": 0,
                        "final_corpus_size_bytes": 0,
                        "corpus_files": 0,
                        "corpus_size_bytes": 0,
                    }
                )
                continue

            run = run_results.get(fuzzer_name)
            if run is None:
                # Defensive fallback: if a future completed without result/exception.
                if not run_last_error:
                    run_last_error = f"missing run result for {fuzzer_name}"
                if not run_error_kind:
                    run_error_kind = "run_exception"
                if first_nonzero_rc == 0:
                    first_nonzero_rc = 1
                run_details.append(
                    {
                        "fuzzer": fuzzer_name,
                        "rc": 1,
                        "crash_found": False,
                        "crash_evidence": "none",
                        "run_error_kind": "run_exception",
                        "error": "missing run result",
                        "new_artifacts": [],
                        "first_artifact": "",
                        "final_cov": 0,
                        "final_ft": 0,
                        "final_iteration": 0,
                        "final_execs_per_sec": 0,
                        "final_rss_mb": 0,
                        "final_corpus_files": 0,
                        "final_corpus_size_bytes": 0,
                        "corpus_files": 0,
                        "corpus_size_bytes": 0,
                    }
                )
                continue

            rc = int(run.rc)
            if first_nonzero_rc == 0 and rc != 0:
                first_nonzero_rc = rc
            if not run_error_kind and run.run_error_kind:
                run_error_kind = run.run_error_kind

            run_details.append(
                {
                    "fuzzer": fuzzer_name,
                    "rc": rc,
                    "crash_found": bool(run.crash_found),
                    "crash_evidence": run.crash_evidence,
                    "run_error_kind": run.run_error_kind,
                    "error": run.error or "",
                    "new_artifacts": [str(p) for p in (run.new_artifacts or [])],
                    "first_artifact": run.first_artifact or "",
                    "final_cov": int(run.final_cov),
                    "final_ft": int(run.final_ft),
                    "final_iteration": int(run.final_iteration),
                    "final_execs_per_sec": int(run.final_execs_per_sec),
                    "final_rss_mb": int(run.final_rss_mb),
                    "final_corpus_files": int(run.final_corpus_files),
                    "final_corpus_size_bytes": int(run.final_corpus_size_bytes),
                    "corpus_files": int(run.corpus_files),
                    "corpus_size_bytes": int(run.corpus_size_bytes),
                }
            )
            if run.error and not run_last_error:
                run_last_error = run.error
            if run.crash_found and run.first_artifact:
                crash_candidates.append((fuzzer_name, Path(run.first_artifact), run))

        if crash_candidates:
            last_fuzzer, first, crash_run = crash_candidates[0]
            gen._analyze_and_package(last_fuzzer, first)
            crash_found = True
            last_artifact = str(first)
            run_rc = int(crash_run.rc)
            crash_evidence = crash_run.crash_evidence
        else:
            run_rc = first_nonzero_rc
            crash_evidence = "none"

        if crash_found:
            msg = "Fuzzing completed (crash found and packaged)."
        elif run_last_error:
            msg = "Fuzzing run failed."
        else:
            msg = "Fuzzing completed."

        crash_signature = ""
        same_crash_repeats = 0
        if crash_found and last_fuzzer and last_artifact:
            crash_signature = _calc_crash_signature(last_fuzzer, last_artifact)
            same_crash_repeats = (prev_crash_repeats + 1) if (prev_crash_sig and crash_signature == prev_crash_sig) else 0

        out = {
            **state,
            "last_step": "run",
            "last_error": run_last_error,
            "crash_found": crash_found,
            "run_rc": run_rc,
            "crash_evidence": crash_evidence,
            "run_error_kind": run_error_kind,
            "run_details": run_details,
            "run_batch_plan": run_batch_plan,
            "last_crash_artifact": last_artifact,
            "last_fuzzer": last_fuzzer,
            "crash_signature": crash_signature,
            "same_crash_repeats": same_crash_repeats,
            "message": msg,
        }
        if run_error_kind == "workflow_time_budget_exceeded":
            out["failed"] = True
            out["last_error"] = out.get("last_error") or "time budget exceeded during run phase"
            out["message"] = "workflow stopped (time budget exceeded)"
        if crash_found and same_crash_repeats >= max_same_crash_repeats:
            out["failed"] = True
            out["last_error"] = (
                "same crash signature repeated after crash-fix attempts "
                f"(repeats={same_crash_repeats + 1}, threshold={max_same_crash_repeats + 1})"
            )
            out["message"] = "workflow stopped (same crash repeated)"
        _wf_log(
            cast(dict[str, Any], out),
            (
                f"<- run ok crash_found={crash_found} rc={run_rc} evidence={crash_evidence} "
                f"same_crash_repeats={same_crash_repeats} dt={_fmt_dt(time.perf_counter()-t0)}"
            ),
        )
        return out
    except Exception as e:
        out = {**state, "last_step": "run", "last_error": str(e), "message": "run failed"}
        _wf_log(cast(dict[str, Any], out), f"<- run err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_fix_crash(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "fix_crash")
    if stop_now:
        return state

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> fix_crash")

    repo_root = gen.repo_root
    snapshot = snapshot_repo_text(repo_root)
    crash_info = repo_root / "crash_info.md"
    crash_analysis = repo_root / "crash_analysis.md"
    last_artifact = (state.get("last_crash_artifact") or "").strip()
    last_fuzzer = (state.get("last_fuzzer") or "").strip()

    info_text = crash_info.read_text(encoding="utf-8", errors="replace") if crash_info.is_file() else ""
    analysis_text = crash_analysis.read_text(encoding="utf-8", errors="replace") if crash_analysis.is_file() else ""
    harness_error = bool(re.search(r"HARNESS ERROR", analysis_text, re.IGNORECASE))

    if harness_error:
        prompt = _render_opencode_prompt("fix_crash_harness_error")
    else:
        prompt = _render_opencode_prompt("fix_crash_upstream_bug")

    ctx_parts: list[str] = []
    if last_fuzzer:
        ctx_parts.append(f"fuzzer: {last_fuzzer}")
    if last_artifact:
        ctx_parts.append(f"crashing_artifact: {last_artifact}")
    if info_text:
        ctx_parts.append("=== crash_info.md ===\n" + info_text)
    if analysis_text:
        ctx_parts.append("=== crash_analysis.md ===\n" + analysis_text)
    context = "\n\n".join(ctx_parts)

    attempts = int(state.get("crash_fix_attempts") or 0) + 1
    try:
        gen.patcher.run_codex_command(
            prompt,
            additional_context=context or None,
            timeout=_remaining_time_budget_sec(state),
            max_attempts=1,
            max_cli_retries=1,
        )
        patch_path = repo_root / "fix.patch"
        fix_summary_path = repo_root / "fix_summary.md"
        changed_files = write_patch_from_snapshot(snapshot, repo_root, patch_path)
        patch_bytes = patch_path.stat().st_size if patch_path.exists() else 0
        if not changed_files:
            out = {
                **state,
                "last_step": "fix_crash",
                "last_error": "opencode fix_crash made no textual file changes",
                "crash_fix_attempts": attempts,
                "message": "opencode fix_crash no-op",
                "fix_patch_path": str(patch_path) if patch_path.exists() else "",
                "fix_patch_files": [],
                "fix_patch_bytes": int(patch_bytes),
            }
            _wf_log(cast(dict[str, Any], out), f"<- fix_crash err=no-op dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        # Write a concise fix summary for downstream triage.
        summary_lines = [
            "# Fix Patch Summary",
            "",
            f"- Fix type: {'harness_error' if harness_error else 'upstream_bug'}",
            f"- Patch file: {patch_path}",
            f"- Files changed: {len(changed_files)}",
            "",
        ]
        if changed_files:
            summary_lines.append("## Files")
            summary_lines.extend([f"- {p}" for p in changed_files])
        else:
            summary_lines.append("_No textual changes detected for patch generation._")
        fix_summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8", errors="replace")

        # If a challenge bundle already exists, attach patch artifacts.
        for child in repo_root.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith(("challenge_bundle", "false_positive", "unreproducible")):
                if patch_path.exists():
                    shutil.copy2(patch_path, child / patch_path.name)
                if fix_summary_path.exists():
                    shutil.copy2(fix_summary_path, child / fix_summary_path.name)

        out = {
            **state,
            "last_step": "fix_crash",
            "last_error": "",
            "crash_fix_attempts": attempts,
            "message": "opencode fixed crash" if not harness_error else "opencode fixed harness error",
            "fix_patch_path": str(patch_path) if patch_path.exists() else "",
            "fix_patch_files": changed_files,
            "fix_patch_bytes": int(patch_bytes),
        }
        _wf_log(cast(dict[str, Any], out), f"<- fix_crash ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {
            **state,
            "last_step": "fix_crash",
            "last_error": str(e),
            "crash_fix_attempts": attempts,
            "message": "opencode fix_crash failed",
        }
        _wf_log(cast(dict[str, Any], out), f"<- fix_crash err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _route_after_build_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")):
        return "stop"
    if not (state.get("last_error") or "").strip():
        return "run"
    if (state.get("build_error_kind") or "").strip().lower() == "infra":
        return "stop"
    return "fix_build"


def _route_after_init_state(state: FuzzWorkflowRuntimeState) -> str:
    if bool(state.get("failed")) or (state.get("last_error") or "").strip():
        return "stop"
    raw = (state.get("resume_from_step") or "").strip().lower()
    allowed = {"plan", "synthesize", "build", "fix_build", "run", "fix_crash"}
    if raw in allowed:
        return raw
    return "plan"


def build_fuzz_workflow() -> StateGraph:
    graph: StateGraph = StateGraph(FuzzWorkflowRuntimeState)

    graph.add_node("init", _node_init)
    graph.add_node("plan", _node_plan)
    graph.add_node("synthesize", _node_synthesize)
    graph.add_node("build", _node_build)
    graph.add_node("fix_build", _node_fix_build)
    graph.add_node("fix_crash", _node_fix_crash)
    graph.add_node("run", _node_run)

    graph.set_entry_point("init")

    def _route_after_plan(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")) or (state.get("last_error") or "").strip():
            return "stop"
        return "synthesize"

    def _route_after_synthesize(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")) or (state.get("last_error") or "").strip():
            return "stop"
        return "build"

    def _route_after_build(state: FuzzWorkflowRuntimeState) -> str:
        return _route_after_build_state(state)

    def _route_after_fix_build(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")):
            return "stop"
        if (state.get("last_error") or "").strip():
            return "stop"
        return "build"

    def _route_after_run(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")):
            return "stop"
        if bool(state.get("crash_found")):
            fix_on_crash = bool(state.get("plan_fix_on_crash", True))
            max_fix_rounds = max(0, int(state.get("plan_max_fix_rounds") or 1))
            attempts = int(state.get("crash_fix_attempts") or 0)
            if fix_on_crash and attempts < max_fix_rounds:
                return "fix_crash"
            return "stop"
        return "stop"

    def _route_after_fix_crash(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")):
            return "stop"
        if (state.get("last_error") or "").strip():
            return "stop"
        return "build"

    graph.add_conditional_edges(
        "init",
        _route_after_init_state,
        {
            "plan": "plan",
            "synthesize": "synthesize",
            "build": "build",
            "fix_build": "fix_build",
            "run": "run",
            "fix_crash": "fix_crash",
            "stop": END,
        },
    )
    graph.add_conditional_edges("plan", _route_after_plan, {"synthesize": "synthesize", "stop": END})
    graph.add_conditional_edges("synthesize", _route_after_synthesize, {"build": "build", "stop": END})
    graph.add_conditional_edges("build", _route_after_build, {"run": "run", "fix_build": "fix_build", "stop": END})
    graph.add_conditional_edges("fix_build", _route_after_fix_build, {"build": "build", "stop": END})
    graph.add_conditional_edges("run", _route_after_run, {"fix_crash": "fix_crash", "stop": END})
    graph.add_conditional_edges("fix_crash", _route_after_fix_crash, {"build": "build", "stop": END})

    return graph


def _detect_harness_error(repo_root: Path) -> bool:
    return _wf_summary.detect_harness_error(repo_root)


def _bytes_human(num_bytes: int) -> str:
    return _wf_summary.bytes_human(num_bytes)


def _tree_file_stats(root: Path) -> tuple[int, int]:
    return _wf_summary.tree_file_stats(root)


def _collect_fuzz_inventory(repo_root: Path) -> dict[str, Any]:
    return _wf_summary.collect_fuzz_inventory(repo_root)


def _write_run_summary(out: dict[str, Any]) -> None:
    _wf_summary.write_run_summary(out)


def run_fuzz_workflow(inp: FuzzWorkflowInput) -> str:
    resume_step = (inp.resume_from_step or "").strip().lower()
    resume_root = str(inp.resume_repo_root or "").strip()
    _wf_log(
        None,
        "workflow start "
        f"repo={inp.repo_url} docker_image={inp.docker_image or '(host)'} "
        f"time_budget={inp.time_budget}s run_time_budget={inp.run_time_budget}s "
        f"resume_step={resume_step or '-'} resume_root={resume_root or '-'}",
    )
    t0 = time.perf_counter()
    try:
        max_steps_env = int(os.environ.get("SHERPA_WORKFLOW_MAX_STEPS", "20"))
    except Exception:
        max_steps_env = 20
    max_steps = max(3, min(max_steps_env, 100))
    wf = build_fuzz_workflow().compile()
    raw: Any = wf.invoke(
        {
            "repo_url": inp.repo_url,
            "email": inp.email,
            "time_budget": inp.time_budget,
            "run_time_budget": inp.run_time_budget,
            "workflow_started_at": time.time(),
            "max_len": inp.max_len,
            "docker_image": inp.docker_image,
            "ai_key_path": str(inp.ai_key_path),
            "resume_from_step": (inp.resume_from_step or "").strip().lower(),
            "resume_repo_root": str(inp.resume_repo_root or ""),
            "max_steps": max_steps,
        }
    )
    out = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
    try:
        _write_run_summary(out)
    except Exception:
        pass
    msg = str(out.get("message") or "Fuzzing completed.").strip()
    if bool(out.get("failed")):
        _wf_log(out, f"workflow end status=failed dt={_fmt_dt(time.perf_counter()-t0)}")
        raise RuntimeError(msg or "workflow failed")
    # If we stopped due to an error but didn't mark failed, still surface it.
    last_error = str(out.get("last_error") or "").strip()
    if last_error and not bool(out.get("crash_found")):
        _wf_log(out, f"workflow end status=error dt={_fmt_dt(time.perf_counter()-t0)}")
        raise RuntimeError(last_error)

    _wf_log(out, f"workflow end status=ok dt={_fmt_dt(time.perf_counter()-t0)}")
    return msg
