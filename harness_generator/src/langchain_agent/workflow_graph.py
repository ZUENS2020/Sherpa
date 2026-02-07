from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from fuzz_unharnessed_repo import HarnessGeneratorError, NonOssFuzzHarnessGenerator, RepoSpec


class FuzzWorkflowState(TypedDict, total=False):
    repo_url: str
    email: Optional[str]
    time_budget: int
    max_len: int
    docker_image: Optional[str]
    ai_key_path: str

    step_count: int
    max_steps: int
    last_step: str
    last_error: str
    build_rc: int
    build_stdout_tail: str
    build_stderr_tail: str
    build_attempts: int
    codex_hint: str
    failed: bool
    repo_root: str
    next: str


class FuzzWorkflowRuntimeState(FuzzWorkflowState, total=False):
    generator: NonOssFuzzHarnessGenerator
    crash_found: bool
    message: str


_ALLOWED_NEXT = {"plan", "synthesize", "build", "fix_build", "run", "stop"}


def _wf_log(state: dict[str, Any] | None, msg: str) -> None:
    step_count = ""
    last_step = ""
    nxt = ""
    if state:
        step_count = str(state.get("step_count") or "")
        last_step = str(state.get("last_step") or "")
        nxt = str(state.get("next") or "")
    prefix = "[wf]"
    if step_count or last_step or nxt:
        prefix = f"[wf step={step_count or '-'} last={last_step or '-'} next={nxt or '-'}]"
    print(f"{prefix} {msg}")


def _fmt_dt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def _llm_or_none() -> ChatOpenAI | None:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key or not key.strip():
        return None

    model = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet").strip() or "anthropic/claude-3.5-sonnet"
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip() or "https://openrouter.ai/api/v1"

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
    if not text:
        return None
    # Try to find the first {...} block.
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        val = json.loads(blob)
    except Exception:
        return None
    return val if isinstance(val, dict) else None


def _fallback_next(state: FuzzWorkflowRuntimeState) -> str:
    last_step = (state.get("last_step") or "").strip()
    last_error = (state.get("last_error") or "").strip()
    # Simple deterministic router.
    if last_error:
        if "No fuzzer binaries" in last_error:
            return "synthesize"
        if "build" in last_step:
            return "fix_build"
        if "Codex" in last_error or "plan" in last_error.lower():
            return "plan"
        if "build" in last_error.lower():
            return "fix_build"

    # Forward progress if possible.
    if last_step == "":
        return "plan"
    if last_step == "plan":
        return "synthesize"
    if last_step == "synthesize":
        return "build"
    if last_step == "build":
        return "run"
    return "stop"


@dataclass(frozen=True)
class FuzzWorkflowInput:
    repo_url: str
    email: Optional[str]
    time_budget: int
    max_len: int
    docker_image: Optional[str]
    ai_key_path: Path


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
    max_len = int(state.get("max_len") or 1024)
    docker_image = state.get("docker_image")

    generator = NonOssFuzzHarnessGenerator(
        repo_spec=RepoSpec(url=repo_url),
        ai_key_path=ai_key_path,
        max_len=max_len,
        time_budget_per_target=time_budget,
        docker_image=docker_image,
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
            "build_attempts": int(state.get("build_attempts") or 0),
            "codex_hint": "",
            "failed": False,
            "repo_root": str(generator.repo_root),
        },
    )
    _wf_log(cast(dict[str, Any], out), f"<- init ok repo_root={out.get('repo_root')} dt={_fmt_dt(time.perf_counter()-t0)}")
    return out


def _node_plan(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> plan")
    hint = (state.get("codex_hint") or "").strip()
    try:
        if hint:
            prompt = (
                "You are coordinating a fuzz harness generation workflow.\n"
                "Perform the planning step and produce fuzz/PLAN.md and fuzz/targets.json as required.\n\n"
                "Additional instruction from coordinator:\n" + hint
            )
            gen.patcher.run_codex_command(prompt)
        else:
            gen._pass_plan_targets()
        out = {**state, "last_step": "plan", "last_error": "", "codex_hint": "", "message": "planned"}
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
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> synthesize")
    hint = (state.get("codex_hint") or "").strip()
    try:
        if hint:
            prompt = (
                "You are coordinating a fuzz harness generation workflow.\n"
                "Perform the synthesis step: create harness + fuzz/build.py + build glue under fuzz/.\n\n"
                "Additional instruction from coordinator:\n" + hint
            )
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
            gen.patcher.run_codex_command(prompt, additional_context=ctx or None)
        else:
            gen._pass_synthesize_harness()
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
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), f"-> build attempt={(int(state.get('build_attempts') or 0)+1)}")
    try:
        # Single build attempt (no Codex auto-fix here). If it fails, we'll route to fix_build.
        fuzz_dir = gen.repo_root / "fuzz"
        build_py = fuzz_dir / "build.py"
        build_sh = fuzz_dir / "build.sh"
        if build_py.is_file():
            if getattr(gen, "docker_image", None):
                cmd = [gen._python_runner(), "fuzz/build.py"]
            else:
                cmd = [gen._python_runner(), str(build_py)]
        elif build_sh.is_file():
            cmd = ["bash", "fuzz/build.sh"] if getattr(gen, "docker_image", None) else ["bash", str(build_sh)]
        else:
            raise HarnessGeneratorError("Missing fuzz/build.py (agent must create fuzz/build.py)")

        rc, out, err = gen._run_cmd(cmd, cwd=gen.repo_root, env=os.environ.copy(), timeout=7200)
        bins = gen._discover_fuzz_binaries() if rc == 0 else []

        def _tail(s: str, n: int = 120) -> str:
            lines = (s or "").replace("\r", "\n").splitlines()
            return "\n".join(lines[-n:]).strip()

        attempts = int(state.get("build_attempts") or 0) + 1
        next_state: FuzzWorkflowRuntimeState = {
            **state,
            "build_attempts": attempts,
            "build_rc": int(rc),
            "build_stdout_tail": _tail(out),
            "build_stderr_tail": _tail(err),
            "last_step": "build",
        }

        if rc != 0:
            next_state["last_error"] = f"build failed rc={rc}"
            next_state["message"] = "build failed"
            _wf_log(cast(dict[str, Any], next_state), f"<- build fail rc={rc} dt={_fmt_dt(time.perf_counter()-t0)}")
            return next_state

        if not bins:
            next_state["last_error"] = "No fuzzer binaries found under fuzz/out/ after build"
            next_state["message"] = "build produced no fuzzers"
            _wf_log(cast(dict[str, Any], next_state), f"<- build fail no-fuzzers dt={_fmt_dt(time.perf_counter()-t0)}")
            return next_state

        next_state["last_error"] = ""
        next_state["message"] = f"built ({len(bins)} fuzzers)"
        _wf_log(cast(dict[str, Any], next_state), f"<- build ok fuzzers={len(bins)} dt={_fmt_dt(time.perf_counter()-t0)}")
        return next_state
    except Exception as e:
        out = {**state, "last_step": "build", "last_error": str(e), "message": "build failed"}
        _wf_log(cast(dict[str, Any], out), f"<- build err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_fix_build(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")

    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> fix_build")

    last_error = (state.get("last_error") or "").strip()
    stdout_tail = (state.get("build_stdout_tail") or "").strip()
    stderr_tail = (state.get("build_stderr_tail") or "").strip()
    repo_root = str(gen.repo_root)

    # Ask an LLM to draft a *Codex instruction* tailored to the diagnostics.
    llm = _llm_or_none()
    codex_hint = (state.get("codex_hint") or "").strip()

    if not codex_hint:
        if llm is not None:
            coordinator_prompt = (
                "You are coordinating Codex to fix a fuzz harness build.\n"
                "Given the build diagnostics, produce a short instruction for Codex.\n\n"
                "Requirements for your output:\n"
                "- Output JSON only: {\"codex_hint\": \"...\"}\n"
                "- codex_hint must be 1-10 lines, concrete and minimal.\n"
                "- Tell Codex to only change fuzz/ and minimal build glue.\n"
                "- Acceptance: `python fuzz/build.py` succeeds and leaves at least one executable in fuzz/out/.\n\n"
                f"repo_root={repo_root}\n"
                + (f"last_error={last_error}\n" if last_error else "")
                + ("\n=== STDOUT (tail) ===\n" + stdout_tail + "\n" if stdout_tail else "")
                + ("\n=== STDERR (tail) ===\n" + stderr_tail + "\n" if stderr_tail else "")
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
                "Fix the fuzz build so that running `python fuzz/build.py` succeeds and leaves at least one executable fuzzer under fuzz/out/.\n"
                "Only modify files under fuzz/ and the minimal build glue required.\n"
                "If the harness source is wrong or missing includes/links, fix it. If build.py uses wrong target names or paths, correct it.\n"
                "Do not refactor production code."
            )

    # Now call Codex with a purpose-built prompt including diagnostics.
    context_parts: list[str] = []
    if last_error:
        context_parts.append("=== last_error ===\n" + last_error)
    if stdout_tail:
        context_parts.append("=== build stdout (tail) ===\n" + stdout_tail)
    if stderr_tail:
        context_parts.append("=== build stderr (tail) ===\n" + stderr_tail)
    context = "\n\n".join(context_parts)

    prompt = (
        "You are Codex operating inside a Git repository.\n"
        "Task: fix the fuzz harness/build so the build passes.\n\n"
        "Acceptance criteria:\n"
        "- `python fuzz/build.py` completes successfully\n"
        "- fuzz/out/ contains at least one runnable fuzzer binary\n\n"
        "Constraints:\n"
        "- Keep changes minimal; avoid refactors\n"
        "- Prefer edits under fuzz/ and minimal build glue only\n\n"
        "Coordinator instruction:\n"
        + codex_hint.strip()
        + "\n\nWhen finished, write `fuzz/build.py` into `./done`."
    )

    try:
        _wf_log(cast(dict[str, Any], state), f"fix_build: running codex (hint_lines={len(codex_hint.splitlines())})")
        gen.patcher.run_codex_command(prompt, additional_context=context or None)
        out = {**state, "last_step": "fix_build", "last_error": "", "codex_hint": "", "message": "codex fixed build"}
        _wf_log(cast(dict[str, Any], out), f"<- fix_build ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {**state, "last_step": "fix_build", "last_error": str(e), "message": "codex fix_build failed"}
        _wf_log(cast(dict[str, Any], out), f"<- fix_build err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_run(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> run")
    try:
        bins = gen._discover_fuzz_binaries()
        if not bins:
            raise HarnessGeneratorError("No fuzzer binaries found under fuzz/out/")

        crash_found = False
        for bin_path in bins:
            fuzzer_name = bin_path.name
            try:
                gen._pass_generate_seeds(fuzzer_name)
            except Exception as e:
                # Seed generation is best-effort; do not block fuzzing.
                print(f"[warn] seed generation skipped ({fuzzer_name}): {e}")

            new_artifacts = gen._run_fuzzer(bin_path)
            if new_artifacts:
                first = sorted(new_artifacts)[0]
                gen._analyze_and_package(fuzzer_name, first)
                crash_found = True
                break

        msg = "Fuzzing completed." if not crash_found else "Fuzzing completed (crash found and packaged)."
        out = {**state, "last_step": "run", "last_error": "", "crash_found": crash_found, "message": msg}
        _wf_log(cast(dict[str, Any], out), f"<- run ok crash_found={crash_found} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {**state, "last_step": "run", "last_error": str(e), "message": "run failed"}
        _wf_log(cast(dict[str, Any], out), f"<- run err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_decide(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    t0 = time.perf_counter()
    step_count = int(state.get("step_count") or 0) + 1
    max_steps = int(state.get("max_steps") or 10)
    state = {**state, "step_count": step_count}

    _wf_log(cast(dict[str, Any], state), f"-> decide (max_steps={max_steps})")

    if step_count >= max_steps:
        # Stop to avoid infinite loops.
        failed = bool(state.get("last_error")) and not bool(state.get("crash_found"))
        return {
            **state,
            "failed": failed,
            "next": "stop",
            "message": "workflow stopped (max steps reached)",
        }

    # If we already finished a run, stop by default.
    if (state.get("last_step") == "run") and not (state.get("last_error") or "").strip():
        return {**state, "next": "stop"}

    # If build failed, try fix_build by default (LLM can override).
    if (state.get("last_step") == "build") and (state.get("last_error") or "").strip():
        # Let LLM decide, but default fallback will route to fix_build.
        pass

    llm = _llm_or_none()
    if llm is None:
        out = {**state, "next": _fallback_next(state)}
        _wf_log(cast(dict[str, Any], out), f"<- decide fallback next={out.get('next')} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out

    last_error = (state.get("last_error") or "").strip()
    last_step = (state.get("last_step") or "").strip()
    crash_found = bool(state.get("crash_found"))
    repo_root = (state.get("repo_root") or "").strip()
    docker_image = (state.get("docker_image") or "").strip() or "(host)"

    prompt = (
        "You are a workflow coordinator for fuzz harness generation and fuzzing.\n"
        "You decide the next step and optionally provide a short instruction to guide Codex for that step.\n\n"
        "Constraints:\n"
        "- Allowed next steps: plan, synthesize, build, fix_build, run, stop\n"
        "- Only provide codex_hint when next is plan, synthesize, or fix_build\n"
        "- Keep codex_hint short and actionable (1-6 lines)\n"
        "- Output MUST be a single JSON object with keys: next, codex_hint (optional)\n\n"
        f"State summary:\n- repo_root: {repo_root}\n- docker_image: {docker_image}\n- last_step: {last_step}\n- crash_found: {crash_found}\n"
        + (f"- last_error: {last_error}\n" if last_error else "- last_error: (none)\n")
        + "\nReturn JSON only."
    )

    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        obj = _extract_json_object(text)
        if not obj:
            out = {**state, "next": _fallback_next(state)}
            _wf_log(cast(dict[str, Any], out), f"<- decide (llm) parse_fail next={out.get('next')} dt={_fmt_dt(time.perf_counter()-t0)}")
            return out

        nxt = str(obj.get("next") or "").strip().lower()
        if nxt not in _ALLOWED_NEXT:
            nxt = _fallback_next(state)

        hint = str(obj.get("codex_hint") or "").strip()
        if nxt not in {"plan", "synthesize", "fix_build"}:
            hint = ""

        out = {**state, "next": nxt, "codex_hint": hint}
        _wf_log(cast(dict[str, Any], out), f"<- decide (llm) next={nxt} hint={'yes' if bool(hint) else 'no'} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {**state, "next": _fallback_next(state), "last_error": last_error or str(e)}
        _wf_log(cast(dict[str, Any], out), f"<- decide err={e} next={out.get('next')} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def build_fuzz_workflow() -> StateGraph:
    graph: StateGraph = StateGraph(FuzzWorkflowRuntimeState)

    graph.add_node("init", _node_init)
    graph.add_node("decide", _node_decide)
    graph.add_node("plan", _node_plan)
    graph.add_node("synthesize", _node_synthesize)
    graph.add_node("build", _node_build)
    graph.add_node("fix_build", _node_fix_build)
    graph.add_node("run", _node_run)

    graph.set_entry_point("init")
    graph.add_edge("init", "decide")

    def _route(state: FuzzWorkflowRuntimeState) -> str:
        nxt = str(state.get("next") or "").strip().lower()
        return nxt if nxt in _ALLOWED_NEXT else "stop"

    graph.add_conditional_edges(
        "decide",
        _route,
        {
            "plan": "plan",
            "synthesize": "synthesize",
            "build": "build",
            "fix_build": "fix_build",
            "run": "run",
            "stop": END,
        },
    )

    graph.add_edge("plan", "decide")
    graph.add_edge("synthesize", "decide")
    graph.add_edge("build", "decide")
    graph.add_edge("fix_build", "build")
    graph.add_edge("run", "decide")

    return graph


def run_fuzz_workflow(inp: FuzzWorkflowInput) -> str:
    _wf_log(None, f"workflow start repo={inp.repo_url} docker_image={inp.docker_image or '(host)'} time_budget={inp.time_budget}s")
    t0 = time.perf_counter()
    wf = build_fuzz_workflow().compile()
    raw: Any = wf.invoke(
        {
            "repo_url": inp.repo_url,
            "email": inp.email,
            "time_budget": inp.time_budget,
            "max_len": inp.max_len,
            "docker_image": inp.docker_image,
            "ai_key_path": str(inp.ai_key_path),
            "max_steps": 10,
        }
    )
    out = cast(dict[str, Any], raw) if isinstance(raw, dict) else {}
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
