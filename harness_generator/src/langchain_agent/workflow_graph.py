from __future__ import annotations

import json
import os
import re
import time
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from persistent_config import load_config

from fuzz_unharnessed_repo import (
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
    last_crash_artifact: str
    last_fuzzer: str
    crash_fix_attempts: int
    next: str
    fix_patch_path: str
    fix_patch_files: list[str]
    fix_patch_bytes: int
    summary_path: str
    summary_json_path: str


class FuzzWorkflowRuntimeState(FuzzWorkflowState, total=False):
    generator: NonOssFuzzHarnessGenerator
    crash_found: bool
    message: str


_ALLOWED_NEXT = {"plan", "synthesize", "build", "fix_build", "fix_crash", "run", "stop"}


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


def _has_codex_key() -> bool:
    # Check for any available API key (OpenAI, OpenRouter, or OpenCode)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and openai_key.strip():
        return True
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key and openrouter_key.strip():
        return True
    try:
        cfg = load_config()
        if cfg.openai_api_key and cfg.openai_api_key.strip():
            return True
        if cfg.openrouter_api_key and cfg.openrouter_api_key.strip():
            return True
    except Exception:
        pass
    opencode_key = os.environ.get("OPENCODE_API_KEY")
    if opencode_key and opencode_key.strip():
        return True
    return False


def _is_tinyxml2_repo(repo_root: Path) -> bool:
    return (repo_root / "tinyxml2.h").is_file() or (repo_root / "tinyxml2.cpp").is_file()


def _slug_from_repo_url(repo_url: str) -> str:
    base = repo_url.rstrip("/").split("/")[-1]
    if base.endswith(".git"):
        base = base[: -len(".git")]
    base = re.sub(r"[^a-zA-Z0-9._-]+", "-", base).strip("-")
    return base or "repo"


def _alloc_output_workdir(repo_url: str) -> Path | None:
    out_root = os.environ.get("SHERPA_OUTPUT_DIR", "").strip()
    if not out_root:
        return None
    base = Path(out_root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    slug = _slug_from_repo_url(repo_url)
    return base / f"{slug}-{uuid.uuid4().hex[:8]}"


def _write_builtin_tinyxml2_plan(repo_root: Path) -> None:
    fuzz_dir = repo_root / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    plan_md = fuzz_dir / "PLAN.md"
    targets_json = fuzz_dir / "targets.json"
    plan_md.write_text(
        "\n".join(
            [
                "# Fuzz Plan",
                "",
                "Primary fuzzer: tinyxml2_fuzz",
                "",
                "Target: tinyxml2::XMLDocument::Parse(const char*, size_t)",
                "",
                "Rationale:",
                "- Parses attacker-controlled XML strings",
                "- High-level entrypoint used by typical consumers",
                "- Minimal initialization required",
                "",
            ]
        ),
        encoding="utf-8",
    )
    targets_json.write_text(
        json.dumps(
            [
                {
                    "name": "tinyxml2_fuzz",
                    "api": "tinyxml2::XMLDocument::Parse",
                    "lang": "c-cpp",
                    "proto": "const uint8_t*,size_t",
                    "build_target": "libtinyxml2",
                    "reason": "Top-level XML parser entrypoint with attacker-controlled input.",
                    "evidence": ["tinyxml2.cpp"],
                }
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_builtin_tinyxml2_scaffold(repo_root: Path, max_len: int) -> None:
    fuzz_dir = repo_root / "fuzz"
    fuzz_out = fuzz_dir / "out"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    fuzz_out.mkdir(parents=True, exist_ok=True)

    fuzzer_cc = """#include <stdint.h>
#include <stddef.h>
#include "tinyxml2.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  tinyxml2::XMLDocument doc;
  doc.Parse(reinterpret_cast<const char*>(data), size);
  return 0;
}
"""
    build_py = f"""#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path

def run(cmd, cwd):
    print("[build] " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def find_lib(build_dir: Path) -> Path:
    for p in build_dir.rglob("libtinyxml2.a"):
        return p
    for p in build_dir.rglob("tinyxml2.lib"):
        return p
    raise RuntimeError("libtinyxml2 not found under build/")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    build_dir = repo / "build"
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)
    cxx = os.environ.get("CXX", "clang++")
    cc = os.environ.get("CC", "clang")

    run([
        "cmake",
        "-S", ".",
        "-B", str(build_dir),
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DBUILD_SHARED_LIBS=OFF",
        f"-DCMAKE_C_COMPILER={{cc}}",
        f"-DCMAKE_CXX_COMPILER={{cxx}}",
    ], cwd=repo)

    run(["cmake", "--build", str(build_dir), "--config", "Release"], cwd=repo)

    lib = find_lib(build_dir)
    out_dir = repo / "fuzz" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    fuzzer_src = repo / "fuzz" / "tinyxml2_fuzz.cc"
    bin_path = out_dir / "tinyxml2_fuzz"

    flags = [
        "-std=c++11",
        "-g",
        "-O1",
        "-fno-omit-frame-pointer",
        "-fsanitize=fuzzer,address",
    ]

    run([
        cxx,
        *flags,
        "-I", str(repo),
        "-o", str(bin_path),
        str(fuzzer_src),
        str(lib),
    ], cwd=repo)

if __name__ == "__main__":
    main()
"""
    readme = f"""# Local fuzz scaffold (tinyxml2)

1. Build: `python fuzz/build.py`
2. Run: `fuzz/out/tinyxml2_fuzz fuzz/corpus/tinyxml2_fuzz`

The harness feeds attacker-controlled XML bytes into `tinyxml2::XMLDocument::Parse`.
Default max_len: {max_len}
"""

    (fuzz_dir / "tinyxml2_fuzz.cc").write_text(fuzzer_cc, encoding="utf-8")
    (fuzz_dir / "build.py").write_text(build_py, encoding="utf-8")
    (fuzz_dir / "README.md").write_text(readme, encoding="utf-8")
    (fuzz_dir / "tinyxml2_fuzz.options").write_text(f"-max_len={max_len}\n", encoding="utf-8")


def _fallback_next(state: FuzzWorkflowRuntimeState) -> str:
    last_step = (state.get("last_step") or "").strip()
    last_error = (state.get("last_error") or "").strip()
    if bool(state.get("crash_found")):
        return "fix_crash"
    if last_step == "fix_crash":
        return "build" if not last_error else "stop"
    # Simple deterministic router.
    if last_error:
        if "No fuzzer binaries" in last_error:
            return "synthesize"
        if "build" in last_step:
            return "fix_build"
        if "OpenCode" in last_error or "plan" in last_error.lower():
            return "plan"
        if "build" in last_error.lower():
            return "fix_build"

    # Forward progress if possible.
    if last_step in {"", "init"}:
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
    model: Optional[str] = None


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
    codex_cli = (os.environ.get("SHERPA_CODEX_CLI") or os.environ.get("CODEX_CLI") or "opencode").strip()

    workdir = _alloc_output_workdir(repo_url)
    generator = NonOssFuzzHarnessGenerator(
        repo_spec=RepoSpec(url=repo_url, workdir=workdir),
        ai_key_path=ai_key_path,
        max_len=max_len,
        time_budget_per_target=time_budget,
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
            "build_attempts": int(state.get("build_attempts") or 0),
            "codex_hint": "",
            "failed": False,
            "repo_root": str(generator.repo_root),
            "last_crash_artifact": "",
            "last_fuzzer": "",
            "crash_fix_attempts": int(state.get("crash_fix_attempts") or 0),
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
    if not _has_codex_key():
        if _is_tinyxml2_repo(gen.repo_root):
            _write_builtin_tinyxml2_plan(gen.repo_root)
            out = {**state, "last_step": "plan", "last_error": "", "codex_hint": "", "message": "planned (builtin)"}
            _wf_log(cast(dict[str, Any], out), f"<- plan builtin ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out
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
            prompt = (
                "You are coordinating a fuzz harness generation workflow.\n"
                "Perform the planning step and produce fuzz/PLAN.md and fuzz/targets.json as required.\n\n"
                "IMPORTANT: Do NOT run any build, compile, or test commands. Only create/edit files.\n\n"
                "Additional instruction from coordinator:\n" + hint
            )
            gen.patcher.run_codex_command(prompt)
        else:
            gen._pass_plan_targets()
        out = {**state, "last_step": "plan", "last_error": "", "codex_hint": "", "message": "planned"}
        _wf_log(cast(dict[str, Any], out), f"<- plan ok dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        # Built-in fallback for known repos (keeps web flow usable without OpenCode).
        try:
            if _is_tinyxml2_repo(gen.repo_root):
                _write_builtin_tinyxml2_plan(gen.repo_root)
                out = {**state, "last_step": "plan", "last_error": "", "codex_hint": "", "message": "planned (builtin)"}
                _wf_log(cast(dict[str, Any], out), f"<- plan builtin ok dt={_fmt_dt(time.perf_counter()-t0)}")
                return out
        except Exception:
            pass
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
    if not _has_codex_key():
        if _is_tinyxml2_repo(gen.repo_root):
            max_len = int(state.get("max_len") or 1024)
            _write_builtin_tinyxml2_scaffold(gen.repo_root, max_len)
            out = {**state, "last_step": "synthesize", "last_error": "", "codex_hint": "", "message": "synthesized (builtin)"}
            _wf_log(cast(dict[str, Any], out), f"<- synthesize builtin ok dt={_fmt_dt(time.perf_counter()-t0)}")
            return out
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
            prompt = (
                "You are coordinating a fuzz harness generation workflow.\n"
                "Perform the synthesis step: create harness + fuzz/build.py + build glue under fuzz/.\n\n"
                "IMPORTANT: Do NOT run any build, compile, or test commands. Only create/edit files.\n\n"
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
        # Built-in fallback for known repos (keeps web flow usable without OpenCode).
        try:
            if _is_tinyxml2_repo(gen.repo_root):
                max_len = int(state.get("max_len") or 1024)
                _write_builtin_tinyxml2_scaffold(gen.repo_root, max_len)
                out = {**state, "last_step": "synthesize", "last_error": "", "codex_hint": "", "message": "synthesized (builtin)"}
                _wf_log(cast(dict[str, Any], out), f"<- synthesize builtin ok dt={_fmt_dt(time.perf_counter()-t0)}")
                return out
        except Exception:
            pass
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
    # Single build attempt (no OpenCode auto-fix here). If it fails, we'll route to fix_build.
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

        build_env = os.environ.copy()
        if getattr(gen, "docker_image", None):
            include_root = "/work"
        else:
            include_root = str(gen.repo_root)
        for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            prev = build_env.get(key, "").strip()
            build_env[key] = f"{include_root}:{prev}" if prev else include_root

        rc, out, err = gen._run_cmd(cmd, cwd=gen.repo_root, env=build_env, timeout=7200)
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
                "- IMPORTANT: Tell OpenCode to NOT run any commands — only edit files.\n"
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

    # Now call OpenCode with a purpose-built prompt including diagnostics.
    context_parts: list[str] = []
    if last_error:
        context_parts.append("=== last_error ===\n" + last_error)
    if stdout_tail:
        context_parts.append("=== build stdout (tail) ===\n" + stdout_tail)
    if stderr_tail:
        context_parts.append("=== build stderr (tail) ===\n" + stderr_tail)
    context = "\n\n".join(context_parts)

    prompt = (
        "You are OpenCode operating inside a Git repository.\n"
        "Task: fix the fuzz harness/build source code so the build will pass when run later.\n\n"
        "Goal (will be verified by a separate automated system — do NOT run these yourself):\n"
        "- `python fuzz/build.py` should complete successfully\n"
        "- fuzz/out/ should contain at least one runnable fuzzer binary\n\n"
        "CRITICAL: Do NOT run any commands (no cmake, make, python, bash, gcc, clang, etc.).\n"
        "Only edit source files. The build will be executed by the workflow after you finish.\n\n"
        "Constraints:\n"
        "- Keep changes minimal; avoid refactors\n"
        "- Prefer edits under fuzz/ and minimal build glue only\n\n"
        "Coordinator instruction:\n"
        + codex_hint.strip()
        + "\n\nWhen finished, write `fuzz/build.py` into `./done`."
    )

    try:
        _wf_log(cast(dict[str, Any], state), f"fix_build: running opencode (hint_lines={len(codex_hint.splitlines())})")
        gen.patcher.run_codex_command(prompt, additional_context=context or None)
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
                last_artifact = str(first)
                last_fuzzer = fuzzer_name
                break

        msg = "Fuzzing completed." if not crash_found else "Fuzzing completed (crash found and packaged)."
        out = {
            **state,
            "last_step": "run",
            "last_error": "",
            "crash_found": crash_found,
            "last_crash_artifact": last_artifact,
            "last_fuzzer": last_fuzzer,
            "message": msg,
        }
        _wf_log(cast(dict[str, Any], out), f"<- run ok crash_found={crash_found} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
    except Exception as e:
        out = {**state, "last_step": "run", "last_error": str(e), "message": "run failed"}
        _wf_log(cast(dict[str, Any], out), f"<- run err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out


def _node_fix_crash(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")

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
        prompt = (
            "You are OpenCode. The crash was diagnosed as a HARNESS ERROR.\n"
            "Task: fix the fuzz harness/build glue so the crash no longer happens for the same input.\n\n"
            "Constraints:\n"
            "- Only modify files under fuzz/ or minimal build glue required for the harness.\n"
            "- Do not change upstream/product code unless absolutely required.\n"
            "- Keep changes minimal and targeted.\n\n"
            "Goal (will be verified by a separate automated system — do NOT run these yourself):\n"
            "- The fuzzer should build successfully.\n"
            "- Running the fuzzer with the previous crashing input should no longer crash.\n\n"
            "CRITICAL: Do NOT run any commands. Only edit source files.\n\n"
            "When finished, write the key file you modified into ./done."
        )
    else:
        prompt = (
            "You are OpenCode. Fix the underlying bug in the target repository so the crash no longer occurs.\n\n"
            "Constraints:\n"
            "- Keep changes minimal and focused on correctness/security.\n"
            "- Do NOT disable the harness or skip input processing.\n"
            "- Avoid broad refactors.\n\n"
            "Goal (will be verified by a separate automated system — do NOT run these yourself):\n"
            "- The fuzzer should build successfully.\n"
            "- The previous crashing input should no longer crash.\n\n"
            "CRITICAL: Do NOT run any commands. Only edit source files.\n\n"
            "When finished, write the key file you modified into ./done."
        )

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
        gen.patcher.run_codex_command(prompt, additional_context=context or None)
        patch_path = repo_root / "fix.patch"
        fix_summary_path = repo_root / "fix_summary.md"
        changed_files = write_patch_from_snapshot(snapshot, repo_root, patch_path)
        patch_bytes = patch_path.stat().st_size if patch_path.exists() else 0

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

    # If we already finished a run:
    # - crash found => try to fix and continue
    # - no crash => stop
    if state.get("last_step") == "run":
        if bool(state.get("crash_found")):
            return {**state, "next": "fix_crash"}
        if not (state.get("last_error") or "").strip():
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
        "You decide the next step and optionally provide a short instruction to guide OpenCode for that step.\n\n"
        "Constraints:\n"
        "- Allowed next steps: plan, synthesize, build, fix_build, fix_crash, run, stop\n"
        "- Only provide codex_hint when next is plan, synthesize, fix_build, or fix_crash\n"
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
        if nxt not in {"plan", "synthesize", "fix_build", "fix_crash"}:
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
    graph.add_node("fix_crash", _node_fix_crash)
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
            "fix_crash": "fix_crash",
            "run": "run",
            "stop": END,
        },
    )

    graph.add_edge("plan", "decide")
    graph.add_edge("synthesize", "decide")
    graph.add_edge("build", "decide")
    graph.add_edge("fix_build", "build")
    graph.add_edge("fix_crash", "build")
    graph.add_edge("run", "decide")

    return graph


def _detect_harness_error(repo_root: Path) -> bool:
    analysis_path = repo_root / "crash_analysis.md"
    if not analysis_path.is_file():
        return False
    try:
        text = analysis_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return bool(re.search(r"HARNESS ERROR", text, re.IGNORECASE))


def _write_run_summary(out: dict[str, Any]) -> None:
    repo_root_raw = out.get("repo_root")
    if not repo_root_raw:
        return
    repo_root = Path(str(repo_root_raw))
    if not repo_root.exists():
        return

    crash_found = bool(out.get("crash_found"))
    last_error = str(out.get("last_error") or "").strip()
    failed = bool(out.get("failed"))
    status = "error" if (failed or last_error) else ("crash_found" if crash_found else "ok")
    harness_error = _detect_harness_error(repo_root)

    bundle_dirs = [
        d.name
        for d in repo_root.iterdir()
        if d.is_dir() and d.name.startswith(("challenge_bundle", "false_positive", "unreproducible"))
    ]

    data = {
        "repo_url": out.get("repo_url"),
        "repo_root": str(repo_root),
        "status": status,
        "message": out.get("message"),
        "last_step": out.get("last_step"),
        "step_count": out.get("step_count"),
        "build_attempts": out.get("build_attempts"),
        "build_rc": out.get("build_rc"),
        "last_error": last_error,
        "crash_found": crash_found,
        "last_fuzzer": out.get("last_fuzzer"),
        "last_crash_artifact": out.get("last_crash_artifact"),
        "harness_error": harness_error,
        "fix_patch_path": out.get("fix_patch_path") or "",
        "fix_patch_files": out.get("fix_patch_files") or [],
        "fix_patch_bytes": out.get("fix_patch_bytes") or 0,
        "crash_info_path": str(repo_root / "crash_info.md"),
        "crash_analysis_path": str(repo_root / "crash_analysis.md"),
        "reproducer_path": str(repo_root / "reproduce.py"),
        "bundles": bundle_dirs,
        "timestamp": time.time(),
    }

    summary_json = repo_root / "run_summary.json"
    summary_md = repo_root / "run_summary.md"
    try:
        summary_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    md_lines = [
        "# Run Summary",
        "",
        f"- Status: {status}",
        f"- Repo: {data['repo_url']}",
        f"- Repo root: {data['repo_root']}",
        f"- Last step: {data['last_step']}",
        f"- Build attempts: {data['build_attempts']}",
        f"- Crash found: {crash_found}",
        f"- Harness error: {harness_error}",
    ]
    if last_error:
        md_lines.extend(["", "## Last Error", "```text", last_error, "```"])
    if crash_found:
        md_lines.extend(
            [
                "",
                "## Crash",
                f"- Fuzzer: {data['last_fuzzer']}",
                f"- Artifact: {data['last_crash_artifact']}",
                f"- crash_info.md: {data['crash_info_path']}",
                f"- crash_analysis.md: {data['crash_analysis_path']}",
            ]
        )
    if data["fix_patch_path"]:
        md_lines.extend(
            [
                "",
                "## Fix Patch",
                f"- Patch: {data['fix_patch_path']}",
                f"- Files changed: {len(data['fix_patch_files'])}",
            ]
        )
        if data["fix_patch_files"]:
            md_lines.extend([f"- {p}" for p in data["fix_patch_files"]])
    if bundle_dirs:
        md_lines.extend(["", "## Bundles"] + [f"- {b}" for b in bundle_dirs])

    try:
        summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def run_fuzz_workflow(inp: FuzzWorkflowInput) -> str:
    _wf_log(None, f"workflow start repo={inp.repo_url} docker_image={inp.docker_image or '(host)'} time_budget={inp.time_budget}s")
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
            "max_len": inp.max_len,
            "docker_image": inp.docker_image,
            "ai_key_path": str(inp.ai_key_path),
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
