from __future__ import annotations

import hashlib
import json
import os
import re
import time
import shutil
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from persistent_config import load_config

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
    run_rc: int
    crash_evidence: str
    run_error_kind: str
    run_details: list[dict[str, Any]]
    last_crash_artifact: str
    last_fuzzer: str
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
    print(f"{prefix} {msg}", flush=True)


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _validate_targets_json(repo_root: Path) -> tuple[bool, str]:
    targets = repo_root / "fuzz" / "targets.json"
    if not targets.is_file():
        return False, "missing fuzz/targets.json"
    try:
        data = json.loads(targets.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return False, f"invalid json in fuzz/targets.json: {e}"

    if not isinstance(data, list) or not data:
        return False, "targets.json must be a non-empty JSON array"

    allowed_lang = {"c-cpp", "cpp", "c", "c++", "java"}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"targets[{i}] must be an object"
        for key in ("name", "api", "lang"):
            val = item.get(key)
            if not isinstance(val, str) or not val.strip():
                return False, f"targets[{i}].{key} must be a non-empty string"
        lang = str(item.get("lang") or "").strip().lower()
        if lang not in allowed_lang:
            return False, f"targets[{i}].lang unsupported: {item.get('lang')}"
    return True, ""


def _summarize_build_error(last_error: str, stdout_tail: str, stderr_tail: str) -> dict[str, str]:
    combined = "\n".join(x for x in [last_error, stdout_tail, stderr_tail] if x).strip()
    low = combined.lower()
    error_type = "unknown"
    if any(k in low for k in ["missing fuzz/build.py", "no such file", "cannot find", "not found"]):
        error_type = "missing_file"
    elif any(k in low for k in ["undefined reference", "ld:", "linker", "collect2"]):
        error_type = "link_error"
    elif any(k in low for k in ["error:", "fatal error:", "compilation terminated", "clang", "gcc"]):
        error_type = "compile_error"
    elif any(k in low for k in ["traceback", "exception", "module not found", "syntaxerror"]):
        error_type = "script_error"

    evidence_lines = [ln.strip() for ln in combined.splitlines() if ln.strip()]
    evidence = "\n".join(evidence_lines[-12:])
    return {
        "error_type": error_type,
        "evidence": evidence,
    }


def _collect_key_artifact_hashes(repo_root: Path) -> dict[str, str]:
    pairs = [
        ("fuzz/targets.json", repo_root / "fuzz" / "targets.json"),
        ("fuzz/build.py", repo_root / "fuzz" / "build.py"),
        ("fuzz/PLAN.md", repo_root / "fuzz" / "PLAN.md"),
    ]
    out: dict[str, str] = {}
    for name, path in pairs:
        if not path.is_file():
            continue
        try:
            out[name] = _sha256_text(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
    return out


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

1. Build: `cd fuzz && python build.py`
2. Run: `fuzz/out/tinyxml2_fuzz fuzz/corpus/tinyxml2_fuzz`

The harness feeds attacker-controlled XML bytes into `tinyxml2::XMLDocument::Parse`.
Default max_len: {max_len}
"""

    (fuzz_dir / "tinyxml2_fuzz.cc").write_text(fuzzer_cc, encoding="utf-8")
    (fuzz_dir / "build.py").write_text(build_py, encoding="utf-8")
    (fuzz_dir / "README.md").write_text(readme, encoding="utf-8")
    (fuzz_dir / "tinyxml2_fuzz.options").write_text(f"-max_len={max_len}\n", encoding="utf-8")


def _enter_step(state: FuzzWorkflowRuntimeState, step_name: str) -> tuple[FuzzWorkflowRuntimeState, bool]:
    step_count = int(state.get("step_count") or 0) + 1
    max_steps = int(state.get("max_steps") or 10)
    next_state = cast(FuzzWorkflowRuntimeState, {**state, "step_count": step_count})
    if step_count >= max_steps:
        failed = bool(next_state.get("last_error")) and not bool(next_state.get("crash_found"))
        out = cast(
            FuzzWorkflowRuntimeState,
            {
                **next_state,
                "last_step": step_name,
                "failed": failed,
                "message": "workflow stopped (max steps reached)",
            },
        )
        _wf_log(cast(dict[str, Any], out), f"<- {step_name} stop=max_steps")
        return out, True
    return next_state, False


def _make_plan_hint(repo_root: Path) -> str:
    hints: list[str] = []
    plan_path = repo_root / "fuzz" / "PLAN.md"
    targets_path = repo_root / "fuzz" / "targets.json"

    if targets_path.is_file():
        try:
            raw = json.loads(targets_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(raw, list) and raw:
                names = [str(it.get("name") or "").strip() for it in raw if isinstance(it, dict)]
                names = [n for n in names if n]
                if names:
                    hints.append(f"Prioritize targets in fuzz/targets.json: {', '.join(names[:3])}.")
        except Exception:
            pass

    if plan_path.is_file():
        try:
            for line in plan_path.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if s.startswith(("Primary fuzzer:", "Target:")):
                    hints.append(s)
                if len(hints) >= 3:
                    break
        except Exception:
            pass

    hints.extend(
        [
            "Keep harness deterministic and only touch fuzz/ plus minimal build glue.",
            "Ensure fuzz/build.py leaves at least one runnable fuzzer under fuzz/out/.",
        ]
    )
    return "\n".join(hints[:6])


def _derive_plan_policy(repo_root: Path) -> tuple[bool, int]:
    """Derive stop/repair policy from fuzz/PLAN.md (with safe defaults).

    Supported PLAN.md hints (case-insensitive):
    - "Crash policy: report-only"  -> do not enter fix_crash
    - "Crash policy: fix"          -> enter fix_crash (default)
    - "Max fix rounds: <N>"        -> max fix_crash rounds before stop (default 1)
    """
    fix_on_crash = True
    max_fix_rounds = 1
    plan_path = repo_root / "fuzz" / "PLAN.md"
    if not plan_path.is_file():
        return fix_on_crash, max_fix_rounds

    try:
        text = plan_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return fix_on_crash, max_fix_rounds

    m_policy = re.search(r"crash\s*policy\s*:\s*([^\n\r]+)", text, re.IGNORECASE)
    if m_policy:
        val = m_policy.group(1).strip().lower()
        if "report" in val or "triage" in val:
            fix_on_crash = False
        elif "fix" in val:
            fix_on_crash = True

    m_rounds = re.search(r"max\s*fix\s*rounds\s*:\s*(\d+)", text, re.IGNORECASE)
    if m_rounds:
        try:
            max_fix_rounds = max(0, min(int(m_rounds.group(1)), 20))
        except Exception:
            pass

    return fix_on_crash, max_fix_rounds


_OPENCODE_PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "opencode_prompts.md"


@lru_cache(maxsize=1)
def _load_opencode_prompt_templates() -> dict[str, str]:
    if not _OPENCODE_PROMPT_FILE.is_file():
        raise RuntimeError(f"OpenCode prompt template file not found: {_OPENCODE_PROMPT_FILE}")
    text = _OPENCODE_PROMPT_FILE.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"<!--\s*TEMPLATE:\s*([a-zA-Z0-9_]+)\s*-->\s*\n(.*?)\n<!--\s*END TEMPLATE\s*-->",
        re.DOTALL,
    )
    templates: dict[str, str] = {}
    for name, body in pattern.findall(text):
        templates[name.strip().lower()] = body.strip()
    if not templates:
        raise RuntimeError(f"No templates found in {_OPENCODE_PROMPT_FILE}")
    return templates


def _render_opencode_prompt(name: str, **kwargs: object) -> str:
    templates = _load_opencode_prompt_templates()
    key = name.strip().lower()
    if key not in templates:
        raise RuntimeError(f"OpenCode prompt template '{name}' not found in {_OPENCODE_PROMPT_FILE}")
    out = templates[key]
    for k, v in kwargs.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out.strip() + "\n"


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
    docker_image = (state.get("docker_image") or "").strip()
    if not docker_image:
        raise ValueError("Docker execution is mandatory; docker_image is required")
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
            "run_rc": 0,
            "crash_evidence": "none",
            "run_error_kind": "",
            "last_crash_artifact": "",
            "last_fuzzer": "",
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
        if _is_tinyxml2_repo(gen.repo_root):
            _write_builtin_tinyxml2_plan(gen.repo_root)
            fix_on_crash, max_fix_rounds = _derive_plan_policy(gen.repo_root)
            out = {
                **state,
                "last_step": "plan",
                "last_error": "",
                "codex_hint": _make_plan_hint(gen.repo_root),
                "plan_fix_on_crash": fix_on_crash,
                "plan_max_fix_rounds": max_fix_rounds,
                "message": "planned (builtin)",
            }
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
            prompt = _render_opencode_prompt("plan_with_hint", hint=hint)
            gen.patcher.run_codex_command(prompt)
        else:
            gen._pass_plan_targets()

        strict_targets = (os.environ.get("SHERPA_PLAN_STRICT_TARGETS_SCHEMA", "1").strip().lower() in {"1", "true", "yes", "on"})
        ok_targets, targets_err = _validate_targets_json(gen.repo_root)
        if strict_targets and not ok_targets:
            _wf_log(cast(dict[str, Any], state), f"plan: targets.json schema invalid -> {targets_err}; retrying once")
            prompt = _render_opencode_prompt("plan_fix_targets_schema", schema_error=targets_err)
            gen.patcher.run_codex_command(prompt)
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
        # Built-in fallback for known repos (keeps web flow usable without OpenCode).
        try:
            if _is_tinyxml2_repo(gen.repo_root):
                _write_builtin_tinyxml2_plan(gen.repo_root)
                fix_on_crash, max_fix_rounds = _derive_plan_policy(gen.repo_root)
                out = {
                    **state,
                    "last_step": "plan",
                    "last_error": "",
                    "codex_hint": _make_plan_hint(gen.repo_root),
                    "plan_fix_on_crash": fix_on_crash,
                    "plan_max_fix_rounds": max_fix_rounds,
                    "message": "planned (builtin)",
                }
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
    state, stop_now = _enter_step(state, "synthesize")
    if stop_now:
        return state
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
    state, stop_now = _enter_step(state, "build")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), f"-> build attempt={(int(state.get('build_attempts') or 0)+1)}")
    try:
        fuzz_dir = gen.repo_root / "fuzz"
        build_py = fuzz_dir / "build.py"
        build_sh = fuzz_dir / "build.sh"

        def _tail(s: str, n: int = 120) -> str:
            lines = (s or "").replace("\r", "\n").splitlines()
            return "\n".join(lines[-n:]).strip()

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
            _wf_log(cast(dict[str, Any], state), f"build cmd attempt {attempt}/{max_local_attempts} -> {' '.join(build_cmd)}")
            rc, out, err = gen._run_cmd(list(build_cmd), cwd=build_cwd, env=build_env, timeout=7200)
            attempts_used += 1

            # Backward-compatibility shim: older generated scripts may hardcode "fuzz/..."
            # and therefore need repo-root cwd.
            if rc != 0 and fallback_cmd is not None and fallback_cwd is not None and _is_repo_root_cwd_issue(out, err):
                _wf_log(
                    cast(dict[str, Any], state),
                    f"build retry from repo-root cwd -> {' '.join(fallback_cmd)}",
                )
                rc, out, err = gen._run_cmd(list(fallback_cmd), cwd=fallback_cwd, env=build_env, timeout=7200)
                attempts_used += 1

            if rc != 0 and retry_with_clean and build_cmd_clean is not None:
                combined = (out or "") + "\n" + (err or "")
                if not re.search(r"unrecognized arguments: --clean", combined, re.IGNORECASE):
                    _wf_log(cast(dict[str, Any], state), "build failed; retrying once with --clean")
                    rc2, out2, err2 = gen._run_cmd(list(build_cmd_clean), cwd=build_cwd, env=build_env, timeout=7200)
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
            "last_step": "build",
        }

        if final_rc != 0:
            next_state["last_error"] = f"build failed rc={final_rc} after {attempts_used} command run(s)"
            next_state["message"] = "build failed"
            _wf_log(cast(dict[str, Any], next_state), f"<- build fail rc={final_rc} dt={_fmt_dt(time.perf_counter()-t0)}")
            return next_state

        if not final_bins:
            next_state["last_error"] = f"No fuzzer binaries found under fuzz/out/ after {attempts_used} command run(s)"
            next_state["message"] = "build produced no fuzzers"
            _wf_log(cast(dict[str, Any], next_state), f"<- build fail no-fuzzers dt={_fmt_dt(time.perf_counter()-t0)}")
            return next_state

        next_state["last_error"] = ""
        next_state["message"] = f"built ({len(final_bins)} fuzzers)"
        _wf_log(cast(dict[str, Any], next_state), f"<- build ok fuzzers={len(final_bins)} dt={_fmt_dt(time.perf_counter()-t0)}")
        return next_state
    except Exception as e:
        out = {**state, "last_step": "build", "last_error": str(e), "message": "build failed"}
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

    last_error = (state.get("last_error") or "").strip()
    stdout_tail = (state.get("build_stdout_tail") or "").strip()
    stderr_tail = (state.get("build_stderr_tail") or "").strip()
    repo_root = str(gen.repo_root)

    # Fast-path hotfix (minimal, no refactor): a common generated-script issue is
    # linking with `-lz` while the static library is only available by file path.
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
                "- IMPORTANT: Tell OpenCode to NOT run any commands — only edit files.\n"
                "- Acceptance: `(cd fuzz && python build.py)` succeeds and leaves at least one executable in fuzz/out/.\n\n"
                f"repo_root={repo_root}\n"
                + f"error_type={summary['error_type']}\n"
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
                "Fix the fuzz build so that running `(cd fuzz && python build.py)` succeeds and leaves at least one executable fuzzer under fuzz/out/.\n"
                "Only modify files under fuzz/ and the minimal build glue required.\n"
                "If the harness source is wrong or missing includes/links, fix it. If build.py uses wrong target names or paths, correct it.\n"
                "Do not refactor production code."
            )

    # Now call OpenCode with a purpose-built prompt including diagnostics.
    context_parts: list[str] = []
    context_parts.append("=== structured_error ===\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    if last_error:
        context_parts.append("=== last_error ===\n" + last_error)
    if stdout_tail:
        context_parts.append("=== build stdout (tail) ===\n" + stdout_tail)
    if stderr_tail:
        context_parts.append("=== build stderr (tail) ===\n" + stderr_tail)
    context = "\n\n".join(context_parts)

    prompt = _render_opencode_prompt("fix_build_execute", codex_hint=codex_hint.strip())

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
        for bin_path in bins:
            fuzzer_name = bin_path.name
            try:
                gen._pass_generate_seeds(fuzzer_name)
            except Exception as e:
                # Seed generation is best-effort; do not block fuzzing.
                print(f"[warn] seed generation skipped ({fuzzer_name}): {e}")

            run: FuzzerRunResult = gen._run_fuzzer(bin_path)
            run_rc = int(run.rc)
            crash_evidence = run.crash_evidence
            run_error_kind = run.run_error_kind
            run_details.append(
                {
                    "fuzzer": fuzzer_name,
                    "rc": int(run.rc),
                    "crash_found": bool(run.crash_found),
                    "crash_evidence": run.crash_evidence,
                    "run_error_kind": run.run_error_kind,
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
            if run.error:
                run_last_error = run.error
                break

            if run.crash_found and run.first_artifact:
                first = Path(run.first_artifact)
                gen._analyze_and_package(fuzzer_name, first)
                crash_found = True
                last_artifact = str(first)
                last_fuzzer = fuzzer_name
                break

        if run_last_error:
            msg = "Fuzzing run failed."
        else:
            msg = "Fuzzing completed." if not crash_found else "Fuzzing completed (crash found and packaged)."
        out = {
            **state,
            "last_step": "run",
            "last_error": run_last_error,
            "crash_found": crash_found,
            "run_rc": run_rc,
            "crash_evidence": crash_evidence,
            "run_error_kind": run_error_kind,
            "run_details": run_details,
            "last_crash_artifact": last_artifact,
            "last_fuzzer": last_fuzzer,
            "message": msg,
        }
        _wf_log(
            cast(dict[str, Any], out),
            f"<- run ok crash_found={crash_found} rc={run_rc} evidence={crash_evidence} dt={_fmt_dt(time.perf_counter()-t0)}",
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
    graph.add_edge("init", "plan")

    def _route_after_plan(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")) or (state.get("last_error") or "").strip():
            return "stop"
        return "synthesize"

    def _route_after_synthesize(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")) or (state.get("last_error") or "").strip():
            return "stop"
        return "build"

    def _route_after_build(state: FuzzWorkflowRuntimeState) -> str:
        if bool(state.get("failed")):
            return "stop"
        if (state.get("last_error") or "").strip():
            return "fix_build"
        return "run"

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

    graph.add_conditional_edges("plan", _route_after_plan, {"synthesize": "synthesize", "stop": END})
    graph.add_conditional_edges("synthesize", _route_after_synthesize, {"build": "build", "stop": END})
    graph.add_conditional_edges("build", _route_after_build, {"run": "run", "fix_build": "fix_build", "stop": END})
    graph.add_conditional_edges("fix_build", _route_after_fix_build, {"build": "build", "stop": END})
    graph.add_conditional_edges("run", _route_after_run, {"fix_crash": "fix_crash", "stop": END})
    graph.add_conditional_edges("fix_crash", _route_after_fix_crash, {"build": "build", "stop": END})

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


def _bytes_human(num_bytes: int) -> str:
    n = max(0, int(num_bytes))
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    val = float(n)
    while val >= 1024.0 and idx < len(units) - 1:
        val /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(val)}{units[idx]}"
    return f"{val:.1f}{units[idx]}"


def _tree_file_stats(root: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    if not root.is_dir():
        return files, total_bytes
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        files += 1
        try:
            total_bytes += int(p.stat().st_size)
        except Exception:
            pass
    return files, total_bytes


def _collect_fuzz_inventory(repo_root: Path) -> dict[str, Any]:
    fuzz_dir = repo_root / "fuzz"
    out_dir = fuzz_dir / "out"
    corpus_dir = fuzz_dir / "corpus"
    artifacts_dir = out_dir / "artifacts"

    binaries: list[str] = []
    options_files: list[str] = []
    if out_dir.is_dir():
        for p in sorted(out_dir.iterdir()):
            if not p.is_file():
                continue
            name = p.name
            if name.endswith(".options"):
                options_files.append(name)
            if os.access(p, os.X_OK) or p.suffix.lower() == ".exe":
                binaries.append(name)

    artifact_files: list[str] = []
    if artifacts_dir.is_dir():
        for p in sorted(artifacts_dir.rglob("*")):
            if p.is_file():
                artifact_files.append(str(p.relative_to(repo_root)))

    corpus_stats: dict[str, dict[str, Any]] = {}
    corpus_total_files = 0
    corpus_total_bytes = 0
    if corpus_dir.is_dir():
        for d in sorted(corpus_dir.iterdir()):
            if not d.is_dir():
                continue
            files, size_bytes = _tree_file_stats(d)
            corpus_total_files += files
            corpus_total_bytes += size_bytes
            corpus_stats[d.name] = {
                "files": files,
                "bytes": size_bytes,
                "human": _bytes_human(size_bytes),
            }

    return {
        "fuzz_dir": str(fuzz_dir),
        "fuzz_out_dir": str(out_dir),
        "fuzz_corpus_dir": str(corpus_dir),
        "fuzzer_binaries": binaries,
        "fuzzer_count": len(binaries),
        "options_files": options_files,
        "artifact_files": artifact_files,
        "artifact_count": len(artifact_files),
        "corpus_stats": corpus_stats,
        "corpus_total_files": corpus_total_files,
        "corpus_total_bytes": corpus_total_bytes,
        "corpus_total_human": _bytes_human(corpus_total_bytes),
    }


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
    run_details = cast(list[dict[str, Any]], out.get("run_details") or [])
    fuzz_inventory = _collect_fuzz_inventory(repo_root)
    key_artifact_hashes = _collect_key_artifact_hashes(repo_root)

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
        "run_rc": out.get("run_rc"),
        "last_error": last_error,
        "crash_found": crash_found,
        "crash_evidence": out.get("crash_evidence") or "none",
        "run_error_kind": out.get("run_error_kind") or "",
        "run_details": run_details,
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
        "fuzz_inventory": fuzz_inventory,
        "key_artifact_hashes": key_artifact_hashes,
        "plan_policy": {
            "fix_on_crash": bool(out.get("plan_fix_on_crash", True)),
            "max_fix_rounds": int(out.get("plan_max_fix_rounds") or 1),
        },
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
        f"- Run rc: {data['run_rc']}",
        f"- Crash evidence: {data['crash_evidence']}",
        f"- Crash found: {crash_found}",
        f"- Harness error: {harness_error}",
        f"- Fuzzer binaries: {fuzz_inventory['fuzzer_count']}",
        f"- Corpus files: {fuzz_inventory['corpus_total_files']}",
        f"- Corpus size: {fuzz_inventory['corpus_total_human']}",
        f"- Plan crash policy: {'fix' if data['plan_policy']['fix_on_crash'] else 'report-only'}",
        f"- Plan max fix rounds: {data['plan_policy']['max_fix_rounds']}",
        f"- Key artifact hashes: {len(key_artifact_hashes)}",
    ]
    if key_artifact_hashes:
        md_lines.extend(["", "## Key Artifact Hashes"])
        for path, digest in sorted(key_artifact_hashes.items()):
            md_lines.append(f"- {path}: `{digest}`")
    if run_details:
        md_lines.extend(["", "## Fuzzer Effectiveness"])
        for item in run_details:
            md_lines.append(
                "- {fuzzer}: rc={rc}, cov={cov}, ft={ft}, corpus={corp_files}/{corp_size}, rss={rss}MB".format(
                    fuzzer=item.get("fuzzer"),
                    rc=item.get("rc"),
                    cov=item.get("final_cov"),
                    ft=item.get("final_ft"),
                    corp_files=item.get("final_corpus_files"),
                    corp_size=_bytes_human(int(item.get("final_corpus_size_bytes") or 0)),
                    rss=item.get("final_rss_mb"),
                )
            )
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

    out_dir = Path(str(fuzz_inventory.get("fuzz_out_dir") or ""))
    if out_dir.is_dir():
        eff_json = out_dir / "fuzz_effectiveness.json"
        eff_md = out_dir / "fuzz_effectiveness.md"
        eff = {
            "status": status,
            "repo_url": data.get("repo_url"),
            "run_rc": data.get("run_rc"),
            "crash_found": crash_found,
            "crash_evidence": data.get("crash_evidence"),
            "run_details": run_details,
            "fuzz_inventory": fuzz_inventory,
            "timestamp": data.get("timestamp"),
        }
        try:
            eff_json.write_text(json.dumps(eff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        eff_lines = [
            "# Fuzz Effectiveness",
            "",
            f"- Status: {status}",
            f"- Crash found: {crash_found}",
            f"- Run rc: {data.get('run_rc')}",
            f"- Fuzzer binaries: {fuzz_inventory['fuzzer_count']}",
            f"- Corpus files: {fuzz_inventory['corpus_total_files']}",
            f"- Corpus size: {fuzz_inventory['corpus_total_human']}",
        ]
        if run_details:
            eff_lines.extend(["", "## Per Fuzzer"])
            for item in run_details:
                eff_lines.append(
                    "- {fuzzer}: rc={rc}, cov={cov}, ft={ft}, corpus={corp_files}/{corp_size}, exec/s={eps}, rss={rss}MB".format(
                        fuzzer=item.get("fuzzer"),
                        rc=item.get("rc"),
                        cov=item.get("final_cov"),
                        ft=item.get("final_ft"),
                        corp_files=item.get("final_corpus_files"),
                        corp_size=_bytes_human(int(item.get("final_corpus_size_bytes") or 0)),
                        eps=item.get("final_execs_per_sec"),
                        rss=item.get("final_rss_mb"),
                    )
                )
        try:
            eff_md.write_text("\n".join(eff_lines) + "\n", encoding="utf-8")
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
