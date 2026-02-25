from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

from persistent_config import load_config


def wf_log(state: dict[str, Any] | None, msg: str) -> None:
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


def fmt_dt(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        val = json.loads(blob)
    except Exception:
        return None
    return val if isinstance(val, dict) else None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def validate_targets_json(repo_root: Path) -> tuple[bool, str]:
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


def summarize_build_error(last_error: str, stdout_tail: str, stderr_tail: str) -> dict[str, str]:
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


def collect_key_artifact_hashes(repo_root: Path) -> dict[str, str]:
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
            out[name] = sha256_text(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
    return out


def has_codex_key() -> bool:
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


def slug_from_repo_url(repo_url: str) -> str:
    base = repo_url.rstrip("/").split("/")[-1]
    if base.endswith(".git"):
        base = base[: -len(".git")]
    base = re.sub(r"[^a-zA-Z0-9._-]+", "-", base).strip("-")
    return base or "repo"


def alloc_output_workdir(repo_url: str) -> Path | None:
    out_root = os.environ.get("SHERPA_OUTPUT_DIR", "").strip()
    if not out_root:
        return None
    base = Path(out_root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    slug = slug_from_repo_url(repo_url)
    return base / f"{slug}-{uuid.uuid4().hex[:8]}"


def enter_step(state: dict[str, Any], step_name: str) -> tuple[dict[str, Any], bool]:
    started_at = float(state.get("workflow_started_at") or time.time())
    time_budget = max(1, int(state.get("time_budget") or 900))
    elapsed = time.time() - started_at
    if elapsed >= time_budget:
        out = {
            **state,
            "last_step": step_name,
            "failed": True,
            "last_error": f"time budget exceeded: elapsed={elapsed:.1f}s budget={time_budget}s",
            "message": "workflow stopped (time budget exceeded)",
        }
        wf_log(out, f"<- {step_name} stop=time_budget elapsed={elapsed:.1f}s budget={time_budget}s")
        return out, True

    step_count = int(state.get("step_count") or 0) + 1
    max_steps = int(state.get("max_steps") or 10)
    next_state = {**state, "step_count": step_count}
    if step_count >= max_steps:
        failed = bool(next_state.get("last_error")) and not bool(next_state.get("crash_found"))
        out = {
            **next_state,
            "last_step": step_name,
            "failed": failed,
            "message": "workflow stopped (max steps reached)",
        }
        wf_log(out, f"<- {step_name} stop=max_steps")
        return out, True
    return next_state, False


def remaining_time_budget_sec(state: dict[str, Any], *, min_timeout: int = 5) -> int:
    _ = min_timeout
    started_at = float(state.get("workflow_started_at") or time.time())
    total_budget = max(1, int(state.get("time_budget") or 900))
    elapsed = max(0.0, time.time() - started_at)
    remaining = int(total_budget - elapsed)
    if remaining <= 0:
        return 0
    return remaining


def time_budget_exceeded_state(state: dict[str, Any], *, step_name: str) -> dict[str, Any]:
    started_at = float(state.get("workflow_started_at") or time.time())
    time_budget = max(1, int(state.get("time_budget") or 900))
    elapsed = max(0.0, time.time() - started_at)
    out = {
        **state,
        "last_step": step_name,
        "failed": True,
        "last_error": f"time budget exceeded: elapsed={elapsed:.1f}s budget={time_budget}s",
        "message": "workflow stopped (time budget exceeded)",
    }
    wf_log(out, f"<- {step_name} stop=time_budget elapsed={elapsed:.1f}s budget={time_budget}s")
    return out


def make_plan_hint(repo_root: Path) -> str:
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


def derive_plan_policy(repo_root: Path) -> tuple[bool, int]:
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
def load_opencode_prompt_templates() -> dict[str, str]:
    if not _OPENCODE_PROMPT_FILE.is_file():
        raise RuntimeError(f"OpenCode prompt template file not found: {_OPENCODE_PROMPT_FILE}")
    text = _OPENCODE_PROMPT_FILE.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"<!--\\s*TEMPLATE:\\s*([a-zA-Z0-9_]+)\\s*-->\\s*\\n(.*?)\\n<!--\\s*END TEMPLATE\\s*-->",
        re.DOTALL,
    )
    templates: dict[str, str] = {}
    for name, body in pattern.findall(text):
        templates[name.strip().lower()] = body.strip()
    if not templates:
        raise RuntimeError(f"No templates found in {_OPENCODE_PROMPT_FILE}")
    return templates


def render_opencode_prompt(name: str, **kwargs: object) -> str:
    templates = load_opencode_prompt_templates()
    key = name.strip().lower()
    if key not in templates:
        raise RuntimeError(f"OpenCode prompt template '{name}' not found in {_OPENCODE_PROMPT_FILE}")
    out = templates[key]
    for k, v in kwargs.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out.strip() + "\n"
