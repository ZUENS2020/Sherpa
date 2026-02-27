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

_DEFAULT_TIME_BUDGET_SEC = 900
_UNLIMITED_TIME_BUDGET_SENTINEL_SEC = 2_147_483_647


def parse_budget_value(raw: Any, *, default: int = _DEFAULT_TIME_BUDGET_SEC) -> int:
    """Parse a budget field while preserving explicit 0 (unlimited)."""
    if raw is None:
        return int(default)
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def is_unlimited_budget(raw: Any, *, default: int = _DEFAULT_TIME_BUDGET_SEC) -> bool:
    return parse_budget_value(raw, default=default) <= 0


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


def classify_build_failure(
    last_error: str,
    stdout_tail: str,
    stderr_tail: str,
    *,
    build_rc: int,
    has_fuzzer_binaries: bool,
) -> tuple[str, str]:
    combined = "\n".join(x for x in [last_error, stdout_tail, stderr_tail] if x).strip()
    low = combined.lower()

    infra_checks: list[tuple[str, list[str]]] = [
        (
            "docker_daemon_unavailable",
            [
                "cannot connect to the docker daemon",
                "is the docker daemon running",
                "lookup sherpa-docker",
                "permission denied while trying to connect to the docker daemon",
                "error during connect",
            ],
        ),
        (
            "buildkit_unavailable",
            [
                "buildx component is missing or broken",
                "buildkit is enabled but the buildx component is missing or broken",
                "docker buildx",
            ],
        ),
        (
            "registry_dns_resolution_failed",
            [
                "temporary failure in name resolution",
                "lookup registry-1.docker.io",
                "server misbehaving",
            ],
        ),
        (
            "registry_tls_handshake_timeout",
            [
                "tls handshake timeout",
                "tls: handshake failure",
                "x509: certificate",
            ],
        ),
        (
            "registry_or_network_unavailable",
            [
                "failed to resolve source metadata",
                "dial tcp",
                "proxyconnect tcp",
                "connection refused",
                "i/o timeout",
                "connection reset by peer",
                "context deadline exceeded",
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
                "time budget exceeded",
            ],
        ),
    ]
    for code, needles in infra_checks:
        if any(n in low for n in needles):
            return "infra", code

    if build_rc == 0 and not has_fuzzer_binaries:
        return "source", "no_fuzzer_binaries"

    if "missing fuzz/build.py" in low:
        return "source", "missing_build_script"
    if any(k in low for k in ["no such file", "cannot find", "not found"]):
        return "source", "missing_source_file"
    if any(k in low for k in ["undefined reference", "ld:", "linker", "collect2"]):
        return "source", "link_error"
    if any(k in low for k in ["error:", "fatal error:", "compilation terminated", "clang", "gcc"]):
        return "source", "compile_error"
    if any(k in low for k in ["traceback", "exception", "module not found", "syntaxerror"]):
        return "source", "script_error"

    return "unknown", "unknown_build_failure"


def build_failure_recovery_advice(error_kind: str, error_code: str) -> str:
    if error_kind != "infra":
        return ""

    common = (
        "You can tune/disable retries via SHERPA_WORKFLOW_BUILD_LOCAL_RETRIES "
        "and SHERPA_DOCKER_BUILD_RETRIES if runtime becomes too long."
    )
    recovery: dict[str, str] = {
        "docker_daemon_unavailable": (
            "Docker daemon appears unreachable. Verify docker service is running, "
            "current user has permission to access /var/run/docker.sock, and retry. "
            + common
        ),
        "buildkit_unavailable": (
            "BuildKit/buildx is unavailable. Install/repair docker buildx plugin or "
            "temporarily set DOCKER_BUILDKIT=0 for classic builder fallback. "
            + common
        ),
        "registry_dns_resolution_failed": (
            "Registry DNS resolution failed. Check host/container DNS settings, "
            "configure stable resolvers (for example 8.8.8.8/1.1.1.1), and retry. "
            + common
        ),
        "registry_tls_handshake_timeout": (
            "Registry TLS handshake timed out. Check outbound HTTPS path/proxy, "
            "verify system clock/cert trust chain, and retry with network backoff. "
            + common
        ),
        "registry_or_network_unavailable": (
            "Container registry/network is unstable. Check proxy/firewall rules and "
            "egress connectivity to registry endpoints before retrying. "
            + common
        ),
        "resource_exhausted": (
            "Build host resources are exhausted. Free disk/memory (e.g., docker system prune) "
            "and retry. "
            + common
        ),
        "build_command_timeout": (
            "Build command timed out. Increase time budget or reduce retry counts for this run. "
            + common
        ),
    }
    return recovery.get(error_code, f"Infrastructure build failure detected ({error_code}). {common}")


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
    time_budget = parse_budget_value(state.get("time_budget"), default=_DEFAULT_TIME_BUDGET_SEC)
    elapsed = time.time() - started_at
    if time_budget > 0 and elapsed >= time_budget:
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
    total_budget = parse_budget_value(state.get("time_budget"), default=_DEFAULT_TIME_BUDGET_SEC)
    if total_budget <= 0:
        return _UNLIMITED_TIME_BUDGET_SENTINEL_SEC
    elapsed = max(0.0, time.time() - started_at)
    remaining = int(total_budget - elapsed)
    if remaining <= 0:
        return 0
    return remaining


def time_budget_exceeded_state(state: dict[str, Any], *, step_name: str) -> dict[str, Any]:
    started_at = float(state.get("workflow_started_at") or time.time())
    time_budget = parse_budget_value(state.get("time_budget"), default=_DEFAULT_TIME_BUDGET_SEC)
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
        r"<!--\s*TEMPLATE:\s*([a-zA-Z0-9_]+)\s*-->\s*(.*?)\s*<!--\s*END TEMPLATE\s*-->",
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
