#!/usr/bin/env python3

#────────────
#
# Copyright 2025 Artificial Intelligence Cyber Challenge
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of 
# this software and associated documentation files (the “Software”), to deal in the 
# Software without restriction, including without limitation the rights to use, 
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
# Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ────────────

"""harness_generator/src/codex_helper.py
──────────────────────────────────────

Wrapper around the OpenCode CLI.

This helper preserves the public API and the success contract used throughout
the codebase:

    - The agent must write a sentinel file `./done` when finished.
    - We only treat a run as successful if a `git diff HEAD` is produced.

Key implementation goals:
    - **Windows compatibility**: avoid `pty` and Unix-only signal handling.
    - Robust retry + timeout behavior.
    - Stream output live to stdout while capturing it.

The CLI used is the OpenCode binary `opencode` in non-interactive mode (`opencode run`).
"""

from __future__ import annotations

from loguru import logger

import logging
import json
import hashlib
import os
import re
import signal
import shlex
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Sequence

try:
    from git import Repo, exc as git_exc  # type: ignore
except Exception:  # pragma: no cover
    Repo = None  # type: ignore
    git_exc = None  # type: ignore

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)
_ENSURED_OPENCODE_IMAGES: set[str] = set()
_SENSITIVE_KEYS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "MINIMAX_API_KEY",
    "ANTHROPIC_API_KEY",
    "DATABASE_URL",
    "POSTGRES_PASSWORD",
)
_SENSITIVE_KV_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASS))\s*=\s*([^\s,;]+)"
)
_AUTH_BEARER_RE = re.compile(r"(?i)\b(Authorization\s*:\s*Bearer\s+)([^\s]+)")


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _redact_text(text: str, *, env: dict | None = None) -> str:
    if not text:
        return text
    out = text
    env_map = env or {}
    for key in _SENSITIVE_KEYS:
        val = str(env_map.get(key) or os.environ.get(key) or "").strip()
        if val:
            out = out.replace(val, "***")
    out = _SENSITIVE_KV_RE.sub(lambda m: f"{m.group(1)}=***", out)
    out = _AUTH_BEARER_RE.sub(lambda m: f"{m.group(1)}***", out)
    return out


def _redact_cmd_for_log(cmd: Sequence[str], *, env: dict | None = None) -> str:
    items: list[str] = []
    i = 0
    while i < len(cmd):
        tok = str(cmd[i])
        if tok == "-e" and i + 1 < len(cmd):
            kv = str(cmd[i + 1])
            if "=" in kv:
                k, _ = kv.split("=", 1)
                items.extend(["-e", f"{k}=***"])
            else:
                items.extend(["-e", "***"])
            i += 2
            continue
        items.append(tok)
        i += 1
    return _redact_text(shlex.join(items), env=env)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _opencode_context_max_lines() -> int:
    raw = (os.environ.get("SHERPA_OPENCODE_CONTEXT_MAX_LINES") or "50").strip()
    try:
        return max(10, min(int(raw), 500))
    except Exception:
        return 50


def _opencode_context_max_chars() -> int:
    raw = (os.environ.get("SHERPA_OPENCODE_CONTEXT_MAX_CHARS") or "16000").strip()
    try:
        return max(512, min(int(raw), 200_000))
    except Exception:
        return 16_000


def _opencode_policy_enabled() -> bool:
    return _bool_env("SHERPA_OPENCODE_POLICY_ENABLED", True)


def _opencode_default_policy_path() -> Path:
    return Path(__file__).resolve().parent / "langchain_agent" / "prompts" / "opencode_global_policy.md"


def _opencode_policy_path() -> Path:
    raw = (os.environ.get("SHERPA_OPENCODE_POLICY_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _opencode_default_policy_path()


def _load_opencode_policy_text() -> tuple[Path, str]:
    path = _opencode_policy_path()
    if not path.is_file():
        return path, ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return path, ""
    return path, text


def _opencode_stage_skills_enabled() -> bool:
    return _bool_env("SHERPA_OPENCODE_STAGE_SKILLS_ENABLED", True)


def _opencode_stage_skills_strict() -> bool:
    return _bool_env("SHERPA_OPENCODE_STAGE_SKILLS_STRICT", False)


def _opencode_default_stage_skills_path() -> Path:
    return Path(__file__).resolve().parent / "langchain_agent" / "opencode_skills"


def _opencode_stage_skills_path() -> Path:
    raw = (os.environ.get("SHERPA_OPENCODE_STAGE_SKILLS_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _opencode_default_stage_skills_path()


def _resolve_stage_session_group(stage_skill: str) -> str:
    stage = str(stage_skill or "").strip()
    if not stage:
        return "default"
    if stage in {
        "plan",
        "plan_fix_targets_schema",
        "synthesize",
        "synthesize_complete_scaffold",
        "plan_repair_build",
        "synthesize_repair_build",
        "plan_repair_crash",
        "synthesize_repair_crash",
    }:
        return "planning_synth"
    if stage == "fix_build":
        return "fix_build"
    if stage in {"fix_crash_harness_error", "fix_crash_upstream_bug"}:
        return "fix_crash"
    return stage


def _load_stage_skill_text(stage_skill: str) -> tuple[Path, str]:
    root = _opencode_stage_skills_path()
    path = root / stage_skill / "SKILL.md"
    if not path.is_file():
        return path, ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return path, ""
    return path, text


def _compact_text_for_opencode(text: str, *, max_lines: int, max_chars: int) -> str:
    src = str(text or "")
    if not src:
        return ""
    # Do not truncate task/context/policy/skill payloads. The model must read
    # full stage contracts and evidence files to avoid partial-guidance errors.
    return src.strip()


def _write_opencode_materialized_text(
    working_dir: Path,
    *,
    name: str,
    text: str,
    max_lines: int,
    max_chars: int,
) -> tuple[Path, str]:
    compact = _compact_text_for_opencode(text, max_lines=max_lines, max_chars=max_chars)
    out_dir = working_dir / ".git" / "sherpa-opencode"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(compact + ("\n" if compact else ""), encoding="utf-8")
    return path, compact


def _append_opencode_metadata(repo_root: Path, payload: dict) -> None:
    """Append runtime OpenCode metadata outside the git working tree.

    Writing this file inside the repo can pollute `git diff HEAD` detection and
    cause false-positive "edits produced" signals.
    """
    try:
        override = (os.environ.get("SHERPA_OPENCODE_METADATA_PATH") or "").strip()
        if override:
            path = Path(override).expanduser().resolve()
        else:
            sink_root = Path("/tmp/sherpa-opencode-metadata")
            sink_root.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(repo_root.resolve()))
            path = sink_root / f"{slug}.jsonl"

        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _extract_changed_paths_from_diff(diff_text: str, *, limit: int = 20) -> list[str]:
    text = str(diff_text or "")
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        m = re.match(r"^diff --git a/(.+?) b/(.+)$", line.strip())
        if m:
            p = str(m.group(2) or "").strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
                if len(out) >= limit:
                    return out
            continue
        m = re.match(r"^[ MADRCU?!]{1,2}\s+(.+)$", line.strip())
        if m:
            p = str(m.group(1) or "").strip()
            if p.startswith("->"):
                continue
            if p and p not in seen:
                seen.add(p)
                out.append(p)
                if len(out) >= limit:
                    return out
    return out


def _build_blocklist() -> list[str]:
    # Default: block build/test/fuzz/run commands but allow read-only tools.
    always_allow = {"grep", "egrep", "fgrep", "rg", "ripgrep"}
    default = [
        "make",
        "cmake",
        "ninja",
        "meson",
        "bazel",
        "gradle",
        "mvn",
        "mvnw",
        "go",
        "cargo",
        "dotnet",
        "msbuild",
        "gcc",
        "g++",
        "clang",
        "clang++",
        "cc",
        "c++",
        "javac",
        "java",
        "python",
        "python3",
        "pip",
        "pip3",
        "pytest",
        "tox",
        "npm",
        "yarn",
        "pnpm",
        "bun",
    ]
    extra = [
        c.strip()
        for c in os.environ.get("SHERPA_OPENCODE_BLOCKLIST", "").split(",")
        if c.strip()
    ]
    allow = {
        c.strip()
        for c in os.environ.get("SHERPA_OPENCODE_ALLOWLIST", "").split(",")
        if c.strip()
    }
    merged = []
    for c in default + extra:
        if c and c not in allow and c not in always_allow and c not in merged:
            merged.append(c)
    return merged


def _create_block_shims(commands: list[str]) -> str:
    shim_dir = tempfile.mkdtemp(prefix="opencode-block-")
    if os.name == "nt":
        for cmd in commands:
            path = Path(shim_dir) / f"{cmd}.cmd"
            path.write_text(
                "@echo off\r\n"
                "echo [sherpa] blocked command: %0\r\n"
                "exit /b 126\r\n",
                encoding="utf-8",
            )
    else:
        for cmd in commands:
            path = Path(shim_dir) / cmd
            path.write_text(
                "#!/usr/bin/env sh\n"
                "echo \"[sherpa] blocked command: $0\" >&2\n"
                "exit 126\n",
                encoding="utf-8",
            )
            try:
                path.chmod(0o755)
            except Exception:
                pass
    return shim_dir


def _apply_opencode_exec_policy(env: dict) -> None:
    if not _bool_env("SHERPA_OPENCODE_NO_EXEC", True):
        return
    commands = _build_blocklist()
    if not commands:
        return
    shim_dir = _create_block_shims(commands)
    env["PATH"] = shim_dir + os.pathsep + env.get("PATH", "")
    env["SHERPA_OPENCODE_BLOCKED_CMDS"] = ",".join(commands)
    env["SHERPA_OPENCODE_SHIM_DIR"] = shim_dir


def _docker_opencode_image() -> str:
    return os.environ.get("SHERPA_OPENCODE_DOCKER_IMAGE", "").strip()



def _opencode_auto_build_enabled() -> bool:
    return _bool_env("SHERPA_OPENCODE_AUTO_BUILD", True)


def _opencode_dockerfile_path() -> str:
    return os.environ.get("SHERPA_OPENCODE_DOCKERFILE", "/app/docker/Dockerfile.opencode").strip()


def _opencode_build_context() -> str:
    return os.environ.get("SHERPA_OPENCODE_BUILD_CONTEXT", "/app").strip()


def _opencode_build_args() -> list[str]:
    raw = os.environ.get("SHERPA_OPENCODE_BUILD_ARGS", "").strip()
    if not raw:
        return []
    out: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out += ["--build-arg", token]
    return out


def _opencode_base_image_candidates() -> list[str]:
    raw = os.environ.get("SHERPA_OPENCODE_BASE_IMAGES", "").strip()
    if raw:
        vals = [v.strip() for v in raw.split(",") if v.strip()]
        # Keep order while deduplicating.
        out: list[str] = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out
    return [
        "node:20-slim",
    ]


def _is_opencode_build_transient_error(output: str) -> bool:
    low = (output or "").lower()
    needles = [
        "tls handshake timeout",
        "failed to fetch anonymous token",
        "net/http: request canceled",
        "context deadline exceeded",
        "connection reset by peer",
        "i/o timeout",
        "unexpected eof",
        '": eof',
    ]
    return any(n in low for n in needles)


def _run_streaming_combined(
    cmd: Sequence[str],
    *,
    env: dict | None = None,
    cwd: Path | str | None = None,
    tail_lines: int = 160,
) -> tuple[int, str, str]:
    """Run command with stdout/stderr merged and stream all output.

    Returns: (returncode, output_for_scan, output_tail)
    """
    cmd_list = list(cmd)
    logger.info(f"[*] ➜  {_redact_cmd_for_log(cmd_list, env=env)}")
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        errors="replace",
    )
    assert proc.stdout is not None

    tail_buf: deque[str] = deque(maxlen=max(20, tail_lines))
    scan_buf: deque[str] = deque(maxlen=2400)

    try:
        for line in proc.stdout:
            safe_line = _redact_text(line, env=env)
            logger.info("{}", safe_line.rstrip("\n"))
            tail_buf.append(safe_line.rstrip("\n"))
            scan_buf.append(safe_line)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass

    rc = proc.wait()
    output_for_scan = "".join(scan_buf)
    output_tail = "\n".join(tail_buf)
    return rc, output_for_scan, output_tail


def _opencode_repo_slug(working_dir: Path) -> str:
    # Keep the path readable while ensuring uniqueness across concurrent jobs.
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", working_dir.name or "repo").strip("-") or "repo"
    digest = hashlib.sha1(str(working_dir.resolve()).encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{stem}-{digest}"


def _resolve_opencode_home_dir(shared_out: str, working_dir: Path | None = None) -> str:
    job_id = str(os.environ.get("SHERPA_JOB_ID") or "").strip()
    session_group = str(os.environ.get("SHERPA_OPENCODE_SESSION_GROUP") or "").strip()
    job_seg = re.sub(r"[^a-zA-Z0-9._-]+", "-", job_id).strip("-") if job_id else ""
    group_seg = re.sub(r"[^a-zA-Z0-9._-]+", "-", session_group).strip("-") if session_group else ""
    suffix = ""
    if job_seg:
        suffix += f"/{job_seg}"
    if shared_out and shared_out.strip():
        base = f"{shared_out.rstrip('/')}/.opencode-home"
        if working_dir is not None:
            path = f"{base}{suffix}/{_opencode_repo_slug(working_dir)}"
            if group_seg:
                path += f"/{group_seg}"
            return path
        return f"{base}{suffix}" if suffix else base
    if working_dir is not None:
        path = f"/tmp/.opencode-home{suffix}/{_opencode_repo_slug(working_dir)}"
        if group_seg:
            path += f"/{group_seg}"
        return path
    return "/tmp"


def _session_state_path(working_dir: Path) -> Path:
    return working_dir / ".git" / "sherpa-opencode" / "session_state.json"


def _load_session_state(working_dir: Path) -> dict:
    path = _session_state_path(working_dir)
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _save_session_state(working_dir: Path, state: dict) -> None:
    path = _session_state_path(working_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _opencode_provider_map_from_config(config_path: str) -> dict[str, set[str]]:
    if not config_path:
        return {}
    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    provider_node = payload.get("provider")
    if not isinstance(provider_node, dict):
        return {}

    out: dict[str, set[str]] = {}
    for raw_provider, raw_cfg in provider_node.items():
        provider = str(raw_provider or "").strip()
        if not provider:
            continue
        models: set[str] = set()
        if isinstance(raw_cfg, dict):
            raw_models = raw_cfg.get("models")
            if isinstance(raw_models, dict):
                for k in raw_models.keys():
                    mk = str(k or "").strip()
                    if mk:
                        models.add(mk)
            elif isinstance(raw_models, list):
                for item in raw_models:
                    mk = str(item or "").strip()
                    if mk:
                        models.add(mk)
        out[provider] = models
    return out


def _normalize_model_for_opencode(model: str, *, config_path: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        return ""
    # Already provider-qualified.
    if "/" in raw:
        return raw

    providers = _opencode_provider_map_from_config(config_path)

    # Match by configured provider model table first.
    matched_providers: list[str] = []
    for provider, configured_models in providers.items():
        for configured in configured_models:
            if configured == raw:
                matched_providers.append(provider)
                break
            if "/" in configured and configured.split("/", 1)[1] == raw:
                matched_providers.append(provider)
                break
    if len(matched_providers) == 1:
        return f"{matched_providers[0]}/{raw}"

    # If only one provider is configured, prefer it.
    if len(providers) == 1:
        only = next(iter(providers.keys()))
        return f"{only}/{raw}"

    # Heuristic for common GLM short model ids (hosted on jdcloud).
    if raw.lower().startswith("glm"):
        return f"jdcloud/{raw}"

    return raw


def _resolve_opencode_model(env: dict[str, str]) -> str | None:
    env_model = str(env.get("OPENCODE_MODEL", "") or "").strip()
    if env_model:
        cfg_path = str(env.get("OPENCODE_CONFIG", "") or "").strip()
        return _normalize_model_for_opencode(env_model, config_path=cfg_path)

    openrouter_model = str(env.get("OPENROUTER_MODEL", "") or "").strip()
    if openrouter_model:
        # OpenCode built-in models don't need provider prefix
        if openrouter_model.startswith("opencode/"):
            return openrouter_model
        # Already has openrouter prefix
        if openrouter_model.startswith("openrouter/"):
            return openrouter_model
        # Add openrouter prefix for other models
        return f"openrouter/{openrouter_model}"
    return None


def _ensure_opencode_image(image: str, env: dict) -> None:
    if not image or image in _ENSURED_OPENCODE_IMAGES:
        return
    if not _opencode_auto_build_enabled():
        return

    inspect_cmd = ["docker", "image", "inspect", image]
    try:
        probe = subprocess.run(
            inspect_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            _ENSURED_OPENCODE_IMAGES.add(image)
            return
    except FileNotFoundError as e:
        raise RuntimeError("Docker CLI not found; cannot build opencode image") from e

    dockerfile = _opencode_dockerfile_path()
    context_dir = _opencode_build_context()
    user_build_args = _opencode_build_args()
    max_retries_raw = os.environ.get("SHERPA_OPENCODE_BUILD_RETRIES", "3").strip()
    try:
        max_retries = max(1, min(int(max_retries_raw), 6))
    except Exception:
        max_retries = 3

    last_rc = 1
    last_tail = ""
    attempts: list[str] = []
    for base_image in _opencode_base_image_candidates():
        logger.info(f"[OpenCodeHelper] building opencode image with base={base_image}")
        for attempt in range(1, max_retries + 1):
            build_cmd = [
                "docker",
                "build",
                "-t",
                image,
                "-f",
                dockerfile,
                "--build-arg",
                f"OPENCODE_BASE_IMAGE={base_image}",
                *user_build_args,
                context_dir,
            ]
            rc, out_scan, tail = _run_streaming_combined(build_cmd, env=env)
            last_rc = int(rc) if rc is not None else 1
            last_tail = tail
            attempts.append(f"base={base_image} attempt={attempt} rc={last_rc}")
            if last_rc == 0:
                _ENSURED_OPENCODE_IMAGES.add(image)
                return
            if attempt < max_retries and _is_opencode_build_transient_error(out_scan):
                backoff_s = min(2 ** (attempt - 1), 10)
                logger.info(
                    f"[OpenCodeHelper] opencode image build transient error; "
                    f"retrying in {backoff_s}s (base={base_image}, attempt {attempt}/{max_retries})"
                )
                time.sleep(backoff_s)
                continue
            break

    attempts_summary = ", ".join(attempts[-8:])
    raise RuntimeError(
        f"Failed to build opencode image {image} (rc={last_rc}). Attempts: {attempts_summary}. Tail:\n{last_tail}"
    )



def _build_opencode_cmd(
    cli_exe: str,
    argv: list[str],
    working_dir: Path,
    env: dict,
) -> list[str]:
    return [cli_exe] + argv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_git_repo(path: Path) -> "Repo":
    """Return a *Repo* object, initialising a new repository if needed.

    Note: this helper is only used when GitPython is available and the caller
    is *not* using Dockerized git.
    """

    if Repo is None or git_exc is None:
        raise RuntimeError(
            "GitPython is not available. Either install GitPython + git, or run with Dockerized git enabled."
        )

    try:
        repo = Repo(path)
    except git_exc.InvalidGitRepositoryError:
        repo = Repo.init(path)

    # Make sure at least one commit exists so `git diff` behaves.
    if not repo.head.is_valid():
        repo.git.add(A=True)
        try:
            repo.git.commit(m="Initial commit", allow_empty=True)
        except git_exc.GitCommandError:
            # Happens when there is literally nothing to commit yet.
            pass
    return repo


# ---------------------------------------------------------------------------
# Core helper class
# ---------------------------------------------------------------------------


class CodexHelper:
    """Wrapper around OpenCode CLI with robust retry logic.

    Note: the class name is kept for backward compatibility with older imports.
    """

    def __init__(
        self,
        *,
        repo_path: Path,
        ai_key_path: str | None = None,
        copy_repo: bool = True,
        scratch_space: Path | None = None,
        codex_cli: str = "opencode",
        codex_model: str = "sonnet",
        approval_mode: str = "full-auto",
        dangerous_bypass: bool = False,
        sandbox_mode: str | None = None,
        git_docker_image: str | None = None,
    ) -> None:

        self.repo_path = Path(repo_path).expanduser().resolve()
        if not self.repo_path.is_dir():
            raise FileNotFoundError(f"Repository not found: {self.repo_path}")

        self.scratch_space = scratch_space or Path("/tmp")
        # Keep attribute name for compatibility with older config/env.
        self.codex_cli = str(codex_cli or "opencode")
        self.codex_model = codex_model
        self.approval_mode = approval_mode

        # Codex permissions: we run in non-interactive mode.
        # If dangerous_bypass is set, we expand sandbox permissions.
        self.dangerous_bypass = bool(dangerous_bypass)

        # Optional: override Codex sandbox mode.
        self.sandbox_mode = sandbox_mode

        # If set, all git operations (init/add/commit/diff) are executed inside
        # a Docker container using this image. This allows Windows hosts to run
        # without having git installed.
        self.git_docker_image = git_docker_image.strip() if isinstance(git_docker_image, str) and git_docker_image.strip() else None
        

        # Work on an isolated copy when requested so Codex can freely modify.
        if copy_repo:
            self.working_dir = Path(
                tempfile.mkdtemp(prefix="codex-helper-", dir=str(self.scratch_space))
            )
            shutil.copytree(self.repo_path, self.working_dir, dirs_exist_ok=True)
        else:
            self.working_dir = self.repo_path

        self.repo = None
        if self.git_docker_image:
            self._ensure_git_repo_docker()
        else:
            self.repo = _ensure_git_repo(self.working_dir)

        # Optional: allow teams to store an API key in a local file.
        # OpenCode CLI can authenticate via OPENAI_API_KEY (OpenAI-compatible).
        if ai_key_path:
            key_path = Path(ai_key_path).expanduser()
            if key_path.is_file():
                key = key_path.read_text(encoding="utf-8", errors="ignore").strip()
                if key:
                    # Prefer OPENAI_API_KEY to align with OpenCode/OpenAI-compatible tooling.
                    os.environ.setdefault("OPENAI_API_KEY", key)

        self.last_cli_error_kind = ""
        self.last_cli_error_message = ""

        LOGGER.debug("OpenCodeHelper working directory: %s", self.working_dir)

    def _docker_git(self, args: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
        if not self.git_docker_image:
            raise RuntimeError("Docker git is not configured")

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{str(self.working_dir.resolve())}:/repo",
            "-w",
            "/repo",
            self.git_docker_image,
            "git",
        ] + list(args)

        try:
            return subprocess.run(
                cmd,
                check=check,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                errors="replace",
            )
        except FileNotFoundError as e:
            raise RuntimeError("Docker not found in PATH. Install Docker Desktop and ensure 'docker' is available.") from e

    def _ensure_git_repo_docker(self) -> None:
        # Init repo if missing.
        if not (self.working_dir / ".git").exists():
            r = self._docker_git(["init"], check=False)
            if r.returncode != 0:
                raise RuntimeError(f"git init failed in docker: {r.stderr.strip()}")

        # Ensure user config exists for commits.
        self._docker_git(["config", "user.email", "sherpa@example.com"], check=False)
        self._docker_git(["config", "user.name", "sherpa"], check=False)

        # Ensure at least one commit exists so `git diff HEAD` behaves.
        head = self._docker_git(["rev-parse", "--verify", "HEAD"], check=False)
        if head.returncode != 0:
            self._docker_git(["add", "-A"], check=False)
            commit = self._docker_git(["commit", "--allow-empty", "-m", "Initial commit"], check=False)
            # Commit may still fail in edge cases; we tolerate as long as HEAD exists.
            head2 = self._docker_git(["rev-parse", "--verify", "HEAD"], check=False)
            if head2.returncode != 0:
                raise RuntimeError(
                    "Failed to create initial git commit inside docker. "
                    f"stderr={commit.stderr.strip() or head2.stderr.strip()}"
                )

    def _git_add_all(self) -> None:
        if self.git_docker_image:
            r = self._docker_git(["add", "-A"], check=False)
            if r.returncode != 0:
                raise RuntimeError(f"git add failed in docker: {r.stderr.strip()}")
            return

        assert self.repo is not None
        self.repo.git.add(A=True)

    def _git_diff_head(self) -> str:
        if self.git_docker_image:
            r = self._docker_git(["diff", "HEAD"], check=False)
            s = self._docker_git(["status", "--porcelain=v1", "--untracked-files=all"], check=False)
            if r.returncode != 0 or s.returncode != 0:
                # If HEAD is missing for any reason, attempt to repair once.
                self._ensure_git_repo_docker()
                r = self._docker_git(["diff", "HEAD"], check=False)
                s = self._docker_git(["status", "--porcelain=v1", "--untracked-files=all"], check=False)
            diff_text = (r.stdout or "").strip("\n")
            status_text = (s.stdout or "").strip("\n")
            if diff_text and status_text:
                return f"{diff_text}\n\n=== status ===\n{status_text}"
            return diff_text or status_text

        assert self.repo is not None
        diff_text = self.repo.git.diff("HEAD")
        status_text = self.repo.git.status("--porcelain=v1", "--untracked-files=all")
        diff_text = (diff_text or "").strip("\n")
        status_text = (status_text or "").strip("\n")
        if diff_text and status_text:
            return f"{diff_text}\n\n=== status ===\n{status_text}"
        return diff_text or status_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_codex_command(
        self,
        instructions: str | Sequence[str],
        *,
        additional_context: str | None = None,
        stage_skill: str | None = None,
        max_attempts: int = 3,
        timeout: int = 1800,
        max_cli_retries: int = 3,
        initial_backoff: float = 3.0,
        idle_timeout_override: int | None = None,
        activity_watch_paths: Sequence[str] | None = None,
        activity_probe_interval_sec: float | None = None,
    ) -> str | None:
        """Execute OpenCode with robust retry logic and return its stdout or *None*."""

        SENTINEL = "done"

        def _parse_idle_timeout(v: int | None) -> int:
            if v is not None:
                try:
                    return max(0, min(int(v), 86_400))
                except Exception:
                    return 300
            raw = (os.environ.get("SHERPA_OPENCODE_IDLE_TIMEOUT_SEC") or "300").strip()
            try:
                return max(0, min(int(raw), 86_400))
            except Exception:
                return 300

        def _parse_activity_probe_interval(v: float | None) -> float:
            if v is not None:
                try:
                    return max(0.2, min(float(v), 60.0))
                except Exception:
                    return 4.0
            raw = (os.environ.get("SHERPA_OPENCODE_ACTIVITY_PROBE_SEC") or "4").strip()
            try:
                return max(0.2, min(float(raw), 60.0))
            except Exception:
                return 4.0

        idle_timeout_sec = _parse_idle_timeout(idle_timeout_override)
        activity_probe_sec = _parse_activity_probe_interval(activity_probe_interval_sec)
        RETRY_ERRORS = (
            "Connection closed prematurely",
            "internal error",
            "failed to send request",
            "model failed to respond",
            "Network error",
            "ECONNRESET",
            "ETIMEDOUT",
            # Rate limiting / transient overload
            "Too Many Requests",
            "too many requests",
            "rate limit",
            "Rate limit",
            "HTTP 429",
            "429",
            "database is locked",
            # Common Chinese UI/messages when running on a zh-CN system
            "请求太频繁",
            "访问频繁",
            "请稍后再试",
            "Decode server is overloaded",
            "server is overloaded",
        )

        # Fatal errors that should not be retried – raise immediately with a
        # clear message so the caller gets an actionable diagnostic instead of
        # an obscure downstream JSON-parsing failure.
        FATAL_ERRORS = (
            "ProviderModelNotFoundError",
            "AuthenticationError",
            "InvalidAPIKeyError",
            "PermissionDeniedError",
        )

        done_path = self.working_dir / SENTINEL
        self.last_cli_error_kind = ""
        self.last_cli_error_message = ""
        watch_specs: List[str] = []
        for spec in (activity_watch_paths or ()):
            txt = str(spec).strip()
            if txt:
                watch_specs.append(txt)

        def _watch_targets(spec: str) -> List[Path]:
            p = Path(spec)
            if any(ch in spec for ch in "*?[]"):
                try:
                    return [x for x in self.working_dir.glob(spec)]
                except Exception:
                    return []
            if p.is_absolute():
                return [p]
            return [self.working_dir / p]

        def _to_rel(path: Path) -> str:
            try:
                return path.resolve().relative_to(self.working_dir.resolve()).as_posix()
            except Exception:
                return str(path)

        def _snapshot_path(path: Path) -> list[object]:
            try:
                if not path.exists():
                    return ["missing", _to_rel(path)]
                if path.is_file():
                    st = path.stat()
                    return ["file", _to_rel(path), int(st.st_mtime_ns), int(st.st_size)]
                if path.is_dir():
                    max_mtime = 0
                    file_count = 0
                    total_size = 0
                    for child in path.rglob("*"):
                        if not child.is_file():
                            continue
                        try:
                            st = child.stat()
                        except Exception:
                            continue
                        file_count += 1
                        total_size += int(st.st_size)
                        if int(st.st_mtime_ns) > max_mtime:
                            max_mtime = int(st.st_mtime_ns)
                    try:
                        dir_st = path.stat()
                        max_mtime = max(max_mtime, int(dir_st.st_mtime_ns))
                    except Exception:
                        pass
                    return ["dir", _to_rel(path), max_mtime, file_count, total_size]
                st = path.stat()
                return ["other", _to_rel(path), int(st.st_mtime_ns), int(st.st_size)]
            except Exception as e:
                return ["error", str(path), str(e)]

        def _activity_signature() -> str:
            if not watch_specs:
                return ""
            rows: List[list[object]] = []
            for spec in watch_specs:
                targets = _watch_targets(spec)
                if not targets and any(ch in spec for ch in "*?[]"):
                    rows.append(["glob-miss", spec])
                    continue
                for target in targets:
                    rows.append(_snapshot_path(target))
            rows.sort(key=lambda item: json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            return _sha256_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))

        # Build prompt body once.
        if isinstance(instructions, (list, tuple)):
            tasks = "\n".join(str(i) for i in instructions)
        else:
            tasks = str(instructions)
        max_ctx_lines = _opencode_context_max_lines()
        max_ctx_chars = _opencode_context_max_chars()
        task_path, compact_tasks = _write_opencode_materialized_text(
            self.working_dir,
            name="task.txt",
            text=tasks,
            max_lines=max_ctx_lines,
            max_chars=max_ctx_chars,
        )
        stage_skill_name = str(stage_skill or "").strip()
        session_group = _resolve_stage_session_group(stage_skill_name)
        session_state = _load_session_state(self.working_dir)
        session_groups = session_state.get("session_groups")
        if not isinstance(session_groups, dict):
            session_groups = {}
        group_state_raw = session_groups.get(session_group) if session_group else None
        group_state = group_state_raw if isinstance(group_state_raw, dict) else {}

        def _compose_session_memory_text() -> str:
            if not group_state:
                return ""
            attempts = group_state.get("recent_attempts")
            rows = attempts if isinstance(attempts, list) else []
            rows = [x for x in rows if isinstance(x, dict)][-3:]
            if not rows:
                return ""
            parts = ["SESSION MEMORY (recent attempts in this session group):"]
            for idx, row in enumerate(rows, start=1):
                status = str(row.get("status") or "").strip() or "unknown"
                changed = list(row.get("changed_paths") or [])
                changed_str = ", ".join(str(p) for p in changed[:6]) if changed else "-"
                err = str(row.get("error") or "").strip()
                if len(err) > 240:
                    err = err[-240:]
                parts.append(
                    f"{idx}. status={status}; changed_paths={changed_str}; "
                    f"error={err or '-'}"
                )
            return "\n".join(parts).strip()

        continue_session = bool(group_state.get("has_session")) if session_group else False
        if session_group and not continue_session:
            # Defensive fallback: if we already have attempt memory for this group,
            # continue the same conversation chain even when legacy flags are stale.
            recent_attempts = group_state.get("recent_attempts")
            if isinstance(recent_attempts, list) and recent_attempts:
                continue_session = True

        merged_context = str(additional_context or "").strip()
        session_memory_text = _compose_session_memory_text()
        if session_memory_text:
            merged_context = (
                (session_memory_text + "\n\n" + merged_context).strip()
                if merged_context
                else session_memory_text
            )

        context_path: Path | None = None
        compact_context = ""
        if merged_context:
            context_path, compact_context = _write_opencode_materialized_text(
                self.working_dir,
                name="additional_context.txt",
                text=merged_context,
                max_lines=max_ctx_lines,
                max_chars=max_ctx_chars,
            )
        stage_skill_source_path: Path | None = None
        stage_skill_materialized_path: Path | None = None
        compact_stage_skill = ""
        if stage_skill_name and _opencode_stage_skills_enabled():
            stage_skill_source_path, stage_skill_text = _load_stage_skill_text(stage_skill_name)
            if stage_skill_text:
                stage_skill_materialized_path, compact_stage_skill = _write_opencode_materialized_text(
                    self.working_dir,
                    name=f"stage_skill_{stage_skill_name}.md",
                    text=stage_skill_text,
                    max_lines=max_ctx_lines,
                    max_chars=max_ctx_chars,
                )
            else:
                msg = f"[OpenCodeHelper] stage skill missing: {stage_skill_name} ({stage_skill_source_path})"
                if _opencode_stage_skills_strict():
                    raise RuntimeError(msg)
                LOGGER.warning(msg)
        policy_source_path, policy_text = _load_opencode_policy_text() if _opencode_policy_enabled() else (Path(""), "")
        policy_materialized_path: Path | None = None
        compact_policy = ""
        if policy_text:
            policy_materialized_path, compact_policy = _write_opencode_materialized_text(
                self.working_dir,
                name="opencode_policy.md",
                text=policy_text,
                max_lines=max_ctx_lines,
                max_chars=max_ctx_chars,
            )
        elif _opencode_policy_enabled():
            LOGGER.warning("[OpenCodeHelper] global policy file missing: %s", policy_source_path)

        repo_root = str(self.working_dir.resolve())
        repo_path_hint = (
            f"The repository root is {repo_root}. "
            "Use relative paths from the current working directory and do not assume /repo exists."
        )

        prompt_parts: List[str] = [
            "You are OpenCode running in a local Git repository.",
            repo_path_hint,
            "Apply the edits requested below. Avoid refactors and unrelated changes.",
            "IMPORTANT ENV NOTE: The build/fuzz runtime environment is a separate container managed by the workflow, "
            "not this OpenCode execution environment. Do not infer runtime availability from this environment.",
            "Typical runtime images are sherpa-fuzz-cpp:latest or sherpa-fuzz-java:latest; "
            "OpenCode must only edit source files and must not attempt runtime verification.",
            "CRITICAL RULE: You MUST NOT execute build/test/fuzz commands or run binaries. "
            "Read-only commands (grep, egrep, fgrep, rg, ripgrep, ls, cat, find, sed) are allowed for inspection. "
            "Your ONLY job is to create and edit source files. "
            "Do NOT run cmake, make, gcc, clang, python, cargo, javac, mvn, gradle, npm, or similar build/run tools. "
            "The build and test steps are handled by a separate automated system. "
            "If you run build/test commands, the workflow will break.",
            "MANDATORY COMPLETION SIGNAL: You MUST create `./done` before exit. "
            "Without `./done`, this run is treated as failure and all edits are discarded.",
            "When ALL tasks are complete:",
            "  1) Print a short summary.",
            "  2) Create/overwrite a file called 'done' in the repo root (./done).",
            "     Follow the stage contract for done content exactly.",
            "     If the stage skill specifies a required sentinel path/value, that contract overrides generic defaults.",
            "MANDATORY: Read these files before any edit:",
            "  - ./.git/sherpa-opencode/task.txt",
        ]

        if context_path is not None:
            prompt_parts.append("  - ./.git/sherpa-opencode/additional_context.txt")
        if stage_skill_materialized_path is not None:
            prompt_parts.append(f"  - ./.git/sherpa-opencode/stage_skill_{stage_skill_name}.md")
            prompt_parts.append(
                f"STAGE SKILL ({stage_skill_name}): Follow this stage skill as the primary stage contract "
                "(goal, key files/templates, and acceptance criteria)."
            )
        if policy_materialized_path is not None:
            prompt_parts.append("  - ./.git/sherpa-opencode/opencode_policy.md")
            prompt_parts.append("GLOBAL POLICY: Use this as fallback policy when stage-specific instructions are absent.")

        prompt = "\n".join(prompt_parts).strip()
        prompt_hash = _sha256_text(prompt)
        context_hash = _sha256_text(compact_context)
        policy_hash = _sha256_text(compact_policy)
        stage_skill_hash = _sha256_text(compact_stage_skill)

        # ----------------------------------------------------------------
        # Outer loop – retry full patch attempt if no diff produced.
        # ----------------------------------------------------------------
        for attempt in range(1, max_attempts + 1):
            LOGGER.info("[OpenCodeHelper] patch attempt %d/%d", attempt, max_attempts)

            try:
                done_path.unlink(missing_ok=True)
            except Exception as e:
                LOGGER.warning("[OpenCodeHelper] failed to clear pre-attempt done flag: %s", e)

            # Baseline diff for this run: later passes may already have a diff
            # from earlier steps (e.g., Pass A creates fuzz/PLAN.md). We only
            # consider this run successful if the diff changes relative to this
            # baseline.
            try:
                baseline_diff = self._git_diff_head()
            except Exception:
                baseline_diff = ""
            baseline_activity_sig = _activity_signature()

            run_meta: dict = {
                "ts": time.time(),
                "attempt": attempt,
                "max_attempts": max_attempts,
                "codex_cli": self.codex_cli,
                "codex_model": self.codex_model,
                "resolved_model": "",
                "prompt_hash": prompt_hash,
                "context_hash": context_hash,
                "task_file": str(task_path),
                "context_file": str(context_path) if context_path else "",
                "stage_skill": stage_skill_name,
                "stage_skill_source_path": str(stage_skill_source_path) if stage_skill_source_path else "",
                "stage_skill_file": str(stage_skill_materialized_path) if stage_skill_materialized_path else "",
                "stage_skill_hash": stage_skill_hash if compact_stage_skill else "",
                "session_group": session_group,
                "session_continue": continue_session,
                "session_memory_injected": bool(session_memory_text),
                "policy_source_path": str(policy_source_path) if _opencode_policy_enabled() else "",
                "policy_file": str(policy_materialized_path) if policy_materialized_path else "",
                "policy_hash": policy_hash if compact_policy else "",
                "working_dir": str(self.working_dir),
                "status": "running",
                "repo_root": str(self.working_dir),
            }

            def _record_session_attempt(status: str, *, changed_paths: list[str] | None = None, error: str = "") -> None:
                if not session_group:
                    return
                rows = group_state.get("recent_attempts")
                attempts: list[dict[str, Any]] = rows if isinstance(rows, list) else []
                attempts = [x for x in attempts if isinstance(x, dict)]
                attempts.append(
                    {
                        "ts": int(time.time()),
                        "status": str(status or "").strip() or "unknown",
                        "changed_paths": list(changed_paths or [])[:20],
                        "error": str(error or "").strip()[:600],
                        "prompt_hash": prompt_hash,
                        "context_hash": context_hash,
                        "stage_skill": stage_skill_name,
                    }
                )
                attempts = attempts[-8:]
                group_state["recent_attempts"] = attempts
                group_state["last_status"] = str(status or "").strip() or "unknown"
                group_state["updated_at"] = int(time.time())
                group_state["has_session"] = True
                session_groups[session_group] = group_state
                session_state["session_groups"] = session_groups
                _save_session_state(self.working_dir, session_state)

            # ----------------------------------------------------------------
            # Inner loop – retry CLI invocation on transient errors.
            # ----------------------------------------------------------------

            cli_try = 0
            backoff = initial_backoff
            captured_chunks: List[str] = []

            while cli_try < max_cli_retries:
                cli_try += 1
                LOGGER.info("[OpenCodeHelper] launch #%d (backoff=%.1fs)", cli_try, backoff)

                # Resolve CLI path early so missing executables produce an actionable error.
                cli_exe = shutil.which(self.codex_cli)
                if cli_exe is not None and os.name == "nt":
                    # On Windows, npm sometimes provides both `opencode` and `opencode.cmd`.
                    # The extension-less file may not be directly executable via CreateProcess
                    # and can trigger: [WinError 193] %1 is not a valid Win32 application.
                    p = Path(cli_exe)
                    if p.suffix == "" and p.with_suffix(".cmd").is_file():
                        cli_exe = str(p.with_suffix(".cmd"))
                if cli_exe is None and os.name == "nt":
                    # Common location for npm global bin on Windows.
                    appdata = os.environ.get("APPDATA")
                    if appdata:
                        for candidate in (Path(appdata) / "npm" / "opencode.cmd", Path(appdata) / "npm" / "opencode"):
                            if candidate.is_file():
                                cli_exe = str(candidate)
                                break

                if cli_exe is None:
                    raise FileNotFoundError(
                        f"OpenCode CLI not found: '{self.codex_cli}'. "
                        "Ensure 'opencode' is installed and on PATH (e.g. npm global bin), "
                        "or pass the full path via --codex-cli."
                    )

                env = os.environ.copy()
                # Encourage non-interactive, tool-enabled mode.
                env.setdefault(
                    "OPENCODE_PERMISSION",
                    json.dumps(
                        {"permission": "allow", "external_directory": "allow"},
                        separators=(",", ":"),
                    ),
                )
                run_name = ""
                if session_group:
                    env["SHERPA_OPENCODE_SESSION_GROUP"] = session_group
                cmd: list[str] = ["run"]
                if continue_session:
                    cmd.append("--continue")
                model = _resolve_opencode_model(env)
                if model:
                    cmd += ["--model", model]
                run_meta["resolved_model"] = model or ""
                cmd.append(prompt)

                try:
                    _apply_opencode_exec_policy(env)
                    if session_group:
                        # Mark session as active as soon as we dispatch a run, even if this
                        # attempt later ends with no diff/no sentinel. This preserves dialogue
                        # continuity for subsequent retries.
                        _record_session_attempt("running")
                        continue_session = True
                    full_cmd = _build_opencode_cmd(cli_exe, cmd, self.working_dir, env)
                    proc = subprocess.Popen(
                        full_cmd,
                        cwd=self.working_dir,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=env,
                        text=True,
                        errors="replace",
                        start_new_session=(os.name != "nt"),
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"Failed to launch OpenCode CLI: {cli_exe} (cwd={self.working_dir}). "
                        "Make sure Docker is available (for containerized opencode) or "
                        "OpenCode is installed and accessible to the server process."
                    ) from e

                start_time = time.time()
                attempt_started_at = start_time
                saw_retry_error = False
                last_heartbeat = 0.0
                last_activity_ts = start_time
                last_progress_probe_ts = start_time
                last_seen_diff = baseline_diff
                last_seen_activity_sig = baseline_activity_sig
                cleanup_finalized = False

                def _cleanup_docker_run() -> None:
                    if not run_name:
                        return
                    try:
                        subprocess.run(
                            ["docker", "rm", "-f", run_name],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False,
                            text=True,
                            timeout=8,
                        )
                    except Exception:
                        pass

                def _wait_proc_with_timeout(timeout_sec: float) -> bool:
                    try:
                        proc.wait(timeout=timeout_sec)
                    except Exception:
                        return proc.poll() is not None
                    return True

                def _reap_process_group_children(pgid: int, budget_sec: float) -> tuple[int, str]:
                    if os.name == "nt" or pgid <= 0:
                        return 0, "ok"
                    deadline = time.monotonic() + max(0.0, float(budget_sec))
                    reaped = 0
                    status = "ok"
                    while time.monotonic() < deadline:
                        try:
                            pid, _ = os.waitpid(-pgid, os.WNOHANG)
                        except ChildProcessError:
                            break
                        except Exception:
                            status = "failed"
                            break
                        if pid == 0:
                            status = "partial"
                            time.sleep(0.05)
                            continue
                        reaped += 1
                    return reaped, status

                def _reap_any_dead_children(max_rounds: int = 8) -> tuple[int, str]:
                    if os.name == "nt":
                        return 0, "ok"
                    reaped = 0
                    status = "ok"
                    rounds = max(1, min(int(max_rounds), 128))
                    for _ in range(rounds):
                        try:
                            pid, _ = os.waitpid(-1, os.WNOHANG)
                        except ChildProcessError:
                            break
                        except Exception:
                            status = "failed"
                            break
                        if pid == 0:
                            break
                        reaped += 1
                    return reaped, status

                def _terminate_or_kill_proc(force: bool) -> None:
                    if proc.poll() is not None:
                        return
                    if os.name != "nt":
                        try:
                            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
                        except Exception:
                            pass
                    try:
                        if force:
                            proc.kill()
                        else:
                            proc.terminate()
                    except Exception:
                        pass

                def _finalize_proc_lifecycle(reason: str, *, force_kill: bool) -> None:
                    nonlocal cleanup_finalized
                    if cleanup_finalized:
                        return
                    cleanup_finalized = True
                    cleanup_status = "ok"
                    cleanup_error = ""
                    cleanup_reaped_count = 0
                    cleanup_reap_status = "ok"
                    proc_pgid = int(getattr(proc, "pid", 0) or 0)
                    try:
                        if proc.stdout is not None:
                            try:
                                proc.stdout.close()
                            except Exception:
                                pass
                        if proc.poll() is None:
                            _terminate_or_kill_proc(force=False)
                            if not _wait_proc_with_timeout(4.0):
                                _terminate_or_kill_proc(force=True if force_kill else False)
                                if not _wait_proc_with_timeout(4.0):
                                    cleanup_status = "failed"
                                    cleanup_error = "process did not exit after terminate/kill sequence"
                        reaped_pg, reap_status_pg = _reap_process_group_children(proc_pgid, 1.0)
                        cleanup_reaped_count += int(reaped_pg)
                        if reap_status_pg == "failed":
                            cleanup_reap_status = "failed"
                        elif reap_status_pg == "partial" and cleanup_reap_status == "ok":
                            cleanup_reap_status = "partial"
                        reaped_any, reap_status_any = _reap_any_dead_children(16)
                        cleanup_reaped_count += int(reaped_any)
                        if reap_status_any == "failed":
                            cleanup_reap_status = "failed"
                        elif reap_status_any == "partial" and cleanup_reap_status == "ok":
                            cleanup_reap_status = "partial"
                    except Exception as e:
                        cleanup_status = "failed"
                        cleanup_error = str(e)
                    finally:
                        _cleanup_docker_run()
                    run_meta["cleanup_status"] = cleanup_status
                    run_meta["cleanup_reason"] = reason
                    run_meta["cleanup_reaped_count"] = int(cleanup_reaped_count)
                    run_meta["cleanup_reap_status"] = cleanup_reap_status
                    if cleanup_error:
                        run_meta["cleanup_error"] = cleanup_error[:400]
                    if cleanup_reap_status == "failed":
                        run_meta["cleanup_reap_error"] = "failed to reap one or more child processes"
                    if cleanup_status != "ok":
                        LOGGER.warning(
                            "[OpenCodeHelper] process cleanup failed (reason=%s): %s",
                            reason,
                            cleanup_error or "unknown",
                        )
                        logger.info(
                            "[OpenCodeHelper] process cleanup failed "
                            f"(reason={reason}): {cleanup_error or 'unknown'}"
                        )

                def _kill_proc(reason: str = "forced_stop") -> None:
                    _finalize_proc_lifecycle(reason, force_kill=True)

                # Stream output while also watching for done sentinel.
                # NOTE: On Windows, `proc.stdout.readline()` can block forever when the child
                # produces no output. Use a reader thread + queue so the main loop can still
                # enforce timeouts and detect the `done` sentinel.
                assert proc.stdout is not None
                EOF = object()
                out_q: "queue.Queue[object]" = queue.Queue()

                def _stdout_reader() -> None:
                    try:
                        for line in proc.stdout:
                            out_q.put(line)
                    except Exception as e:
                        out_q.put(f"[CodexHelper] (stdout reader) {e}\n")
                    finally:
                        out_q.put(EOF)

                t = threading.Thread(target=_stdout_reader, daemon=True)
                t.start()

                try:
                    while True:
                        now = time.time()
                        elapsed = now - start_time

                        if elapsed > timeout:
                            LOGGER.warning("[CodexHelper] hard timeout; killing opencode")
                            saw_retry_error = True
                            logger.info(f"[OpenCodeHelper] hard timeout after {elapsed:.0f}s; terminating agent")
                            _kill_proc("hard_timeout")
                            break

                        if idle_timeout_sec > 0 and (now - last_progress_probe_ts) >= activity_probe_sec:
                            last_progress_probe_ts = now
                            progress_changed = False
                            try:
                                probed_diff = self._git_diff_head()
                                if probed_diff != last_seen_diff:
                                    last_seen_diff = probed_diff
                                    progress_changed = True
                            except Exception:
                                pass
                            if watch_specs:
                                try:
                                    sig_now = _activity_signature()
                                    if sig_now != last_seen_activity_sig:
                                        last_seen_activity_sig = sig_now
                                        progress_changed = True
                                except Exception:
                                    pass
                            if progress_changed:
                                last_activity_ts = now

                        if idle_timeout_sec > 0:
                            idle_for = now - last_activity_ts
                            if idle_for > idle_timeout_sec:
                                LOGGER.warning(
                                    "[CodexHelper] idle timeout; killing opencode (idle=%.0fs)",
                                    idle_for,
                                )
                                saw_retry_error = True
                                logger.info(
                                    "[OpenCodeHelper] idle timeout after "
                                    f"{idle_for:.0f}s without activity; terminating agent"
                                )
                                _kill_proc("idle_timeout")
                                break

                        # Heartbeat so job logs keep moving even if the agent is quiet.
                        if (now - last_heartbeat) > 10.0:
                            last_heartbeat = now
                            logger.info(f"[OpenCodeHelper] running… elapsed={elapsed:.0f}s")

                        if done_path.exists():
                            stale_done = False
                            done_mtime = 0.0
                            try:
                                done_mtime = float(done_path.stat().st_mtime)
                                stale_done = done_mtime < (attempt_started_at - 1e-3)
                            except Exception:
                                stale_done = False
                            if stale_done:
                                LOGGER.warning(
                                    "[OpenCodeHelper] stale done flag detected (mtime=%.3f < attempt_start=%.3f); removing and continuing",
                                    done_mtime,
                                    attempt_started_at,
                                )
                                logger.info("[OpenCodeHelper] stale done flag detected; removing and continuing")
                                try:
                                    done_path.unlink(missing_ok=True)
                                except Exception as e:
                                    raise RuntimeError(
                                        f"stale done flag could not be removed: {done_path} ({e})"
                                    ) from e
                            else:
                                LOGGER.info("[OpenCodeHelper] done flag detected")
                                logger.info("[OpenCodeHelper] done flag detected; terminating")
                                _kill_proc("done_flag")
                                break

                        # Try to get output without blocking.
                        try:
                            item = out_q.get(timeout=0.2)
                        except queue.Empty:
                            item = None

                        if item is EOF:
                            break
                        if isinstance(item, str) and item:
                            logger.info("{}", item.rstrip("\n"))
                            captured_chunks.append(item)
                            last_activity_ts = now
                            if any(err in item for err in RETRY_ERRORS) and not _bool_env("SHERPA_OPENCODE_IGNORE_RETRY_ERRORS", False):
                                lowered = item.lower()
                                if "overloaded" in lowered:
                                    self.last_cli_error_kind = "provider_overloaded"
                                    self.last_cli_error_message = item.strip()[:500]
                                elif not self.last_cli_error_kind:
                                    self.last_cli_error_kind = "provider_retryable_error"
                                    self.last_cli_error_message = item.strip()[:500]
                                LOGGER.warning("[OpenCodeHelper] retryable error detected → abort")
                                saw_retry_error = True
                                _kill_proc("retryable_error")
                                break
                            # Detect fatal (non-retryable) errors and surface them
                            # immediately so the caller gets a clear diagnostic.
                            for fatal_err in FATAL_ERRORS:
                                if fatal_err in item:
                                    detail = item.strip()[:500]
                                    self.last_cli_error_kind = "provider_fatal_error"
                                    self.last_cli_error_message = detail
                                    LOGGER.error("[OpenCodeHelper] fatal CLI error: %s", detail)
                                    _kill_proc("fatal_error")
                                    raise RuntimeError(
                                        f"OpenCode CLI fatal error: {fatal_err} — {detail}"
                                    )

                        # If process exited and queue is drained, we can stop.
                        if proc.poll() is not None and out_q.empty():
                            break
                finally:
                    # Drain any remaining buffered output.
                    try:
                        while True:
                            item2 = out_q.get_nowait()
                            if item2 is EOF:
                                break
                            if isinstance(item2, str) and item2:
                                logger.info("{}", item2.rstrip("\n"))
                                captured_chunks.append(item2)
                    except Exception:
                        pass
                    try:
                        t.join(timeout=1.0)
                    except Exception:
                        pass
                    _finalize_proc_lifecycle("loop_exit", force_kill=False)

                if str(run_meta.get("cleanup_status") or "") != "ok" or str(run_meta.get("cleanup_reap_status") or "") == "failed":
                    saw_retry_error = True

                if saw_retry_error:
                    if not self.last_cli_error_kind:
                        self.last_cli_error_kind = "provider_retryable_error"
                        self.last_cli_error_message = "retryable OpenCode CLI failure"
                    _record_session_attempt("retryable_error")
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                break

            # After inner loop – did Codex create the sentinel and produce diff?

            diff_now = ""
            try:
                diff_now = self._git_diff_head()
            except Exception:
                diff_now = ""

            diff_changed = bool(diff_now) and diff_now != baseline_diff

            if not done_path.exists():
                if not self.last_cli_error_kind:
                    self.last_cli_error_kind = "missing_sentinel"
                    self.last_cli_error_message = "OpenCode did not create done sentinel"
                LOGGER.warning("[OpenCodeHelper] sentinel not created; next attempt")
                logger.info("[OpenCodeHelper] sentinel not created; next attempt")
                run_meta["status"] = "retry_no_sentinel"
                run_meta["cli_retries_used"] = cli_try
                _append_opencode_metadata(self.working_dir, run_meta)
                _record_session_attempt(
                    "retry_no_sentinel",
                    changed_paths=_extract_changed_paths_from_diff(diff_now, limit=12),
                )
                continue  # outer attempt loop

            # Refresh repo to ensure it sees new changes.
            self._git_add_all()

            if diff_changed or self._git_diff_head() != baseline_diff:
                LOGGER.info("[OpenCodeHelper] diff produced — success")
                _record_session_attempt(
                    "success",
                    changed_paths=_extract_changed_paths_from_diff(diff_now, limit=20),
                )
                run_meta["status"] = "success"
                run_meta["cli_retries_used"] = cli_try
                _append_opencode_metadata(self.working_dir, run_meta)
                return "".join(captured_chunks)

            LOGGER.info("[OpenCodeHelper] sentinel present but no diff; next attempt")
            logger.info("[OpenCodeHelper] sentinel present but no diff; next attempt")
            run_meta["status"] = "retry_no_diff"
            run_meta["cli_retries_used"] = cli_try
            _append_opencode_metadata(self.working_dir, run_meta)
            _record_session_attempt("retry_no_diff")

        LOGGER.warning("[OpenCodeHelper] exhausted attempts — no edits produced")
        if not self.last_cli_error_kind:
            self.last_cli_error_kind = "exhausted_no_edits"
            self.last_cli_error_message = "OpenCode exhausted attempts without producing edits"
        _record_session_attempt("exhausted")
        _append_opencode_metadata(
            self.working_dir,
            {
                "ts": time.time(),
                "status": "exhausted",
                "codex_cli": self.codex_cli,
                "codex_model": self.codex_model,
                "prompt_hash": prompt_hash,
                "context_hash": context_hash,
                "working_dir": str(self.working_dir),
                "repo_root": str(self.working_dir),
            },
        )
        return None
