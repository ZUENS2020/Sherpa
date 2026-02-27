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

import logging
import json
import hashlib
import os
import re
import queue
import shutil
import subprocess
import tempfile
import textwrap
import threading
import time
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


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


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


def _build_blocklist() -> list[str]:
    # Default: block build/test/fuzz/run commands but allow read-only tools.
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
        if c and c not in allow and c not in merged:
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
        "m.daocloud.io/docker.io/library/node:20-slim",
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


def _gitnexus_auto_analyze_enabled() -> bool:
    return _bool_env("SHERPA_GITNEXUS_AUTO_ANALYZE", True)


def _gitnexus_skip_embeddings() -> bool:
    return _bool_env("SHERPA_GITNEXUS_SKIP_EMBEDDINGS", True)


def _opencode_repo_slug(working_dir: Path) -> str:
    # Keep the path readable while ensuring uniqueness across concurrent jobs.
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", working_dir.name or "repo").strip("-") or "repo"
    digest = hashlib.sha1(str(working_dir.resolve()).encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{stem}-{digest}"


def _resolve_opencode_home_dir(shared_out: str, working_dir: Path | None = None) -> str:
    if shared_out and shared_out.strip():
        base = f"{shared_out.rstrip('/')}/.opencode-home"
        if working_dir is not None:
            return f"{base}/{_opencode_repo_slug(working_dir)}"
        return base
    if working_dir is not None:
        return f"/tmp/.opencode-home/{_opencode_repo_slug(working_dir)}"
    return "/tmp"


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

    # Heuristic for common GLM short model ids.
    if raw.lower().startswith("glm-"):
        return f"zai/{raw}"

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
        print(f"[OpenCodeHelper] building opencode image with base={base_image}")
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
            proc = subprocess.run(
                build_cmd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            last_rc = int(proc.returncode) if proc.returncode is not None else 1
            out = proc.stdout or ""
            tail = "\n".join(out.splitlines()[-120:])
            last_tail = tail
            attempts.append(f"base={base_image} attempt={attempt} rc={last_rc}")
            if last_rc == 0:
                _ENSURED_OPENCODE_IMAGES.add(image)
                return
            if attempt < max_retries and _is_opencode_build_transient_error(out):
                backoff_s = min(2 ** (attempt - 1), 10)
                print(
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


def _docker_opencode_env_args(env: dict) -> list[str]:
    allowed_keys = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_MODEL",
        "OPENCODE_MODEL",
        "OPENCODE_PERMISSION",
        "OPENCODE_CONFIG",
        "SHERPA_OPENCODE_NO_EXEC",
        "SHERPA_OPENCODE_BLOCKLIST",
        "SHERPA_OPENCODE_ALLOWLIST",
    ]
    allowed_keys.append("DEEPSEEK_API_KEY")
    args: list[str] = []
    for k in allowed_keys:
        v = env.get(k)
        if v is not None and str(v).strip() != "":
            args += ["-e", f"{k}={v}"]
    return args


def _build_opencode_cmd(
    cli_exe: str,
    argv: list[str],
    working_dir: Path,
    env: dict,
) -> list[str]:
    image = _docker_opencode_image()
    if not image:
        return [cli_exe] + argv

    # Run opencode inside a dedicated container.
    shim_dir = env.get("SHERPA_OPENCODE_SHIM_DIR", "")
    shared_out = env.get("SHERPA_OUTPUT_DIR", "").strip()
    home_dir = _resolve_opencode_home_dir(shared_out, working_dir=working_dir)
    try:
        Path(home_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    path_in_container = "/opencode_shims:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    config_path = env.get("OPENCODE_CONFIG", "").strip()
    config_mount = config_path or "/opencode.json"

    docker_args = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{str(working_dir.resolve())}:/repo",
        "-w",
        "/repo",
        *_docker_opencode_env_args(env),
        "-e",
        f"PATH={path_in_container}",
        "-e",
        f"HOME={home_dir}",
        *(
            ["-v", f"{shim_dir}:/opencode_shims:ro"]
            if shim_dir
            else []
        ),
    ]
    run_name = str(env.get("SHERPA_OPENCODE_RUN_NAME", "") or "").strip()
    if run_name:
        docker_args += ["--name", run_name]
    if config_mount:
        docker_args += ["-v", f"{config_mount}:{config_mount}:ro"]
    if shared_out:
        docker_args += ["-v", f"{shared_out}:{shared_out}"]
    docker_args += [
        image,
        "opencode",
    ]
    return docker_args + argv


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

    def _maybe_prepare_gitnexus_context(self) -> None:
        """Best-effort: build GitNexus index snapshot for OpenCode MCP usage.

        To avoid polluting the mutable working repository (which affects diff-based
        success detection), we copy the current working tree into a persistent
        snapshot under SHERPA_OUTPUT_DIR and analyze that snapshot instead.
        """
        if not _gitnexus_auto_analyze_enabled():
            return

        image = _docker_opencode_image()
        if not image:
            return

        env = os.environ.copy()
        _ensure_opencode_image(image, env)

        shared_out = env.get("SHERPA_OUTPUT_DIR", "").strip()
        if not shared_out:
            LOGGER.warning(
                "[OpenCodeHelper] GitNexus auto analyze skipped: SHERPA_OUTPUT_DIR is empty"
            )
            return

        snapshot_root_raw = (
            env.get("SHERPA_GITNEXUS_SNAPSHOT_ROOT", "").strip()
            or f"{shared_out.rstrip('/')}/.gitnexus-snapshots"
        )
        snapshot_root = Path(snapshot_root_raw)
        snapshot_root.mkdir(parents=True, exist_ok=True)

        repo_key = hashlib.sha1(
            str(self.working_dir.resolve()).encode("utf-8", errors="replace")
        ).hexdigest()[:16]
        snapshot_dir = snapshot_root / f"repo-{repo_key}"

        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        shutil.copytree(
            self.working_dir,
            snapshot_dir,
            symlinks=True,
            ignore=shutil.ignore_patterns(".gitnexus"),
        )

        # Preserve uncommitted state in snapshot by creating a local commit.
        subprocess.run(
            ["git", "-C", str(snapshot_dir), "config", "user.email", "sherpa@example.com"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(snapshot_dir), "config", "user.name", "sherpa"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(snapshot_dir), "add", "-A"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(snapshot_dir), "commit", "--allow-empty", "-m", "sherpa gitnexus snapshot"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )

        home_dir = _resolve_opencode_home_dir(shared_out)
        clean_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{shared_out}:{shared_out}",
            "-e",
            f"HOME={home_dir}",
            image,
            "gitnexus",
            "clean",
            "--all",
            "--force",
        ]
        subprocess.run(
            clean_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
            env=env,
        )

        analyze_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{shared_out}:{shared_out}",
            "-w",
            str(snapshot_dir),
            "-e",
            f"HOME={home_dir}",
            image,
            "gitnexus",
            "analyze",
            str(snapshot_dir),
        ]
        if not _gitnexus_skip_embeddings():
            analyze_cmd.append("--embeddings")

        proc = subprocess.run(
            analyze_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout or "").splitlines()[-80:])
            LOGGER.warning(
                "[OpenCodeHelper] GitNexus analyze failed (non-fatal, rc=%s). Tail:\n%s",
                proc.returncode,
                tail,
            )
        else:
            LOGGER.info("[OpenCodeHelper] GitNexus snapshot analyzed: %s", snapshot_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_codex_command(
        self,
        instructions: str | Sequence[str],
        *,
        additional_context: str | None = None,
        max_attempts: int = 3,
        timeout: int = 1800,
        max_cli_retries: int = 3,
        initial_backoff: float = 3.0,
    ) -> str | None:
        """Execute OpenCode with robust retry logic and return its stdout or *None*."""

        SENTINEL = "done"
        idle_timeout_raw = (os.environ.get("SHERPA_OPENCODE_IDLE_TIMEOUT_SEC") or "300").strip()
        try:
            idle_timeout_sec = max(0, min(int(idle_timeout_raw), 86_400))
        except Exception:
            idle_timeout_sec = 300
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
        )

        done_path = self.working_dir / SENTINEL

        # Build prompt body once (mirrors original behaviour).
        if isinstance(instructions, (list, tuple)):
            tasks = "\n".join(str(i) for i in instructions)
        else:
            tasks = str(instructions)

        prompt_parts: List[str] = [
            "You are OpenCode running in a local Git repository.",
            "The repository is mounted at /repo; use relative paths or /repo (avoid /shared/output).",
            "GitNexus MCP tooling is available for codebase dependency/call-flow analysis; use it before guessing architecture details.",
            "Apply the edits requested below. Avoid refactors and unrelated changes.",
            "IMPORTANT ENV NOTE: The build/fuzz runtime environment is a separate container managed by the workflow, "
            "not this OpenCode execution environment. Do not infer runtime availability from this environment.",
            "Typical runtime images are sherpa-fuzz-cpp:latest or sherpa-fuzz-java:latest; "
            "OpenCode must only edit source files and must not attempt runtime verification.",
            "CRITICAL RULE: You MUST NOT execute build/test/fuzz commands or run binaries. "
            "Read-only commands (rg, ls, cat, find, sed) are allowed for inspection. "
            "Your ONLY job is to create and edit source files. "
            "Do NOT run cmake, make, gcc, clang, python, cargo, javac, mvn, gradle, npm, or similar build/run tools. "
            "The build and test steps are handled by a separate automated system. "
            "If you run build/test commands, the workflow will break.",
            "MANDATORY COMPLETION SIGNAL: You MUST create `./done` before exit. "
            "Without `./done`, this run is treated as failure and all edits are discarded.",
            "When ALL tasks are complete:",
            "  1) Print a short summary.",
            "  2) Create/overwrite a file called 'done' in the repo root (./done).",
            "     Put the relative path to the single most relevant file you created or modified on the first line.",
            "     Example: `echo fuzz/build.py > done`",
            f"## Tasks\n{tasks}",
        ]

        if additional_context:
            prompt_parts.append(
                textwrap.dedent(
                    f"""
                    ---
                    ### Additional context
                    {additional_context.strip()}
                    ---
                    """
                )
            )

        prompt = "\n".join(prompt_parts).strip()
        prompt_hash = _sha256_text(prompt)
        context_hash = _sha256_text(additional_context or "")

        # ----------------------------------------------------------------
        # Outer loop – retry full patch attempt if no diff produced.
        # ----------------------------------------------------------------
        try:
            self._maybe_prepare_gitnexus_context()
        except Exception as e:
            LOGGER.warning("[OpenCodeHelper] GitNexus pre-analysis skipped (non-fatal): %s", e)

        for attempt in range(1, max_attempts + 1):
            LOGGER.info("[OpenCodeHelper] patch attempt %d/%d", attempt, max_attempts)

            done_path.unlink(missing_ok=True)

            # Baseline diff for this run: later passes may already have a diff
            # from earlier steps (e.g., Pass A creates fuzz/PLAN.md). We only
            # consider this run successful if the diff changes relative to this
            # baseline.
            try:
                baseline_diff = self._git_diff_head()
            except Exception:
                baseline_diff = ""

            run_meta: dict = {
                "ts": time.time(),
                "attempt": attempt,
                "max_attempts": max_attempts,
                "codex_cli": self.codex_cli,
                "codex_model": self.codex_model,
                "resolved_model": "",
                "prompt_hash": prompt_hash,
                "context_hash": context_hash,
                "working_dir": str(self.working_dir),
                "status": "running",
                "repo_root": str(self.working_dir),
            }

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

                # If we're using a dedicated opencode container, default to docker CLI.
                if _docker_opencode_image():
                    cli_exe = "docker"

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
                if _docker_opencode_image():
                    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", self.working_dir.name or "repo").strip("-") or "repo"
                    run_name = f"sherpa-opencode-{slug}-{os.getpid()}-{int(time.time())}-{cli_try}-{attempt}".lower()
                    env["SHERPA_OPENCODE_RUN_NAME"] = run_name
                cmd: list[str] = ["run"]
                model = _resolve_opencode_model(env)
                if model:
                    cmd += ["--model", model]
                run_meta["resolved_model"] = model or ""
                cmd.append(prompt)

                try:
                    _apply_opencode_exec_policy(env)
                    image = _docker_opencode_image()
                    if image:
                        _ensure_opencode_image(image, env)
                    full_cmd = _build_opencode_cmd(cli_exe, cmd, self.working_dir, env)
                    proc = subprocess.Popen(
                        full_cmd,
                        cwd=self.working_dir,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=None if _docker_opencode_image() else env,
                        text=True,
                        errors="replace",
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"Failed to launch OpenCode CLI: {cli_exe} (cwd={self.working_dir}). "
                        "Make sure Docker is available (for containerized opencode) or "
                        "OpenCode is installed and accessible to the server process."
                    ) from e

                start_time = time.time()
                saw_retry_error = False
                last_heartbeat = 0.0
                last_activity_ts = start_time
                last_diff_probe_ts = start_time
                last_seen_diff = baseline_diff

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

                def _kill_proc() -> None:
                    if proc.poll() is None:
                        try:
                            proc.terminate()
                            proc.wait(timeout=4)
                        except Exception:
                            pass
                        if proc.poll() is None:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                    _cleanup_docker_run()

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

                        if idle_timeout_sec > 0:
                            idle_for = now - last_activity_ts
                            if idle_for > idle_timeout_sec:
                                LOGGER.warning(
                                    "[CodexHelper] idle timeout; killing opencode (idle=%.0fs)",
                                    idle_for,
                                )
                                saw_retry_error = True
                                print(
                                    "[OpenCodeHelper] idle timeout after "
                                    f"{idle_for:.0f}s without activity; terminating agent"
                                )
                                _kill_proc()
                                break

                        if elapsed > timeout:
                            LOGGER.warning("[CodexHelper] hard timeout; killing opencode")
                            saw_retry_error = True
                            print(f"[OpenCodeHelper] hard timeout after {elapsed:.0f}s; terminating agent")
                            _kill_proc()
                            break

                        # Heartbeat so job logs keep moving even if the agent is quiet.
                        if (now - last_heartbeat) > 10.0:
                            last_heartbeat = now
                            print(f"[OpenCodeHelper] running… elapsed={elapsed:.0f}s")
                            if idle_timeout_sec > 0 and (now - last_diff_probe_ts) >= 8.0:
                                last_diff_probe_ts = now
                                try:
                                    probed_diff = self._git_diff_head()
                                    if probed_diff != last_seen_diff:
                                        last_seen_diff = probed_diff
                                        last_activity_ts = now
                                except Exception:
                                    pass

                        if done_path.exists():
                            LOGGER.info("[OpenCodeHelper] done flag detected")
                            print("[OpenCodeHelper] done flag detected; terminating")
                            _kill_proc()
                            break

                        # Try to get output without blocking.
                        try:
                            item = out_q.get(timeout=0.2)
                        except queue.Empty:
                            item = None

                        if item is EOF:
                            break
                        if isinstance(item, str) and item:
                            print(item, end="")
                            captured_chunks.append(item)
                            last_activity_ts = now
                            if any(err in item for err in RETRY_ERRORS) and not _bool_env("SHERPA_OPENCODE_IGNORE_RETRY_ERRORS", False):
                                LOGGER.warning("[OpenCodeHelper] retryable error detected → abort")
                                saw_retry_error = True
                                _kill_proc()
                                break

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
                                print(item2, end="")
                                captured_chunks.append(item2)
                    except Exception:
                        pass
                    try:
                        t.join(timeout=1.0)
                    except Exception:
                        pass
                    _cleanup_docker_run()

                if saw_retry_error:
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
                LOGGER.warning("[OpenCodeHelper] sentinel not created; next attempt")
                print("[OpenCodeHelper] sentinel not created; next attempt")
                run_meta["status"] = "retry_no_sentinel"
                run_meta["cli_retries_used"] = cli_try
                _append_opencode_metadata(self.working_dir, run_meta)
                continue  # outer attempt loop

            # Refresh repo to ensure it sees new changes.
            self._git_add_all()

            if diff_changed or self._git_diff_head() != baseline_diff:
                LOGGER.info("[OpenCodeHelper] diff produced — success")
                run_meta["status"] = "success"
                run_meta["cli_retries_used"] = cli_try
                _append_opencode_metadata(self.working_dir, run_meta)
                return "".join(captured_chunks)

            LOGGER.info("[OpenCodeHelper] sentinel present but no diff; next attempt")
            print("[OpenCodeHelper] sentinel present but no diff; next attempt")
            run_meta["status"] = "retry_no_diff"
            run_meta["cli_retries_used"] = cli_try
            _append_opencode_metadata(self.working_dir, run_meta)

        LOGGER.warning("[OpenCodeHelper] exhausted attempts — no edits produced")
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


# ---------------------------------------------------------------------------
# Backwards-compat alias – internal code may still import CodexPatcher.
# ---------------------------------------------------------------------------


CodexPatcher = CodexHelper
