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

Wrapper around the OpenAI Codex CLI.

This helper preserves the public API and the success contract used throughout
the codebase:

    - The agent must write a sentinel file `./done` when finished.
    - We only treat a run as successful if a `git diff HEAD` is produced.

Key implementation goals:
    - **Windows compatibility**: avoid `pty` and Unix-only signal handling.
    - Robust retry + timeout behavior.
    - Stream output live to stdout while capturing it.

The CLI used is the Codex binary `codex` in non-interactive mode (`codex exec`).
"""

from __future__ import annotations

import logging
import os
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
    """Wrapper around Codex CLI with robust retry logic.

    Note: the class name is kept for backward compatibility.
    """

    def __init__(
        self,
        *,
        repo_path: Path,
        ai_key_path: str | None = None,
        copy_repo: bool = True,
        scratch_space: Path | None = None,
        codex_cli: str = "codex",
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
        # Keep names for compatibility with older config/env.
        self.codex_cli = str(codex_cli or "codex")
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
        # Codex CLI can authenticate via saved login or API key.
        if ai_key_path:
            key_path = Path(ai_key_path).expanduser()
            if key_path.is_file():
                key = key_path.read_text(encoding="utf-8", errors="ignore").strip()
                if key:
                    # Prefer OPENAI_API_KEY to align with Codex docs and common tooling.
                    os.environ.setdefault("OPENAI_API_KEY", key)
                    # `codex exec` also supports CODEX_API_KEY in automation contexts.
                    os.environ.setdefault("CODEX_API_KEY", key)

        LOGGER.debug("CodexHelper working directory: %s", self.working_dir)

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
            if r.returncode != 0:
                # If HEAD is missing for any reason, attempt to repair once.
                self._ensure_git_repo_docker()
                r = self._docker_git(["diff", "HEAD"], check=False)
            return (r.stdout or "").strip("\n")

        assert self.repo is not None
        return self.repo.git.diff("HEAD")

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
        """Execute Codex with robust retry logic and return its stdout or *None*."""

        SENTINEL = "done"
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
            "You are Codex running in a local Git repository.",
            "Apply the edits requested below. Avoid refactors and unrelated changes.",
            "Do NOT run any build or test commands unless explicitly asked.",
            "When ALL tasks are complete:",
            "  1) Print a short summary.",
            "  2) Create/overwrite a file called 'done' in the repo root (./done).",
            "     Put the relative path to the single most relevant file you created or modified on the first line.",
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

        # ----------------------------------------------------------------
        # Outer loop – retry full patch attempt if no diff produced.
        # ----------------------------------------------------------------

        for attempt in range(1, max_attempts + 1):
            LOGGER.info("[CodexHelper] patch attempt %d/%d", attempt, max_attempts)

            done_path.unlink(missing_ok=True)

            # Baseline diff for this run: later passes may already have a diff
            # from earlier steps (e.g., Pass A creates fuzz/PLAN.md). We only
            # consider this run successful if the diff changes relative to this
            # baseline.
            try:
                baseline_diff = self._git_diff_head()
            except Exception:
                baseline_diff = ""

            # ----------------------------------------------------------------
            # Inner loop – retry CLI invocation on transient errors.
            # ----------------------------------------------------------------

            cli_try = 0
            backoff = initial_backoff
            captured_chunks: List[str] = []
            accept_diff_without_done = os.environ.get("SHERPA_ACCEPT_DIFF_WITHOUT_DONE", "1").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            last_diff_check = 0.0

            while cli_try < max_cli_retries:
                cli_try += 1
                LOGGER.info("[CodexHelper] launch #%d (backoff=%.1fs)", cli_try, backoff)

                # Resolve CLI path early so missing executables produce an actionable error.
                cli_exe = shutil.which(self.codex_cli)
                if cli_exe is not None and os.name == "nt":
                    # On Windows, npm sometimes provides both `codex` and `codex.cmd`.
                    # The extension-less file may not be directly executable via CreateProcess
                    # and can trigger: [WinError 193] %1 is not a valid Win32 application.
                    p = Path(cli_exe)
                    if p.suffix == "" and p.with_suffix(".cmd").is_file():
                        cli_exe = str(p.with_suffix(".cmd"))
                if cli_exe is None and os.name == "nt":
                    # Common location for npm global bin on Windows.
                    appdata = os.environ.get("APPDATA")
                    if appdata:
                        for candidate in (Path(appdata) / "npm" / "codex.cmd", Path(appdata) / "npm" / "codex"):
                            if candidate.is_file():
                                cli_exe = str(candidate)
                                break

                if cli_exe is None:
                    raise FileNotFoundError(
                        f"Codex CLI not found: '{self.codex_cli}'. "
                        "Ensure 'codex' is installed and on PATH (e.g. npm global bin), "
                        "or pass the full path via --codex-cli."
                    )

                # Codex non-interactive mode.
                # - We need write access to edit the repo.
                # - Keep sandbox as narrow as possible; allow override.
                sandbox = (self.sandbox_mode or "workspace-write").strip()
                if self.dangerous_bypass:
                    sandbox = "danger-full-access"

                cmd: list[str] = [
                    cli_exe,
                    "exec",
                    "--full-auto",
                    "--sandbox",
                    sandbox,
                    prompt,
                ]

                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=self.working_dir,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=os.environ.copy(),
                        text=True,
                        errors="replace",
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"Failed to launch Codex CLI: {cmd[0]} (cwd={self.working_dir}). "
                        "Make sure Codex is installed and accessible to the server process."
                    ) from e

                start_time = time.time()
                saw_retry_error = False
                last_heartbeat = 0.0

                def _kill_proc() -> None:
                    if proc.poll() is not None:
                        return
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
                            LOGGER.warning("[CodexHelper] hard timeout; killing codex")
                            saw_retry_error = True
                            print(f"[CodexHelper] hard timeout after {elapsed:.0f}s; terminating agent")
                            _kill_proc()
                            break

                        # Heartbeat so job logs keep moving even if the agent is quiet.
                        if (now - last_heartbeat) > 10.0:
                            last_heartbeat = now
                            print(f"[CodexHelper] running… elapsed={elapsed:.0f}s")

                        # If Codex wrote files but forgot the sentinel, accept the diff and move on.
                        if accept_diff_without_done and (now - last_diff_check) > 2.0:
                            last_diff_check = now
                            try:
                                current_diff = self._git_diff_head()
                                if current_diff and current_diff != baseline_diff:
                                    print("[CodexHelper] diff detected without sentinel; accepting and terminating")
                                    _kill_proc()
                                    break
                            except Exception:
                                pass

                        if done_path.exists():
                            LOGGER.info("[CodexHelper] done flag detected")
                            print("[CodexHelper] done flag detected; terminating")
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
                            if any(err in item for err in RETRY_ERRORS):
                                LOGGER.warning("[CodexHelper] retryable error detected → abort")
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
                if accept_diff_without_done and diff_changed:
                    LOGGER.info("[CodexHelper] diff produced without sentinel — accepting")
                    print("[CodexHelper] diff produced without sentinel — accepting")
                    return "".join(captured_chunks)

                LOGGER.warning("[CodexHelper] sentinel not created; next attempt")
                print("[CodexHelper] sentinel not created; next attempt")
                continue  # outer attempt loop

            # Refresh repo to ensure it sees new changes.
            self._git_add_all()

            if diff_changed or self._git_diff_head() != baseline_diff:
                LOGGER.info("[CodexHelper] diff produced — success")
                return "".join(captured_chunks)

            LOGGER.info("[CodexHelper] sentinel present but no diff; next attempt")
            print("[CodexHelper] sentinel present but no diff; next attempt")

        LOGGER.warning("[CodexHelper] exhausted attempts — no edits produced")
        return None


# ---------------------------------------------------------------------------
# Backwards-compat alias – internal code may still import CodexPatcher.
# ---------------------------------------------------------------------------


CodexPatcher = CodexHelper
