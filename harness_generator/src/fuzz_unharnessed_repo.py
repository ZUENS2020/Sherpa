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

"""
fuzz_unharnessed_repo.py (non-OSS-Fuzz, local workflow)
────────────────────

Refactors the OSS-Fuzz-centric generator into a generic workflow that:
  • clones an arbitrary Git repo,
  • has OpenCode plan targets, synthesize a local libFuzzer/Jazzer harness + build glue,
  • iteratively fixes build errors,
  • generates initial seeds,
  • runs the fuzzer locally,
  • triages any crash and packages a reproducible challenge bundle.

Relies on the existing CodexHelper (OpenCode-backed).
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import queue
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv

try:
    from git import Repo, exc as git_exc  # type: ignore
except Exception:  # pragma: no cover
    Repo = None  # type: ignore
    git_exc = None  # type: ignore


TOOL_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_FUZZ_CPP = TOOL_ROOT / "docker" / "Dockerfile.fuzz-cpp"
DOCKERFILE_FUZZ_JAVA = TOOL_ROOT / "docker" / "Dockerfile.fuzz-java"

DEFAULT_DOCKER_IMAGE_CPP = os.environ.get("SHERPA_DOCKER_IMAGE_CPP", "sherpa-fuzz-cpp:latest")
DEFAULT_DOCKER_IMAGE_JAVA = os.environ.get("SHERPA_DOCKER_IMAGE_JAVA", "sherpa-fuzz-java:latest")

DEFAULT_GIT_DOCKER_IMAGE = os.environ.get("SHERPA_GIT_DOCKER_IMAGE", "alpine/git")

# Clone reliability knobs (useful on restricted networks).
GIT_CLONE_RETRIES = int(os.environ.get("SHERPA_GIT_CLONE_RETRIES", "2"))
GIT_DOCKER_CLONE_TIMEOUT_SEC = int(os.environ.get("SHERPA_GIT_DOCKER_CLONE_TIMEOUT_SEC", "45"))
GIT_HOST_CLONE_TIMEOUT_SEC = int(os.environ.get("SHERPA_GIT_HOST_CLONE_TIMEOUT_SEC", "90"))

# Optional: GitHub mirror support for regions where github.com is unreachable.
#
# Configure via env:
# - SHERPA_GITHUB_MIRROR: base URL used to replace "https://github.com/".
#   Example: "https://mirror.example/github.com/"
# - SHERPA_GIT_MIRRORS: comma-separated list of mirror specs. Each item can be:
#   - A template containing "{url}" (e.g., "https://proxy.example/{url}")
#   - A base URL (e.g., "https://mirror.example/github.com/")
# NOTE: These are intentionally read at runtime (not import time) so a running
# web server can apply updated config without restart.


def _get_sherpa_github_mirror() -> str:
    return os.environ.get("SHERPA_GITHUB_MIRROR", "").strip()


def _get_sherpa_git_mirrors() -> str:
    return os.environ.get("SHERPA_GIT_MIRRORS", "").strip()

# Make CodexHelper discoverable in both "package" and "flat script" use.
try:
    from .codex_helper import CodexHelper  # type: ignore
except Exception:  # pragma: no cover
    from codex_helper import CodexHelper  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_SANITIZER = os.environ.get("SHERPA_SANITIZER", "address")
MAX_BUILD_RETRIES = int(os.environ.get("SHERPA_MAX_BUILD_RETRIES", "3"))
CODEX_ANALYSIS_MODEL = os.environ.get("CODEX_ANALYSIS_MODEL", "sonnet")
CODEX_APPROVAL_MODE = os.environ.get("CODEX_APPROVAL_MODE", "full-auto")

FUZZ_DIR = "fuzz"
FUZZ_OUT_DIR = "fuzz/out"
FUZZ_CORPUS_DIR = "fuzz/corpus"
ARTIFACT_PREFIX = "artifacts"
FUZZ_SYSTEM_PACKAGES_FILE = os.environ.get("SHERPA_FUZZ_SYSTEM_PACKAGES_FILE", "fuzz/system_packages.txt")

# Recognize fuzzer executables by name pattern.
FUZZ_BIN_PAT = re.compile(r".*(fuzz|_fuzzer|Fuzzer)$", re.IGNORECASE)


# ────────────────────────────────────────────────────────────────────────────
# Exceptions
# ────────────────────────────────────────────────────────────────────────────

class HarnessGeneratorError(RuntimeError):
    pass


# ────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────

def make_executable(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def read_text_safely(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def write_text_safely(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", errors="replace")


def _workflow_opencode_cli_retries() -> int:
    raw = (os.environ.get("SHERPA_WORKFLOW_OPENCODE_CLI_RETRIES") or "2").strip()
    try:
        return max(1, min(int(raw), 8))
    except Exception:
        return 2


def _default_diff_excludes() -> set[str]:
    return {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        "out",
        "fuzz/out",
        "fuzz/corpus",
        "fuzz/build",
        "challenge_bundle",
        "unreproducible",
        "false_positive",
    }


def _should_skip_path(path: Path, *, repo_root: Path, exclude_dirs: set[str]) -> bool:
    try:
        rel = path.relative_to(repo_root)
    except Exception:
        return True
    if rel.name == "done":
        return True
    parts = rel.parts
    for i in range(len(parts)):
        seg = "/".join(parts[: i + 1])
        if seg in exclude_dirs or parts[i] in exclude_dirs:
            return True
    return False


def _read_text_for_diff(path: Path, *, max_bytes: int = 2_000_000) -> str | None:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    if len(data) > max_bytes:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return None


def snapshot_repo_text(
    repo_root: Path,
    *,
    exclude_dirs: set[str] | None = None,
    max_bytes: int = 2_000_000,
) -> dict[str, str]:
    """Capture text snapshots of repo files for patch generation."""
    excludes = _default_diff_excludes() | (exclude_dirs or set())
    snap: dict[str, str] = {}
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if _should_skip_path(p, repo_root=repo_root, exclude_dirs=excludes):
            continue
        text = _read_text_for_diff(p, max_bytes=max_bytes)
        if text is None:
            continue
        try:
            rel = p.relative_to(repo_root).as_posix()
        except Exception:
            continue
        snap[rel] = text
    return snap


def write_patch_from_snapshot(
    snapshot: dict[str, str],
    repo_root: Path,
    out_path: Path,
    *,
    exclude_dirs: set[str] | None = None,
    max_bytes: int = 2_000_000,
) -> list[str]:
    """Write a unified diff between snapshot and current repo state."""
    current = snapshot_repo_text(repo_root, exclude_dirs=exclude_dirs, max_bytes=max_bytes)
    changed_files: list[str] = []
    diff_lines: list[str] = []

    all_keys = set(snapshot.keys()) | set(current.keys())
    for rel in sorted(all_keys):
        before = snapshot.get(rel)
        after = current.get(rel)
        if before == after:
            continue
        changed_files.append(rel)
        before_lines = before.splitlines(keepends=True) if before is not None else []
        after_lines = after.splitlines(keepends=True) if after is not None else []
        diff_lines.extend(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )

    if diff_lines:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(diff_lines), encoding="utf-8", errors="replace")
    else:
        out_path.unlink(missing_ok=True)

    return changed_files


def hexdump(path: Path, limit_bytes: int = 512) -> str:
    try:
        return subprocess.check_output(
            ["xxd", "-g1", "-l", str(limit_bytes), str(path)],
            text=True,
        )
    except Exception:
        data = path.read_bytes()[:limit_bytes]
        lines = []
        for off in range(0, len(data), 16):
            chunk = data[off : off + 16]
            hex_bytes = " ".join(f"{b:02x}" for b in chunk)
            ascii_ = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{off:08x}: {hex_bytes:<47}  {ascii_}")
        return "\n".join(lines)


ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", re.MULTILINE)
def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def _tail_lines(s: str, *, max_lines: int = 80) -> str:
    s = (s or "").strip("\n")
    if not s:
        return ""
    lines = strip_ansi(s).splitlines()
    return "\n".join(lines[-max_lines:])


def _is_truthy_env(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _run_cmd_capture(
    cmd: Sequence[str],
    *,
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str, bool]:
    """Run a command capturing stdout/stderr for logging.

    Returns: (rc, stdout, stderr, timed_out)
    """

    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or "", False
    except subprocess.TimeoutExpired as te:
        stdout = te.stdout or ""
        stderr = te.stderr or ""
        return (
            124,
            stdout if isinstance(stdout, str) else "",
            stderr if isinstance(stderr, str) else "",
            True,
        )


def _set_git_core_filemode_off_host(repo_dir: Path) -> None:
    cmd = ["git", "-C", str(repo_dir), "config", "core.filemode", "false"]
    rc, out, err, _ = _run_cmd_capture(cmd)
    if rc != 0:
        if (t := _tail_lines(err)):
            print("[warn] (host/git) config core.filemode stderr (tail):\n" + textwrap.indent(t, "    "))
        if (t := _tail_lines(out)):
            print("[warn] (host/git) config core.filemode stdout (tail):\n" + textwrap.indent(t, "    "))


def _set_git_core_filemode_off_docker(repo_dir: Path) -> None:
    cmd = [
        "docker",
        "run",
        "--rm",
        *_docker_proxy_env_args(),
        "-v",
        f"{str(repo_dir)}:/repo",
        "-w",
        "/repo",
        DEFAULT_GIT_DOCKER_IMAGE,
        "config",
        "core.filemode",
        "false",
    ]
    rc, out, err, _ = _run_cmd_capture(cmd)
    if rc != 0:
        if (t := _tail_lines(err)):
            print("[warn] (docker/git) config core.filemode stderr (tail):\n" + textwrap.indent(t, "    "))
        if (t := _tail_lines(out)):
            print("[warn] (docker/git) config core.filemode stdout (tail):\n" + textwrap.indent(t, "    "))


def _docker_proxy_env_args() -> List[str]:
    """Return docker `-e` args for proxy-related env vars.

    If the proxy points to localhost/127.0.0.1, rewrite it to a host-accessible
    hostname for Docker Desktop (default: host.docker.internal).
    """

    docker_proxy_host = os.environ.get("SHERPA_DOCKER_PROXY_HOST", "host.docker.internal").strip()

    def _pick_env(*names: str) -> str:
        for n in names:
            v = os.environ.get(n)
            if v is not None and v.strip():
                return v.strip()
        return ""

    http_proxy = _pick_env("SHERPA_DOCKER_HTTP_PROXY", "HTTP_PROXY", "http_proxy")
    https_proxy = _pick_env("SHERPA_DOCKER_HTTPS_PROXY", "HTTPS_PROXY", "https_proxy")
    no_proxy = _pick_env("SHERPA_DOCKER_NO_PROXY", "NO_PROXY", "no_proxy")

    def _rewrite_localhost_proxy(value: str) -> str:
        if not value:
            return value
        # Common patterns: http://127.0.0.1:7890, socks5://localhost:1080
        return re.sub(r"(?i)(?<=://)(localhost|127\.0\.0\.1)(?=[:/]|$)", docker_proxy_host, value)

    http_proxy = _rewrite_localhost_proxy(http_proxy)
    https_proxy = _rewrite_localhost_proxy(https_proxy)

    args: List[str] = []
    if http_proxy:
        args.extend(["-e", f"HTTP_PROXY={http_proxy}", "-e", f"http_proxy={http_proxy}"])
    if https_proxy:
        args.extend(["-e", f"HTTPS_PROXY={https_proxy}", "-e", f"https_proxy={https_proxy}"])
    if no_proxy:
        args.extend(["-e", f"NO_PROXY={no_proxy}", "-e", f"no_proxy={no_proxy}"])
    return args


def _host_git_proxy_override_args() -> List[str]:
    """Return `git -c ...` overrides to avoid broken localhost proxy configs.

    Some environments have `http.proxy/https.proxy` set to 127.0.0.1/localhost
    but the proxy app isn't running. In that case, host `git clone` fails even
    when direct network access would work.
    """

    disable = os.environ.get("SHERPA_GIT_DISABLE_PROXY", "").strip().lower() in {"1", "true", "yes"}

    def _git_config_get(key: str) -> str:
        try:
            proc = subprocess.run(
                ["git", "config", "--global", "--get", key],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return (proc.stdout or "").strip() if proc.returncode == 0 else ""
        except Exception:
            return ""

    def _is_local_proxy_unreachable(proxy_value: str) -> bool:
        if not proxy_value:
            return False
        raw = proxy_value.strip()
        # If it's missing a scheme, urlparse won't pick up hostname/port.
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
        host = (parsed.hostname or "").strip().lower()
        port = parsed.port
        if host not in {"127.0.0.1", "localhost"}:
            return False
        if port is None:
            return False
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return False
        except Exception:
            return True

    if not disable:
        http_proxy = _git_config_get("http.proxy")
        https_proxy = _git_config_get("https.proxy")
        if _is_local_proxy_unreachable(http_proxy) or _is_local_proxy_unreachable(https_proxy):
            disable = True

    if not disable:
        return []

    return ["-c", "http.proxy=", "-c", "https.proxy="]


def _candidate_clone_urls(url: str) -> List[str]:
    """Return clone URLs, optionally extended by explicitly configured mirrors."""

    urls: List[str] = [url]
    if not url.startswith("https://github.com/"):
        # Non-GitHub URLs are returned as-is to avoid breaking custom hosts.
        return urls

    mirror_specs: List[str] = []
    sherpa_git_mirrors = _get_sherpa_git_mirrors()
    if sherpa_git_mirrors:
        mirror_specs.extend([p.strip() for p in sherpa_git_mirrors.split(",") if p.strip()])

    sherpa_github_mirror = _get_sherpa_github_mirror()
    if sherpa_github_mirror:
        mirror_specs.append(sherpa_github_mirror)

    gh_path = url[len("https://github.com/") :]

    for spec in mirror_specs:
        candidate = ""
        if "{url}" in spec:
            candidate = spec.replace("{url}", url)
        else:
            base = spec.rstrip("/")
            # Common pattern: <base>/<owner>/<repo>.git
            if base.endswith("github.com") or base.endswith("github.com/"):
                candidate = f"{base}/{gh_path}"
            else:
                candidate = f"{base}/{gh_path}"

        if candidate and candidate not in urls:
            urls.append(candidate)

    return urls


# ────────────────────────────────────────────────────────────────────────────
# Core generator
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class RepoSpec:
    url: str
    ref: Optional[str] = None       # branch/tag/commit
    workdir: Optional[Path] = None  # where to clone; auto if None


@dataclass
class FuzzerRunResult:
    rc: int
    new_artifacts: List[Path]
    crash_found: bool
    crash_evidence: str
    first_artifact: str
    log_tail: str
    error: str
    run_error_kind: str
    final_cov: int = 0
    final_ft: int = 0
    final_corpus_files: int = 0
    final_corpus_size_bytes: int = 0
    final_execs_per_sec: int = 0
    final_rss_mb: int = 0
    final_iteration: int = 0
    corpus_files: int = 0
    corpus_size_bytes: int = 0


_LIBFUZZER_PROGRESS_RE = re.compile(
    r"#(?P<iter>\d+)\s+"
    r"(?P<kind>NEW|REDUCE|pulse)\s+"
    r"cov:\s*(?P<cov>\d+)\s+"
    r"ft:\s*(?P<ft>\d+)\s+"
    r"corp:\s*(?P<corp_files>\d+)/(?P<corp_size>\S+)"
    r"(?:.*?exec/s:\s*(?P<execs>\d+))?"
    r"(?:.*?rss:\s*(?P<rss>\d+)Mb)?",
    re.IGNORECASE,
)


def _parse_size_token_to_bytes(token: str) -> int:
    txt = (token or "").strip()
    if not txt:
        return 0
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)([kmg]?)(?:i?b)?$", txt, re.IGNORECASE)
    if not m:
        return 0
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    scale = 1
    if unit == "k":
        scale = 1024
    elif unit == "m":
        scale = 1024 * 1024
    elif unit == "g":
        scale = 1024 * 1024 * 1024
    return int(val * scale)


def parse_libfuzzer_final_stats(text: str) -> Dict[str, int]:
    """Extract the final libFuzzer progress tuple from a log blob."""
    stats = {
        "iteration": 0,
        "cov": 0,
        "ft": 0,
        "corpus_files": 0,
        "corpus_size_bytes": 0,
        "execs_per_sec": 0,
        "rss_mb": 0,
    }
    if not text:
        return stats
    for line in text.splitlines():
        m = _LIBFUZZER_PROGRESS_RE.search(line)
        if not m:
            continue
        stats["iteration"] = int(m.group("iter") or 0)
        stats["cov"] = int(m.group("cov") or 0)
        stats["ft"] = int(m.group("ft") or 0)
        stats["corpus_files"] = int(m.group("corp_files") or 0)
        stats["corpus_size_bytes"] = _parse_size_token_to_bytes(m.group("corp_size") or "")
        stats["execs_per_sec"] = int(m.group("execs") or 0)
        stats["rss_mb"] = int(m.group("rss") or 0)
    return stats


class NonOssFuzzHarnessGenerator:
    """
    Multi-pass workflow using CodexHelper to:
      1) PLAN targets,
      2) SYNTHESIZE harness + local build glue,
      3) BUILD with retries (Codex fixes),
      4) SEED corpus,
      5) RUN fuzzers,
      6) TRIAGE & PACKAGE results.
    """

    def __init__(
        self,
        repo_spec: RepoSpec,
        *,
        ai_key_path: Path,
        sanitizer: str = DEFAULT_SANITIZER,
        codex_cli: str = "opencode",
        time_budget_per_target: int = 900,  # seconds for an initial run
        codex_dangerous: bool = False,
        codex_sandbox_mode: Optional[str] = None,
        rss_limit_mb: int = 8192,
        max_len: int = 1024,
        max_build_retries: int = MAX_BUILD_RETRIES,
        docker_image: Optional[str] = None,
    ) -> None:
        self.repo_spec = repo_spec
        self.sanitizer = sanitizer
        self.codex_cli = codex_cli
        self.time_budget = time_budget_per_target
        # Workflow can override these per-step/per-batch at runtime.
        self.seed_generation_timeout_sec: Optional[int] = None
        self.current_run_time_budget_sec: Optional[int] = None
        self.current_run_hard_timeout_sec: Optional[int] = None
        self.rss_limit_mb = rss_limit_mb
        self.max_len = max_len
        self.max_build_retries = max_build_retries
        self.docker_image = docker_image
        self.logger = logging.getLogger(__name__)

        # Index of the generation round for this repo (1 == first). The caller
        # may overwrite it when running multiple rounds in the same workdir.
        self.round_index: int = 1

        self.repo_root: Path = self._clone_repo(repo_spec)

        self._dockerfile_path: Optional[Path] = None
        if self.docker_image:
            self.docker_image, self._dockerfile_path = self._resolve_docker_image(self.docker_image)
            self._ensure_docker_image(self.docker_image, dockerfile=self._dockerfile_path)

        self._ensure_fuzz_dirs()

        self.patcher = CodexHelper(
            repo_path=self.repo_root,
            ai_key_path=str(ai_key_path),
            copy_repo=False,                   # operate in-place for determinism
            codex_cli=self.codex_cli,
            codex_model=CODEX_ANALYSIS_MODEL,
            approval_mode=CODEX_APPROVAL_MODE,
            dangerous_bypass=codex_dangerous,
            sandbox_mode=codex_sandbox_mode,
            git_docker_image=self.docker_image if self.docker_image else None,
        )

        print(f"[*] Ready (repo={self.repo_root})")

    def _ensure_fuzz_dirs(self) -> None:
        self.fuzz_dir = self.repo_root / FUZZ_DIR
        self.fuzz_out_dir = self.repo_root / FUZZ_OUT_DIR
        self.fuzz_corpus_dir = self.repo_root / FUZZ_CORPUS_DIR
        self.fuzz_dir.mkdir(parents=True, exist_ok=True)
        self.fuzz_out_dir.mkdir(parents=True, exist_ok=True)
        self.fuzz_corpus_dir.mkdir(parents=True, exist_ok=True)

    def _detect_repo_language(self) -> str:
        """Best-effort language detection for choosing a fuzz runtime image.

        Returns one of: 'java', 'cpp', or 'unknown'.

        NOTE: We intentionally do NOT default to C/C++ anymore. Many repos are
        Python/JS/etc; silently selecting a heavy C++ toolchain image leads to
        confusing failures later.
        """

        # Strong Java signals
        for marker in ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"):
            if (self.repo_root / marker).is_file():
                return "java"
        if list(self.repo_root.rglob("*.java")):
            return "java"

        # C/C++ signals
        for marker in ("CMakeLists.txt", "configure.ac", "configure.in"):
            if (self.repo_root / marker).is_file():
                return "cpp"
        if list(self.repo_root.rglob("*.c")) or list(self.repo_root.rglob("*.cc")) or list(self.repo_root.rglob("*.cpp")) or list(self.repo_root.rglob("*.cxx")):
            return "cpp"

        return "unknown"

    def _resolve_docker_image(self, docker_image: str) -> Tuple[str, Path]:
        """Resolve docker image + dockerfile.

        docker_image may be a concrete image tag, or 'auto' to pick language-specific defaults.
        """

        if docker_image.strip().lower() == "auto":
            lang = self._detect_repo_language()
            if lang == "java":
                return DEFAULT_DOCKER_IMAGE_JAVA, DOCKERFILE_FUZZ_JAVA
            if lang == "cpp":
                return DEFAULT_DOCKER_IMAGE_CPP, DOCKERFILE_FUZZ_CPP
            raise HarnessGeneratorError(
                "Unable to auto-detect a supported fuzz toolchain for this repository. "
                "Supported: C/C++ (libFuzzer) and Java (Jazzer). "
                "Pass an explicit --docker-image (or set docker_image in the Web UI) to force a toolchain, "
                "or target a C/C++/Java project."
            )

        # Explicit image tag: default to C/C++ dockerfile unless user overrides via env.
        return docker_image, DOCKERFILE_FUZZ_CPP

    def _ensure_docker_image(self, image: str, *, dockerfile: Path) -> None:
        """Ensure the requested Docker image exists.

        This lowers the barrier on Windows: user can enable Docker mode and the
        tool will build the fuzz runtime image automatically if it's missing.
        """

        def _wait_for_docker_daemon_ready() -> None:
            wait_raw = os.environ.get("SHERPA_DOCKER_DAEMON_WAIT_SEC", "60")
            try:
                wait_sec = max(5, min(int(wait_raw), 300))
            except Exception:
                wait_sec = 60
            deadline = time.time() + wait_sec
            while True:
                try:
                    probe = subprocess.run(
                        ["docker", "info"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        text=True,
                    )
                    if probe.returncode == 0:
                        return
                except FileNotFoundError:
                    raise HarnessGeneratorError("Docker not found in PATH. Install Docker Desktop and ensure 'docker' is available.")
                except Exception:
                    pass
                if time.time() >= deadline:
                    raise HarnessGeneratorError("Docker daemon is not ready (timeout waiting for docker info).")
                time.sleep(2)

        # Ensure daemon is reachable first; this avoids startup race failures.
        _wait_for_docker_daemon_ready()

        # Fast path: image exists.
        try:
            probe = subprocess.run(
                ["docker", "image", "inspect", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                text=True,
            )
            if probe.returncode == 0:
                return
        except FileNotFoundError:
            raise HarnessGeneratorError("Docker not found in PATH. Install Docker Desktop and ensure 'docker' is available.")
        except Exception:
            # Continue to build attempt; it will error with details.
            pass

        if not dockerfile.is_file():
            raise HarnessGeneratorError(f"Dockerfile not found: {dockerfile}")

        print(f"[*] Docker image '{image}' not found. Building it now …")

        def _docker_network_precheck() -> None:
            """Best-effort precheck for common DNS/TLS connectivity failures."""
            flag = os.environ.get("SHERPA_DOCKER_NETWORK_PRECHECK", "1").strip().lower()
            if flag in {"0", "false", "no", "off"}:
                return

            # Avoid false DNS warnings when busybox probe image is not available locally.
            busybox_probe = subprocess.run(
                ["docker", "image", "inspect", "busybox:latest"],
                cwd=str(TOOL_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            if busybox_probe.returncode != 0:
                print("[*] Docker network precheck skipped (busybox:latest not present locally).")
                return

            probe_cmd = [
                "docker",
                "run",
                "--rm",
                "--pull=never",
                "busybox:latest",
                "sh",
                "-lc",
                "nslookup registry-1.docker.io >/dev/null 2>&1 || nslookup auth.docker.io >/dev/null 2>&1",
            ]
            try:
                proc = subprocess.run(
                    probe_cmd,
                    cwd=str(TOOL_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    check=False,
                    timeout=35,
                )
            except Exception:
                # Best-effort only: do not block build if precheck itself fails unexpectedly.
                return
            if proc.returncode == 0:
                print("[*] Docker network precheck passed (registry DNS reachable).")
                return

            out = (proc.stdout or "").strip()
            low_out = out.lower()
            if "unable to find image" in low_out and "busybox" in low_out:
                print("[*] Docker network precheck skipped (busybox image unavailable at runtime).")
                return
            print(
                "[warn] Docker network precheck failed: cannot resolve Docker registry DNS from container runtime. "
                "Build may fail with name-resolution/TLS timeouts."
            )
            if out:
                print("[warn] precheck output tail:\n" + "\n".join(out.splitlines()[-20:]))
            print(
                "[warn] recovery: verify DNS/proxy settings for Docker daemon and container network, "
                "then retry (set SHERPA_DOCKER_NETWORK_PRECHECK=0 to skip this precheck)."
            )

        _docker_network_precheck()

        def _is_transient_registry_error(lines: list[str]) -> bool:
            blob = "\n".join(lines).lower()
            needles = [
                "tls handshake timeout",
                "i/o timeout",
                "connection reset by peer",
                "context deadline exceeded",
                "temporary failure in name resolution",
                "net/http: request canceled",
                "no such host",
                "unexpected eof",
                '": eof',
                "get \"https://registry-1.docker.io/v2/\": eof",
                "proxyconnect tcp",
                "lookup registry-1.docker.io",
            ]
            return any(n in blob for n in needles)

        def _is_docker_daemon_unavailable(lines: list[str]) -> bool:
            blob = "\n".join(lines).lower()
            needles = [
                "cannot connect to the docker daemon",
                "is the docker daemon running",
                "connection refused",
                "dial tcp",
                "lookup sherpa-docker",
                "no such host",
            ]
            return any(n in blob for n in needles)

        def _is_buildkit_unavailable(lines: list[str]) -> bool:
            blob = "\n".join(lines).lower()
            needles = [
                "buildkit is enabled but the buildx component is missing or broken",
                "buildx component is missing or broken",
                "docker: 'buildx' is not a docker command",
            ]
            return any(n in blob for n in needles)

        def _run_build(build_cmd: list[str], *, buildkit: str | None = None) -> tuple[int, list[str]]:
            print(f"[*] ➜  {' '.join(build_cmd)}")
            try:
                env = os.environ.copy()
                if buildkit is not None:
                    env["DOCKER_BUILDKIT"] = buildkit
                else:
                    env.setdefault("DOCKER_BUILDKIT", "1")
                proc = subprocess.Popen(
                    build_cmd,
                    cwd=str(TOOL_ROOT),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    bufsize=1,
                )
            except FileNotFoundError:
                raise HarnessGeneratorError(
                    "Docker not found in PATH. Install Docker Desktop and ensure 'docker' is available."
                )

            assert proc.stdout is not None
            lines: list[str] = []
            for line in proc.stdout:
                lines.append(line)
                print(line, end="")
            rc = proc.wait()
            return rc, lines

        # Network to Docker Hub can be unstable in some regions; use a safer default.
        tries_raw = os.environ.get("SHERPA_DOCKER_BUILD_RETRIES", "6")
        try:
            max_tries = max(1, min(int(tries_raw), 8))
        except Exception:
            max_tries = 3

        backoff_s = 2.0
        last_rc = 1
        last_lines: list[str] = []
        force_classic_builder = os.environ.get("DOCKER_BUILDKIT", "").strip() == "0"
        for attempt in range(1, max_tries + 1):
            if force_classic_builder:
                cmd = ["docker", "build", "-t", image, "-f", str(dockerfile), str(TOOL_ROOT)]
                rc, lines = _run_build(cmd, buildkit="0")
            else:
                cmd = ["docker", "build", "--progress=plain", "-t", image, "-f", str(dockerfile), str(TOOL_ROOT)]
                rc, lines = _run_build(cmd, buildkit="1")

            if rc != 0:
                # Older docker builders do not support --progress; retry without it.
                if not force_classic_builder and any("unknown flag: --progress" in ln for ln in lines):
                    cmd = ["docker", "build", "-t", image, "-f", str(dockerfile), str(TOOL_ROOT)]
                    rc, lines = _run_build(cmd, buildkit="1")

                # BuildKit can be enabled without buildx; retry with classic builder.
                if rc != 0 and _is_buildkit_unavailable(lines):
                    if not force_classic_builder:
                        print("[warn] buildx unavailable; switching to classic docker builder (DOCKER_BUILDKIT=0).")
                    force_classic_builder = True
                    cmd = ["docker", "build", "-t", image, "-f", str(dockerfile), str(TOOL_ROOT)]
                    rc, lines = _run_build(cmd, buildkit="0")

            last_rc = rc
            last_lines = lines
            if rc == 0:
                return

            if attempt < max_tries and (_is_transient_registry_error(lines) or _is_docker_daemon_unavailable(lines)):
                reason = "daemon/network transient error"
                print(f"[warn] docker build {reason}; retrying in {backoff_s:.0f}s ({attempt}/{max_tries})")
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 20.0)
                continue
            break

        tail = "".join(last_lines[-120:]).strip()
        if tail:
            low_tail = tail.lower()
            recovery = ""
            if any(x in low_tail for x in ["temporary failure in name resolution", "no such host", "lookup registry-1.docker.io"]):
                recovery = (
                    "\nRecovery suggestion: Docker registry DNS lookup failed. Check daemon DNS config and "
                    "host resolver settings, then retry."
                )
            elif "tls handshake timeout" in low_tail or "x509:" in low_tail:
                recovery = (
                    "\nRecovery suggestion: Docker registry TLS handshake failed. Check proxy/CA certificates "
                    "and system clock, then retry."
                )
            elif "cannot connect to the docker daemon" in low_tail or "is the docker daemon running" in low_tail:
                recovery = (
                    "\nRecovery suggestion: Docker daemon is unreachable. Start/restart Docker service and "
                    "verify socket permission before retrying."
                )
            raise HarnessGeneratorError(
                f"Docker build failed (rc={last_rc}). Last output tail:\n{tail}{recovery}"
            )
        raise HarnessGeneratorError(f"Docker build failed (rc={last_rc}).")

    def _python_runner(self) -> str:
        # When executing build/run inside Docker, use the container's python.
        return "python3" if self.docker_image else sys.executable

    def _dockerize_cmd(self, cmd: Sequence[str], *, cwd: Path, env: Optional[Dict[str, str]]) -> List[str]:
        if not self.docker_image:
            return list(cmd)

        def _docker_container_name(mount_src: str, cmd_args: Sequence[str]) -> str:
            base = Path(mount_src).name
            base = re.sub(r"[^a-zA-Z0-9_.-]+", "-", base).strip("-.").lower()
            if not base:
                base = "sherpa"
            suffix = hashlib.sha1(mount_src.encode("utf-8", errors="ignore")).hexdigest()[:8]
            cmd_sig = hashlib.sha1(
                "\x1f".join(str(x) for x in cmd_args).encode("utf-8", errors="ignore")
            ).hexdigest()[:6]
            nonce = uuid.uuid4().hex[:6]
            name = f"{base}-{suffix}-{cmd_sig}-{nonce}"
            # Docker name limit is generous, but keep it short for readability.
            return name[:63]

        def _map_host_path_to_container(p: str) -> Optional[str]:
            """Map a host path under repo_root to a container path under /work.

            We mount repo_root into the container at /work, but many call sites naturally
            construct absolute host paths (especially on Windows). When passed to docker
            as the container argv, those host paths do not exist inside the container.
            """
            if not p:
                return None
            if p.startswith("/work/") or p == "/work":
                return None
            if not os.path.isabs(p):
                return None

            had_trailing_slash = p.endswith("/") or p.endswith("\\")

            # Use normcase-based prefix matching for Windows (case-insensitive paths).
            # Path.relative_to() is case-sensitive and will fail for e.g. 'C:\\' vs 'c:\\'.
            repo_root_abs = os.path.abspath(str(self.repo_root.resolve()))
            host_abs = os.path.abspath(p)

            repo_norm = os.path.normcase(repo_root_abs)
            host_norm = os.path.normcase(host_abs)

            if host_norm == repo_norm:
                rel_posix = "."
            elif host_norm.startswith(repo_norm + os.sep):
                rel = os.path.relpath(host_abs, repo_root_abs)
                rel_posix = rel.replace("\\", "/")
            else:
                return None

            container_path = "/work" if rel_posix in (".", "") else f"/work/{rel_posix}"
            if had_trailing_slash and not container_path.endswith("/"):
                container_path += "/"
            return container_path

        def _translate_arg(a: str) -> str:
            # Handle flags like -artifact_prefix=C:\...\artifacts/
            if "=" in a:
                k, v = a.split("=", 1)
                mapped = _map_host_path_to_container(v)
                if mapped is not None:
                    return f"{k}={mapped}"
            mapped = _map_host_path_to_container(a)
            return mapped if mapped is not None else a

        def _filter_env(e: Optional[Dict[str, str]]) -> Dict[str, str]:
            if not e:
                return {}

            allow_exact = {
                "HTTP_PROXY",
                "HTTPS_PROXY",
                "NO_PROXY",
                "http_proxy",
                "https_proxy",
                "no_proxy",
                # Sanitizer tuning
                "ASAN_OPTIONS",
                "UBSAN_OPTIONS",
                "MSAN_OPTIONS",
                "LSAN_OPTIONS",
                "TSAN_OPTIONS",
                # Toolchain overrides
                "CC",
                "CXX",
                "CFLAGS",
                "CXXFLAGS",
                "LDFLAGS",
                "CPATH",
                "C_INCLUDE_PATH",
                "CPLUS_INCLUDE_PATH",
                # Jazzer/Java
                "JAVA_TOOL_OPTIONS",
                "JAZZER_JVM_ARGS",
                # Keys (if used)
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
            }
            allow_prefixes = (
                "SHERPA_",
                "JAZZER_",
            )

            filtered: Dict[str, str] = {}
            for k, v in e.items():
                if v is None:
                    continue
                if k in allow_exact or k.startswith(allow_prefixes):
                    filtered[k] = str(v)
            return filtered

        mount_src = str(self.repo_root.resolve())
        rel = "."
        try:
            rel = os.path.relpath(str(cwd.resolve()), str(self.repo_root.resolve()))
        except Exception:
            rel = "."
        rel = "." if rel in (".", "") else rel.replace("\\", "/")
        workdir_in_container = "/work" if rel == "." else f"/work/{rel}"

        docker_cmd: List[str] = [
            "docker",
            "run",
            "--rm",
            "--name",
            _docker_container_name(mount_src, cmd),
            "--label",
            f"sherpa.repo_root={Path(mount_src).name}",
            "--label",
            f"sherpa.repo_root_sha1={hashlib.sha1(mount_src.encode('utf-8', errors='ignore')).hexdigest()}",
            "-v",
            f"{mount_src}:/work",
            "-w",
            workdir_in_container,
        ]

        filtered_env = _filter_env(env)
        if "CC" not in filtered_env:
            filtered_env["CC"] = "clang"
        if "CXX" not in filtered_env:
            filtered_env["CXX"] = "clang++"
        if filtered_env:
            for k, v in filtered_env.items():
                docker_cmd += ["-e", f"{k}={v}"]

        # Default: translate all args for container.
        translated_for_exec = [_translate_arg(a) for a in cmd]
        translated_cmd = list(translated_for_exec)

        # Prepare optional shell prelude for two cases:
        # 1) fuzzer run: ensure -artifact_prefix directory exists
        # 2) build run: auto-install declared system deps before invoking build.py/build.sh
        artifacts_path = ""
        for a in translated_for_exec:
            if a.startswith("-artifact_prefix="):
                artifacts_path = a.split("=", 1)[1]
                break

        def _is_build_entry_arg(a: str) -> bool:
            norm = (a or "").strip().replace("\\", "/")
            if norm in {"build.py", "./build.py", "fuzz/build.py", "build.sh", "./build.sh", "fuzz/build.sh"}:
                return True
            return norm.endswith("/build.py") or norm.endswith("/build.sh")

        should_autoinstall = any(_is_build_entry_arg(a) for a in translated_for_exec)
        dep_setup = ""
        if should_autoinstall and _is_truthy_env("SHERPA_AUTO_INSTALL_SYSTEM_DEPS", True):
            dep_rel = (FUZZ_SYSTEM_PACKAGES_FILE or "fuzz/system_packages.txt").replace("\\", "/").strip("/")
            dep_file = f"/work/{dep_rel}"
            dep_setup = textwrap.dedent(
                f"""
                dep_file={shlex.quote(dep_file)}
                if [ -f "$dep_file" ]; then
                    pkgs=""
                    while IFS= read -r line || [ -n "$line" ]; do
                        line="${{line%%#*}}"
                        line="$(printf '%s' "$line" | tr -d '\\r' | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
                        [ -n "$line" ] || continue
                        if printf '%s' "$line" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9+._-]*$'; then
                            pkgs="$pkgs $line"
                        else
                            echo "[warn] (docker/deps) skip invalid package token: $line"
                        fi
                    done < "$dep_file"

                    if [ -n "$pkgs" ]; then
                        missing_pkgs=""
                        for p in $pkgs; do
                            if command -v dpkg-query >/dev/null 2>&1; then
                                if dpkg-query -W -f='${{Status}}' "$p" 2>/dev/null | grep -q 'install ok installed'; then
                                    continue
                                fi
                            fi
                            missing_pkgs="$missing_pkgs $p"
                        done

                        if [ -z "$missing_pkgs" ]; then
                            echo "[*] (docker/deps) all requested packages already installed; skipping"
                        else
                            echo "[*] (docker/deps) installing from $dep_file:$missing_pkgs"
                            if command -v apt-get >/dev/null 2>&1; then
                                if ! apt-get update -o Acquire::Retries=3 -o Acquire::ForceIPv4=true; then
                                    echo "[warn] (docker/deps) apt-get update failed; continuing without auto-install"
                                elif ! DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends $missing_pkgs; then
                                    echo "[warn] (docker/deps) apt-get install failed; continuing without auto-install"
                                fi
                            elif command -v dnf >/dev/null 2>&1; then
                                if ! dnf install -y $missing_pkgs; then
                                    echo "[warn] (docker/deps) dnf install failed; continuing without auto-install"
                                fi
                            elif command -v yum >/dev/null 2>&1; then
                                if ! yum install -y $missing_pkgs; then
                                    echo "[warn] (docker/deps) yum install failed; continuing without auto-install"
                                fi
                            elif command -v apk >/dev/null 2>&1; then
                                if ! apk add --no-cache $missing_pkgs; then
                                    echo "[warn] (docker/deps) apk add failed; continuing without auto-install"
                                fi
                            else
                                echo "[warn] (docker/deps) no supported package manager found; skipping auto-install"
                            fi
                        fi
                    fi
                fi
                """
            ).strip()

        if artifacts_path or dep_setup:
            exec_cmd = " ".join(shlex.quote(a) for a in translated_for_exec)
            shell_parts: List[str] = ["set -u"]
            if artifacts_path:
                shell_parts.append(f"mkdir -p {shlex.quote(artifacts_path)}")
            if dep_setup:
                shell_parts.append(dep_setup)
            shell_parts.append(f"exec {exec_cmd}")
            translated_cmd = ["sh", "-lc", "\n".join(shell_parts)]

        docker_cmd.append(self.docker_image)
        docker_cmd += translated_cmd
        return docker_cmd

    # ────────────────────────────────────────────────────────────────────
    # Public entry
    # ────────────────────────────────────────────────────────────────────

    def generate(self) -> None:
        """
        Execute the end-to-end workflow.
        """
        print("[*] Pass A: Planning candidate fuzz targets …")
        self._pass_plan_targets()

        print("[*] Pass B: Synthesizing harness & local build glue …")
        self._pass_synthesize_harness()

        print("[*] Pass C: Building with retries …")
        self._build_with_retries()

        print("[*] Discovering new fuzzers …")
        bins = self._discover_fuzz_binaries()
        if not bins:
            raise HarnessGeneratorError("No fuzzer binaries found under fuzz/out/")

        crash_found = False
        last_fuzzer = ""
        last_artifact = ""
        run_rc = 0
        crash_evidence = "none"
        run_error_kind = ""
        for bin_path in bins:
            fuzzer_name = bin_path.name
            try:
                print(f"[*] Pass D: Generating initial seeds for {fuzzer_name} …")
                self._pass_generate_seeds(fuzzer_name)
            except HarnessGeneratorError as e:
                print(f"[!] Seed generation failed ({fuzzer_name}): {e}")

            print(f"[*] Pass E: Running {fuzzer_name} for ~{self.time_budget}s …")
            run = self._run_fuzzer(bin_path)
            run_rc = run.rc
            crash_evidence = run.crash_evidence
            run_error_kind = run.run_error_kind

            if run.error:
                raise HarnessGeneratorError(run.error)

            if run.crash_found and run.first_artifact:
                print(f"[!] Found {len(run.new_artifacts)} bug artifact(s), evidence={run.crash_evidence}.")
                first = Path(run.first_artifact)
                print(f"    → analyzing first: {first}")
                self._analyze_and_package(fuzzer_name, first)
                crash_found = True
                last_fuzzer = fuzzer_name
                last_artifact = str(first)
                # Stop after first validated crash to keep the demo tight.
                break
            else:
                print(f"[*] No artifacts produced by {fuzzer_name} in the time budget.")

        print("[*] Workflow complete.")
        self._write_run_summary(
            crash_found=crash_found,
            last_fuzzer=last_fuzzer,
            last_artifact=last_artifact,
            run_rc=run_rc,
            crash_evidence=crash_evidence,
            run_error_kind=run_error_kind,
        )

    # ────────────────────────────────────────────────────────────────────
    # Step A – Plan
    # ────────────────────────────────────────────────────────────────────

    def _pass_plan_targets(self, *, timeout: int = 1800) -> None:
        """
        Ask Codex to mine/score candidates and author PLAN.md + targets.json.
        """
        instructions = textwrap.dedent(
            f"""
            **Goal:** Analyze this repository and produce a realistic fuzz plan.
            **Deliverables (create inside `{FUZZ_DIR}/`):**
            1) `PLAN.md` — brief rationale describing the top 3–10 *public, attacker-reachable*
               entrypoints (file/packet/string parsers) with justification for real-world reachability,
               expected initialization, and any tricky preconditions.
            2) `targets.json` — JSON array of ranked candidates with fields:
               ```json
               [{{"name": "...",
                  "api": "qualified::symbol_or_Class.method",
                  "lang": "c-cpp|java",
                  "proto": "const uint8_t*,size_t|byte[]|InputStream",
                  "build_target": "cmake target or path if known",
                  "reason": "...",
                  "evidence": ["path:line", "..."]}}]
               ```
            3) Choose the single **best** candidate for a first harness and record its canonical
               fuzzer name (e.g., `xyz_format_fuzz`) at the top of `PLAN.md`.

            **Rules:**
            - Favor the highest-level API that ingests untrusted data (files/streams/packets).
            - Avoid low-level helpers (e.g., `_read_u32`) unless nothing higher validates input.
            - Prefer targets with small/clear init and good branch structure.
                        - Prefer the **lowest dependency footprint** target first: prioritize APIs that compile with
                            toolchain + repository-local code only (or standard runtime libs already present).
                        - If a candidate requires extra system/dev packages (e.g. new `apt`/`dnf` libraries),
                            rank it lower and choose an in-repo/low-dependency alternative when possible.
            - If compile_commands.json is needed, note it in `PLAN.md`, but do not generate it yet.

            **Do not run commands**; only write the files above and any small metadata you need.
            MANDATORY: when finished, you MUST write the path to `{FUZZ_DIR}/PLAN.md` into `./done`.
            If `./done` is missing, this step is treated as failed.
            """
        ).strip()

        stdout = self.patcher.run_codex_command(
            instructions,
            timeout=timeout,
            max_attempts=1,
            max_cli_retries=_workflow_opencode_cli_retries(),
        )
        if stdout is None:
            raise HarnessGeneratorError("Codex did not produce a plan (`fuzz/PLAN.md`).")

        print(f"[*] Codex planning done (truncated):\n{stdout[:900]}")

    # ────────────────────────────────────────────────────────────────────
    # Step B – Synthesize harness & build glue
    # ────────────────────────────────────────────────────────────────────

    def _pass_synthesize_harness(self, *, timeout: int = 1800) -> None:
        """
        Ask Codex to create a harness and local build system under fuzz/.
        """
        plan_md = self.fuzz_dir / "PLAN.md"
        targets_json = self.fuzz_dir / "targets.json"
        plan_text = read_text_safely(plan_md)
        targets_text = read_text_safely(targets_json)

        instructions = textwrap.dedent(
            f"""
            **Goal:** Create a *local* fuzzing scaffold for the chosen top target from `PLAN.md`.

            **Requirements (create under `{FUZZ_DIR}/`):**
            - One harness:
              - **C/C++**: `<name>_fuzz.cc` implementing:
                ```c++
                extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {{
                    // minimal realistic init; call the public API; no stubs; no UB
                }}
                ```
              - **Java**: `<Name>Fuzzer.java` compatible with **Jazzer**.
                        - **`build.py`**:
                            - Cross-platform, non-interactive build script runnable as `(cd fuzz && python build.py)`.
                            - Resolve all paths from `Path(__file__).resolve()` so it works regardless of caller cwd.
                            - Detect common build systems (CMake/Meson/Autotools/Make) and do the minimal work to build the library and the fuzzer.
                            - For C/C++: prefer **clang/clang++** and produce a libFuzzer-style binary when possible.
                            - Emit fuzzer binaries into `{FUZZ_OUT_DIR}/`.
                            - For Java: fetch/setup **Jazzer** locally and emit runnable target(s) into `{FUZZ_OUT_DIR}/`.
            - **.options** (libFuzzer) near each binary if helpful (e.g., `-max_len={self.max_len}`).
            - **README.md** explaining the entrypoint and how to run the fuzzer.
            - Ensure seeds will be looked up from `{FUZZ_CORPUS_DIR}/<fuzzer_name>/`.
                        - If external system packages are strictly required, create `{FUZZ_SYSTEM_PACKAGES_FILE}`
                            with one package name per line (comments with `#` are allowed, no shell commands).

            **Critical constraints:**
            - Use **public/documented APIs**; avoid low-level helpers.
            - Perform **minimal real-world init** (contexts/handles via proper constructors).
            - Avoid harness mistakes (double-free, wrong types, lifetime bugs).
                        - Keep dependency footprint minimal: prefer targets/build paths that require no new
                            external system packages beyond the existing image/toolchain.
                        - Do not introduce new third-party library dependencies just to make a harness compile;
                            if the selected target needs unavailable deps, choose a lower-dependency target instead.
            - Do not vendor large third-party code; use the repo as-is.
            - Prefer `compile_commands.json` if available; otherwise add just enough build glue in `build.py`.

            **Acceptance criteria:**
            - After `(cd {FUZZ_DIR} && python build.py)`, at least one fuzzer binary must exist in `{FUZZ_OUT_DIR}/`.
            - The harness compiles with symbols; ASan/UBSan enabled for C/C++.
            - The harness reaches some code with a trivial input (will be tested soon).

            Do not run any commands here; only create/modify files.
            MANDATORY: when finished, you MUST write `{FUZZ_OUT_DIR}` into `./done`.
            If `./done` is missing, this step is treated as failed.
            """
        ).strip()

        context = (
            "=== fuzz/PLAN.md ===\n" + plan_text +
            "\n\n=== fuzz/targets.json ===\n" + targets_text
        )
        stdout = self.patcher.run_codex_command(
            instructions,
            additional_context=context,
            timeout=timeout,
            max_attempts=1,
            max_cli_retries=_workflow_opencode_cli_retries(),
        )
        if stdout is None:
            raise HarnessGeneratorError("Codex did not create harness/build scaffold under fuzz/.")

        print(f"[*] Codex synthesis done (truncated):\n{stdout[:900]}")

    # ────────────────────────────────────────────────────────────────────
    # Step C – Build with retries (feedback to Codex)
    # ────────────────────────────────────────────────────────────────────

    def _build_with_retries(self) -> None:
        build_py = self.fuzz_dir / "build.py"
        build_sh = self.fuzz_dir / "build.sh"
        build_dir = self.repo_root / "build"
        fuzz_build_dir = self.repo_root / "fuzz" / "build"

        def _build_py_supports_clean_flag(path: Path) -> bool:
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return False
            # Best-effort heuristic. We intentionally keep this permissive and cheap.
            return "--clean" in txt

        def _list_static_libs_for_diagnostics() -> str:
            """Return a short listing of built static libraries under build/.

            This is intentionally concise (helps the agent fix path/name assumptions).
            """

            if not build_dir.exists():
                return f"(no build dir at {build_dir})"

            if self.docker_image:
                # Keep output short and deterministic.
                bash_script = (
                    "set -e; "
                    "if [ -d /work/build ]; then "
                    "(find /work/build -maxdepth 4 -type f \\( "
                    "-name '*.a' -o -name '*.lib' -o -name '*.so' -o -name '*.dylib' \\) "
                    "-printf '%p (%s bytes)\\n' 2>/dev/null || true) | head -n 80; "
                    "else echo '(no /work/build dir)'; fi"
                )
                cmd = ["bash", "-lc", bash_script]
                rc, out, err = self._run_cmd(cmd, cwd=self.repo_root, timeout=120)
                blob = (out or "") + ("\n" + err if err else "")
                blob = strip_ansi(blob).strip()
                return blob if blob else "(no static libs found or listing empty)"

            # Host mode
            try:
                libs: List[str] = []
                for p in build_dir.rglob("*"):
                    if not p.is_file():
                        continue
                    if p.suffix.lower() in {".a", ".lib", ".so", ".dylib"}:
                        try:
                            libs.append(f"{p.relative_to(self.repo_root)} ({p.stat().st_size} bytes)")
                        except Exception:
                            libs.append(str(p.relative_to(self.repo_root)))
                    if len(libs) >= 80:
                        break
                return "\n".join(libs) if libs else "(no static libs found under build/)"
            except Exception as e:
                return f"(failed to list libs under build/: {e})"

        build_cwd = self.fuzz_dir
        fallback_cmd: Optional[List[str]] = None
        if build_py.is_file():
            build_cmd = [self._python_runner(), "build.py"]
            fallback_cmd = [self._python_runner(), f"{FUZZ_DIR}/build.py"]
            build_cmd_clean: Optional[List[str]] = None
            if _build_py_supports_clean_flag(build_py):
                build_cmd_clean = list(build_cmd) + ["--clean"]
        elif build_sh.is_file():
            # Backwards compatibility (older harness scaffolds).
            build_cmd = ["bash", "build.sh"]
            fallback_cmd = ["bash", f"{FUZZ_DIR}/build.sh"]
            make_executable(build_sh)
            build_cmd_clean = None
        else:
            raise HarnessGeneratorError(
                f"Neither {build_py} nor {build_sh} was found (agent must create fuzz/build.py)."
            )

        errors_accum = ""
        build_env = os.environ.copy()
        if self.docker_image:
            # Ensure libFuzzer-compatible toolchain when running inside Docker.
            build_env.setdefault("CC", "clang")
            build_env.setdefault("CXX", "clang++")
            # Define _GNU_SOURCE for projects that use POSIX extensions (e.g., lseek)
            build_env.setdefault("CFLAGS", "-D_GNU_SOURCE")
            build_env.setdefault("CXXFLAGS", "-D_GNU_SOURCE")
            # Avoid stale compiler choices in cached CMake dirs.
            for stale_dir in (fuzz_build_dir, build_dir):
                if stale_dir.exists():
                    try:
                        shutil.rmtree(stale_dir)
                    except Exception:
                        pass

        def _repo_has_c_cpp_main() -> bool:
            exts = {".c", ".cc", ".cpp", ".cxx"}
            try:
                checked = 0
                for p in self.repo_root.rglob("*"):
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
            replaced = text.replace(
                " + [harness_cpp, VULNERABLE_CPP] + ",
                f" + ['{define_flag}', harness_cpp, VULNERABLE_CPP] + ",
            )
            if replaced != text:
                return replaced, True
            return text, False

        def _try_hotfix_stdlib_mismatch_and_main_conflict(diag_text: str) -> bool:
            if not build_py.is_file():
                return False

            low = (diag_text or "").lower()
            abi_mismatch = any(
                token in low
                for token in [
                    "undefined reference to `std::__cxx11",
                    "undefined reference to `std::",
                    "vtable for std::",
                    "libclang_rt.fuzzer",
                ]
            )
            multiple_main = ("multiple definition of `main'" in low) or ("multiple definition of main" in low)

            try:
                text = build_py.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return False

            has_libcpp_flag = "-stdlib=libc++" in text
            if not (abi_mismatch or has_libcpp_flag or multiple_main):
                return False

            changed = False
            if has_libcpp_flag:
                text2 = re.sub(r'^[ \t]*["\']-stdlib=libc\+\+["\'],?\s*\n?', "", text, flags=re.MULTILINE)
                text2 = text2.replace('"-stdlib=libc++",', "").replace('"-stdlib=libc++"', "")
                text2 = text2.replace("'-stdlib=libc++',", "").replace("'-stdlib=libc++'", "")
                if text2 != text:
                    text = text2
                    changed = True

            need_main_rename = multiple_main or _repo_has_c_cpp_main()
            if need_main_rename and "-Dmain=vuln_main" not in text:
                text, injected = _inject_define_into_flag_list(text, "-Dmain=vuln_main")
                changed = changed or injected

            if not changed:
                return False

            try:
                build_py.write_text(text, encoding="utf-8", errors="replace")
                print("[*] Applied local hotfix for stdlib mismatch/main conflict in fuzz/build.py")
                return True
            except Exception:
                return False

        def _try_hotfix_libfuzzer_main_conflict(diag_text: str) -> bool:
            if not build_py.is_file():
                return False
            low = (diag_text or "").lower()
            if "multiple definition of `main'" not in low and "multiple definition of main" not in low:
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
                if replaced == text:
                    return False
                text = replaced
            else:
                text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

            try:
                build_py.write_text(text, encoding="utf-8", errors="replace")
                print("[*] Applied local hotfix for libFuzzer main conflict in fuzz/build.py")
                return True
            except Exception:
                return False

        for attempt in range(1, self.max_build_retries + 1):
            print(f"[*] Build attempt {attempt}/{self.max_build_retries} → {' '.join(build_cmd)}")

            rc, out, err = self._run_cmd(list(build_cmd), cwd=build_cwd, env=build_env)

            combined_first = strip_ansi((out or "") + "\n" + (err or ""))
            combined_first_l = combined_first.lower()
            if (
                rc != 0
                and fallback_cmd is not None
                and (
                    ("no such file or directory" in combined_first_l and "fuzz/" in combined_first_l)
                    or "can't open file '/work/fuzz/fuzz/" in combined_first_l
                    or "can't open file 'fuzz/" in combined_first_l
                )
            ):
                print("[*] Build appears to require repo-root cwd; retrying from repo root")
                rc, out, err = self._run_cmd(list(fallback_cmd), cwd=self.repo_root, env=build_env)

            if rc != 0 and _try_hotfix_stdlib_mismatch_and_main_conflict(strip_ansi((out or "") + "\n" + (err or ""))):
                print("[*] Retrying build after applying local stdlib/main-conflict hotfix")
                continue

            if rc != 0 and _try_hotfix_libfuzzer_main_conflict(strip_ansi((out or "") + "\n" + (err or ""))):
                print("[*] Retrying build after applying local main-conflict hotfix")
                continue

            # Optional retry-with-clean for flaky/stale CMake caches.
            if rc != 0 and build_cmd_clean is not None:
                combined = strip_ansi((out or "") + "\n" + (err or ""))
                # Avoid looping: only retry clean once per attempt.
                if not re.search(r"unrecognized arguments: --clean", combined, re.IGNORECASE):
                    print(f"[*] Build failed; retrying once with --clean → {' '.join(build_cmd_clean)}")
                    rc2, out2, err2 = self._run_cmd(list(build_cmd_clean), cwd=build_cwd, env=build_env)
                else:
                    rc2, out2, err2 = rc, out, err
                # If --clean itself is unsupported, keep original rc/out/err.
                combined2 = strip_ansi((out2 or "") + "\n" + (err2 or ""))
                if re.search(r"unrecognized arguments: --clean", combined2, re.IGNORECASE):
                    print("[warn] build.py does not support --clean; continuing without it")
                else:
                    rc, out, err = rc2, out2, err2

            # Detect two categories of issues:
            #   1. The build script exited with non-zero status (classic compilation failure).
            #   2. The script exited cleanly (rc==0) **but did not emit any fuzzer binaries**
            #      under fuzz/out/.  The latter is surprisingly common when build.sh only
            #      compiles auxiliary objects or writes *.options files.

            binaries = self._discover_fuzz_binaries() if rc == 0 else []

            if rc == 0 and binaries:
                print(f"[*] Build succeeded. Discovered {len(binaries)} fuzzer binary(ies).")
                return

            # Prepare diagnostics for Codex – prefer stderr when non-zero rc, otherwise stdout.
            diag = err if rc != 0 else out
            libs_diag = _list_static_libs_for_diagnostics()
            if libs_diag:
                diag = (
                    (diag or "")
                    + "\n\n=== build dir artifacts (static libs) ===\n"
                    + libs_diag
                    + "\n"
                )

            print(
                "[!] Build produced no runnable fuzzers." if rc == 0 else f"[!] Build failed (rc={rc}).",
                "Sending diagnostics back to Codex …",
            )

            errors_accum = (errors_accum + "\n\n" + diag)[-20000:]  # keep last 20k

            fix_prompt = textwrap.dedent(
                """
                The *fuzz* build is still incorrect:
                {problem}

                Read the diagnostics below and apply the **minimal** edits necessary so that running
                `(cd fuzz && python build.py)` completes successfully **and** leaves at least one executable
                fuzzer binary in `fuzz/out/` (files ending with `fuzz`, `_fuzzer`, or `Fuzzer`).

                Do not refactor production code or add features; only fix the build glue or harness.
                Prefer the **minimal-dependency** solution: avoid adding new external/system package
                requirements. If the current harness target requires unavailable third-party deps,
                retarget to an existing low-dependency API in this repo instead of introducing new deps.
                If external system packages are truly necessary, update `{FUZZ_SYSTEM_PACKAGES_FILE}`
                with package names only (one per line, comments allowed, no shell syntax/commands).
                Modify files under `fuzz/` and the minimal build files elsewhere. Do **not** run the
                build yourself; just output patches. Keep emitting binaries to `fuzz/out/`.

                When done, write `fuzz/build.py` into `./done`.
                """
            ).strip().format(problem=("Build finished with rc=0 but no binaries found" if rc == 0 else f"Non-zero exit code {rc}"))

            stdout = self.patcher.run_codex_command(fix_prompt, additional_context=errors_accum)

            if stdout is None and attempt == self.max_build_retries:
                raise HarnessGeneratorError("Codex failed to resolve build errors after retries.")

        # final build try
        rc, out, err = self._run_cmd(list(build_cmd), cwd=build_cwd, env=build_env)
        combined_final = strip_ansi((out or "") + "\n" + (err or ""))
        combined_final_l = combined_final.lower()
        if (
            rc != 0
            and fallback_cmd is not None
            and (
                ("no such file or directory" in combined_final_l and "fuzz/" in combined_final_l)
                or "can't open file '/work/fuzz/fuzz/" in combined_final_l
                or "can't open file 'fuzz/" in combined_final_l
            )
        ):
            rc, out, err = self._run_cmd(list(fallback_cmd), cwd=self.repo_root, env=build_env)
        if rc != 0:
            raise HarnessGeneratorError("Build still failing after Codex retries.")

        # Build script exited cleanly on final attempt; ensure it produced binaries.
        if not self._discover_fuzz_binaries():
            raise HarnessGeneratorError(
                "Build completed after retries but no fuzzer binaries were found in fuzz/out/."
            )

    # ────────────────────────────────────────────────────────────────────
    # Step D – Generate initial seeds
    # ────────────────────────────────────────────────────────────────────

    def _pass_generate_seeds(self, fuzzer_name: str) -> None:
        harness_src = self._locate_harness_source_for(fuzzer_name)
        harness_text = read_text_safely(harness_src) if harness_src else ""
        corpus_dir = self.fuzz_corpus_dir / fuzzer_name
        corpus_dir.mkdir(parents=True, exist_ok=True)

        instructions = textwrap.dedent(
            f"""
            Create 1–5 **meaningful seed files** inside `{corpus_dir.relative_to(self.repo_root)}` for the
            new harness `{fuzzer_name}`. Prefer small, realistic inputs that exercise typical paths. Use
            appropriate file extensions if known. If binary, you may write contents via hex bytes.

            Only create seed files (no code changes). When finished, write the path to one seed file into `./done`.
            """
        ).strip()

        seed_timeout = getattr(self, "seed_generation_timeout_sec", None)
        patcher_kwargs: Dict[str, object] = {}
        if isinstance(seed_timeout, int) and seed_timeout > 0:
            patcher_kwargs["timeout"] = seed_timeout
        stdout = self.patcher.run_codex_command(
            instructions,
            additional_context=harness_text or "(no harness found)",
            **patcher_kwargs,
        )
        if stdout is None:
            raise HarnessGeneratorError("Codex did not generate any seed files.")
        print(f"[*] Codex seed creation done (truncated):\n{stdout[:600]}")

    # ────────────────────────────────────────────────────────────────────
    # Step E – Run fuzzer
    # ────────────────────────────────────────────────────────────────────

    def _run_fuzzer(self, bin_path: Path) -> FuzzerRunResult:
        """
        Run a single local fuzzer binary with sane defaults.
        Returns a structured result including artifacts and crash evidence.
        """
        bin_dir = bin_path.parent
        artifacts_dir = bin_dir / ARTIFACT_PREFIX
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.setdefault("ASAN_OPTIONS", "exitcode=76:detect_leaks=0")
        env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1")
        env.setdefault("LLVM_SYMBOLIZER_PATH", which("llvm-symbolizer") or "")

        corpus_dir = self.fuzz_corpus_dir / bin_path.name
        corpus_dir.mkdir(parents=True, exist_ok=True)

        pre_existing = set(p for p in artifacts_dir.glob("*") if p.is_file())

        run_time_budget_raw = getattr(self, "current_run_time_budget_sec", self.time_budget)
        if run_time_budget_raw is None:
            run_time_budget_raw = self.time_budget
        run_time_budget = int(run_time_budget_raw)
        hard_timeout = int(getattr(self, "current_run_hard_timeout_sec", 0) or 0)
        if hard_timeout <= 0 and run_time_budget > 0:
            hard_timeout = max(60, run_time_budget + 120)
        run_idle_timeout_raw = os.environ.get("SHERPA_RUN_IDLE_TIMEOUT_SEC", "120")
        try:
            run_idle_timeout = max(0, min(int(str(run_idle_timeout_raw).strip()), 86400))
        except Exception:
            run_idle_timeout = 120

        cmd = [
            str(bin_path),
            "-artifact_prefix=" + str(artifacts_dir) + "/",
            "-print_final_stats=1",
            f"-max_len={self.max_len}",
        ]
        if run_time_budget > 0:
            cmd.append(f"-max_total_time={run_time_budget}")

        print(f"[*] ➜  {' '.join(cmd)}")
        rc, out, err = self._run_cmd(
            cmd,
            cwd=self.repo_root,
            env=env,
            extra_inputs=[str(corpus_dir)],
            timeout=hard_timeout,
            idle_timeout=run_idle_timeout,
        )

        # Dump the tail for quick reading.
        log = (out + "\n=== STDERR ===\n" + err).replace("\r", "\n")
        tail = "\n".join(log.splitlines()[-200:])
        print(tail)

        libfuzzer_stats = parse_libfuzzer_final_stats(log)

        # Detect new artifacts
        post = set(p for p in artifacts_dir.glob("*") if p.is_file())
        new_artifacts = sorted(post - pre_existing)
        if new_artifacts:
            print("[*] New artifact(s):")
            for p in new_artifacts:
                print("    •", p.relative_to(self.repo_root))
        crash_evidence = "none"
        first_artifact = ""
        timeout_artifact_count = 0

        if new_artifacts:
            sorted_artifacts = sorted(new_artifacts)
            timeout_like_artifacts = [
                p for p in sorted_artifacts if p.name.startswith(("timeout-", "slow-unit-"))
            ]
            timeout_artifact_count = len(timeout_like_artifacts)
            crash_like_artifacts = [p for p in sorted_artifacts if p not in timeout_like_artifacts]

            if crash_like_artifacts:
                crash_evidence = "artifact"
                first_artifact = str(crash_like_artifacts[0])
            elif timeout_like_artifacts:
                # timeout-/slow-unit-* are performance/hang signals, not stable crash proof.
                crash_evidence = "timeout_artifact"
                first_artifact = str(timeout_like_artifacts[0])

        def _is_sanitizer_crash(text: str) -> bool:
            if not text:
                return False
            if re.search(r"AddressSanitizer failed to allocate", text, re.IGNORECASE):
                return False
            if re.search(r"ReserveShadowMemoryRange failed", text, re.IGNORECASE):
                return False
            if re.search(r"failed to mmap", text, re.IGNORECASE):
                return False
            if re.search(r"==[0-9]+==ERROR: (Address|Undefined|Memory|Thread|Leak)Sanitizer", text):
                return True
            if re.search(r"SUMMARY: (AddressSanitizer|UndefinedBehaviorSanitizer|MemorySanitizer|ThreadSanitizer)", text):
                return True
            if re.search(r"\bruntime error:\b", text, re.IGNORECASE):
                return True
            if re.search(r"ERROR: libFuzzer: deadly signal", text):
                return True
            return False

        if crash_evidence == "none" and _is_sanitizer_crash(log):
            crash_evidence = "sanitizer_log"
            synthetic = artifacts_dir / f"crash-log-{int(time.time())}.txt"
            write_text_safely(synthetic, strip_ansi(log))
            new_artifacts = [synthetic]
            first_artifact = str(synthetic)
            print(f"[*] Sanitizer crash detected without artifact, synthesized evidence: {synthetic.relative_to(self.repo_root)}")

        crash_found = crash_evidence in {"artifact", "sanitizer_log"}
        error = ""
        run_error_kind = ""
        if crash_evidence == "timeout_artifact":
            run_error_kind = "run_timeout"
            error = (
                f"fuzzer produced timeout-like artifacts for {bin_path.name} "
                f"(count={timeout_artifact_count})"
            )
        if rc != 0 and not crash_found:
            lowered = log.lower()
            if "idle-timeout" in lowered:
                run_error_kind = "run_idle_timeout"
                error = (
                    f"fuzzer run idle-timeout for {bin_path.name}: "
                    f"no output for {run_idle_timeout}s"
                )
            elif "[timeout]" in lowered:
                if not run_error_kind:
                    run_error_kind = "run_timeout"
                if not error:
                    error = f"fuzzer run timed out for {bin_path.name}"
            else:
                if not run_error_kind:
                    run_error_kind = "nonzero_exit_without_crash"
                if not error:
                    error = f"fuzzer run failed rc={rc} for {bin_path.name}; no crash artifact/sanitizer evidence found"

        corpus_files = 0
        corpus_size_bytes = 0
        try:
            for p in corpus_dir.rglob("*"):
                if p.is_file():
                    corpus_files += 1
                    try:
                        corpus_size_bytes += int(p.stat().st_size)
                    except Exception:
                        pass
        except Exception:
            pass

        return FuzzerRunResult(
            rc=int(rc),
            new_artifacts=list(new_artifacts),
            crash_found=crash_found,
            crash_evidence=crash_evidence,
            first_artifact=first_artifact,
            log_tail=tail,
            error=error,
            run_error_kind=run_error_kind,
            final_cov=int(libfuzzer_stats.get("cov", 0)),
            final_ft=int(libfuzzer_stats.get("ft", 0)),
            final_corpus_files=int(libfuzzer_stats.get("corpus_files", 0)),
            final_corpus_size_bytes=int(libfuzzer_stats.get("corpus_size_bytes", 0)),
            final_execs_per_sec=int(libfuzzer_stats.get("execs_per_sec", 0)),
            final_rss_mb=int(libfuzzer_stats.get("rss_mb", 0)),
            final_iteration=int(libfuzzer_stats.get("iteration", 0)),
            corpus_files=corpus_files,
            corpus_size_bytes=corpus_size_bytes,
        )

    # ────────────────────────────────────────────────────────────────────
    # Step F – Analyze & package
    # ────────────────────────────────────────────────────────────────────

    def _analyze_and_package(self, fuzzer_name: str, artifact_path: Path) -> None:
        """
        Produce crash_info.md, crash_analysis.md, and a local reproducer script,
        then bundle everything into challenge_bundle/.
        """
        # 1) Reproducer: run binary with the artifact as sole input.
        bin_path = self.fuzz_out_dir / fuzzer_name
        if not bin_path.exists():
            # fallback: search
            bins = self._discover_fuzz_binaries()
            for b in bins:
                if b.name == fuzzer_name:
                    bin_path = b
                    break

        repro_cmd = f"{bin_path} -runs=1 {artifact_path}"
        print(f"[*] Reproducing with: {repro_cmd}")

        rc, out, err = self._run_cmd(
            [str(bin_path), "-runs=1", str(artifact_path)],
            cwd=self.repo_root,
            env=os.environ.copy(),
        )
        combined = strip_ansi(out + ("\n=== STDERR ===\n" + err if err else ""))

        # 2) crash_info.md
        harness_src = self._locate_harness_source_for(fuzzer_name)
        harness_text = read_text_safely(harness_src) if harness_src else "*not found*"
        hd = hexdump(artifact_path)
        info_md = [
            "# Crash Info",
            "",
            "## Reproducer command",
            "```bash",
            repro_cmd,
            "```",
            "",
            "## Reproducer output",
            "```text",
            combined,
            "```",
            "",
            "## Harness Source",
            "```c",
            harness_text.replace("```", "```​"),  # guard
            "```",
            "",
            "## Crashing input (hexdump)",
            "```text",
            hd,
            "```",
            "",
        ]
        write_text_safely(self.repo_root / "crash_info.md", "\n".join(info_md))
        print("[*] crash_info.md written.")

        # 3) Ask Codex for crash_analysis.md
        context_blob = (
            "=== crash_info.md ===\n" + (self.repo_root / "crash_info.md").read_text(encoding="utf-8", errors="replace")
        )
        analysis_prompt = textwrap.dedent(
            """
            You are an experienced security researcher.

            Using the context provided, write `crash_analysis.md` with sections:
            1. Bug Type
            2. Bug Summary
            3. Bug Impact (real-world reachability / exploitability / constraints)
            4. How to Patch

            Notes:
              • If evidence suggests a harness error (misuse of the API, bad args, UB in harness),
                explicitly mark **HARNESS ERROR** and set severity to None.
              • Otherwise, be concise but specific; include the likely root cause and patch guidance.
            """
        ).strip()
        stdout = self.patcher.run_codex_command(analysis_prompt, additional_context=context_blob)
        if stdout is None:
            print("[!] Codex did not produce crash_analysis.md")

        # 4) Ask Codex to create a minimal reproducer script (local env, not OSS-Fuzz)
        info = read_text_safely(self.repo_root / "crash_info.md")
        analysis = read_text_safely(self.repo_root / "crash_analysis.md")
        reproducer_ctx = "=== crash_info.md ===\n" + info + "\n\n=== crash_analysis.md ===\n" + analysis

        reproduce_prompt = textwrap.dedent(
                        """
                        Create `reproduce.py` in repo root that:
                            • Assumes the fuzzer binary has already been built and placed in `fuzz/out/`.
                            • Locates the first fuzzer executable in `fuzz/out/` (also consider `*.exe` on Windows).
                            • Runs the fuzzer with the minimized crashing input to demonstrate the issue.
                            • Wrap the invocation in an external timeout using Python's subprocess timeout, so hangs terminate.
                            • Exit non-zero on crash/timeout; otherwise zero.

                        Requirements:
                            • Must run on native Windows (no bash/coreutils/ulimit).
                            • Use only the Python standard library.

                        Only create `reproduce.py`. Do not modify other files.
                        """
                ).strip()

        stdout = self.patcher.run_codex_command(reproduce_prompt, additional_context=reproducer_ctx)
        if stdout is None:
            print("[!] Agent did not produce reproduce.py")

        # 4b) Validate that the reproducer actually triggers the crash. If it does not or
        # if it fails prematurely with an AddressSanitizer shadow-memory error (common
        # when the reproducer forgets to limit memory or uses an incorrect binary), we
        # feed the diagnostics back into Codex and request fixes – up to a small number
        # of iterations. This keeps the developer experience tight: the first challenge
        # bundle the user opens will "just work" instead of requiring manual tweaks.

        reproducer_ok = self._ensure_working_reproducer(max_retries=3)

        # 5) If not harness error, ask Codex for a comprehensive justification that
        # confirms this is a *true* positive finding and not an artifact of the
        # test harness.  The resulting markdown is saved as
        # `true_positive_justification.md`.

        justification_path = self.repo_root / "true_positive_justification.md"
        analysis_path = self.repo_root / "crash_analysis.md"
        if not (analysis_path.exists() and re.search(r"HARNESS ERROR", analysis_path.read_text(encoding="utf-8", errors="ignore"), re.IGNORECASE)):
            justification_prompt = textwrap.dedent(
                """
                Using crash_info.md and crash_analysis.md, write `true_positive_justification.md`.

                The document must persuade a skeptical reviewer that this is a **genuine
                vulnerability in the upstream project**, not a bug in the fuzzer harness.

                Required sections:
                  1. Why the crash is not caused by the harness (with concrete evidence).
                  2. Root cause summary (concise, technical).
                  3. Real-world reachability scenario.
                  4. Potential impact / exploitability.
                  5. Suggested fix direction in upstream code.

                Keep it under ~400 words, clear and professional.
                """
            ).strip()

            ctx = "=== crash_info.md ===\n" + read_text_safely(self.repo_root / "crash_info.md") + "\n\n=== crash_analysis.md ===\n" + read_text_safely(analysis_path)
            self.patcher.run_codex_command(justification_prompt, additional_context=ctx)

        files_to_copy = [
            "crash_info.md",
            "crash_analysis.md",
            "true_positive_justification.md",
            "reproduce.py",
            "fix.patch",
            "fix_summary.md",
            "run_summary.md",
            "run_summary.json",
        ]

        # 6) Package challenge bundle
        bundle_name = "challenge_bundle" if self.round_index == 1 else f"challenge_bundle_{self.round_index}"
        bundle = self.repo_root / bundle_name

        def _prepare_bundle_dir(dst: Path) -> None:
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink(missing_ok=True)
            dst.mkdir(parents=True, exist_ok=True)

            for rel in files_to_copy:
                src = self.repo_root / rel
                if src.is_file():
                    shutil.copy2(src, dst / src.name)

            # Always copy fuzz/ directory after individual files (may contain reproduce deps)
            rel_dir = FUZZ_DIR
            src_dir = self.repo_root / rel_dir
            if src_dir.is_dir():
                out_dir = dst / src_dir.name
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                shutil.copytree(src_dir, out_dir)

        def _move_bundle_to(dst: Path) -> None:
            # LLM fix/justification steps may unexpectedly mutate the workspace.
            # Rebuild bundle if missing, then move; fallback to copy+remove if rename fails.
            if not bundle.is_dir():
                print(f"[warn] bundle missing before move ({bundle}); rebuilding...")
                _prepare_bundle_dir(bundle)
            if not bundle.is_dir():
                raise HarnessGeneratorError(f"bundle directory missing before move: {bundle}")

            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink(missing_ok=True)

            try:
                bundle.rename(dst)
                return
            except Exception as rename_err:
                print(f"[warn] bundle rename failed ({rename_err}); falling back to copytree")
                shutil.copytree(bundle, dst)
                shutil.rmtree(bundle, ignore_errors=True)

        _prepare_bundle_dir(bundle)
        # Final classification policy (single-shot, avoid double-renames):
        # 1) HARNESS ERROR => false_positive
        # 2) reproducer failed => unreproducible
        # 3) otherwise keep challenge_bundle
        harness_error = False
        analysis_text = ""
        if analysis_path.exists():
            analysis_text = analysis_path.read_text(encoding="utf-8", errors="ignore")
            harness_error = bool(re.search(r"HARNESS ERROR", analysis_text, re.IGNORECASE))

        if harness_error:
            # Generate a concise justification explaining why this is a harness error.
            justification_path = self.repo_root / "false_positive_justification.md"
            fp_prompt = textwrap.dedent(
                """
                Write `false_positive_justification.md`.

                Explain in under 300 words why the observed crash/timeout is
                attributable to a misuse or bug in the fuzzing harness rather
                than a flaw in the target project.  Summarise what the harness
                does wrong (e.g., incorrect API usage, invalid parameters, not
                respecting preconditions) and how this leads to the detected
                fault.  Provide guidance on how to fix the harness so the bug
                disappears.
                """
            ).strip()
            self.patcher.run_codex_command(fp_prompt, additional_context=analysis_text)

            false_pos_dir = self.repo_root / ("false_positive" if self.round_index == 1 else f"false_positive_{self.round_index}")
            _move_bundle_to(false_pos_dir)

            if justification_path.exists():
                shutil.copy2(justification_path, false_pos_dir / justification_path.name)

            print("[!] Crash determined to be caused by harness error → recorded as false_positive.")
            return

        if not reproducer_ok:
            unrepro = self.repo_root / ("unreproducible" if self.round_index == 1 else f"unreproducible_{self.round_index}")
            _move_bundle_to(unrepro)
            print("[!] Could not reliably reproduce the crash → recorded under unreproducible/.")
        else:
            print(f"[*] Challenge bundle ready → {bundle.relative_to(self.repo_root)}")

    def _write_run_summary(
        self,
        *,
        crash_found: bool,
        last_fuzzer: str = "",
        last_artifact: str = "",
        error: str | None = None,
        run_rc: int | None = None,
        crash_evidence: str = "none",
        run_error_kind: str = "",
    ) -> None:
        summary_json = self.repo_root / "run_summary.json"
        summary_md = self.repo_root / "run_summary.md"
        analysis_path = self.repo_root / "crash_analysis.md"
        harness_error = False
        if analysis_path.is_file():
            try:
                text = analysis_path.read_text(encoding="utf-8", errors="ignore")
                harness_error = bool(re.search(r"HARNESS ERROR", text, re.IGNORECASE))
            except Exception:
                harness_error = False

        bundle_dirs = [
            d.name
            for d in self.repo_root.iterdir()
            if d.is_dir() and d.name.startswith(("challenge_bundle", "false_positive", "unreproducible"))
        ]

        payload = {
            "repo_root": str(self.repo_root),
            "status": "error" if error else ("crash_found" if crash_found else "ok"),
            "time_budget": self.time_budget,
            "max_len": self.max_len,
            "docker_image": self.docker_image,
            "run_rc": run_rc,
            "crash_evidence": crash_evidence,
            "run_error_kind": run_error_kind,
            "crash_found": crash_found,
            "last_fuzzer": last_fuzzer,
            "last_crash_artifact": last_artifact,
            "harness_error": harness_error,
            "error": error or "",
            "crash_info_path": str(self.repo_root / "crash_info.md"),
            "crash_analysis_path": str(self.repo_root / "crash_analysis.md"),
            "reproducer_path": str(self.repo_root / "reproduce.py"),
            "bundles": bundle_dirs,
            "timestamp": time.time(),
        }

        try:
            summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

        md_lines = [
            "# Run Summary",
            "",
            f"- Status: {payload['status']}",
            f"- Repo root: {payload['repo_root']}",
            f"- Time budget: {payload['time_budget']}s",
            f"- Run rc: {payload['run_rc']}",
            f"- Crash evidence: {payload['crash_evidence']}",
            f"- Crash found: {payload['crash_found']}",
            f"- Harness error: {payload['harness_error']}",
        ]
        if error:
            md_lines.extend(["", "## Error", "```text", error, "```"])
        if crash_found:
            md_lines.extend(
                [
                    "",
                    "## Crash",
                    f"- Fuzzer: {last_fuzzer}",
                    f"- Artifact: {last_artifact}",
                    f"- crash_info.md: {payload['crash_info_path']}",
                    f"- crash_analysis.md: {payload['crash_analysis_path']}",
                ]
            )
        if bundle_dirs:
            md_lines.extend(["", "## Bundles"] + [f"- {b}" for b in bundle_dirs])
        try:
            summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    # ────────────────────────────────────────────────────────────────────
    # Discovery & helpers
    # ────────────────────────────────────────────────────────────────────

    def _clone_repo(self, spec: RepoSpec) -> Path:
        def _clone_with_host_git(dest: Path) -> Path:
            dest_parent = dest.parent
            dest_parent.mkdir(parents=True, exist_ok=True)

            last_rc: Optional[int] = None
            attempted: List[str] = []
            for clone_url in _candidate_clone_urls(spec.url):
                attempted.append(clone_url)
                try:
                    if dest.exists():
                        shutil.rmtree(dest, ignore_errors=True)
                except Exception:
                    pass

                print(f"[*] (host/git) Cloning {clone_url} → {dest}")
                proxy_overrides = _host_git_proxy_override_args()
                if proxy_overrides:
                    print("[warn] (host/git) detected broken localhost proxy; disabling git http(s).proxy for this operation")
                clone_cmd = ["git", *proxy_overrides, "clone", "--depth", "1", clone_url, str(dest)]
                print(f"[*] ➜  {' '.join(clone_cmd)}")
                rc, out, err, timed_out = _run_cmd_capture(clone_cmd, timeout=GIT_HOST_CLONE_TIMEOUT_SEC)
                last_rc = rc
                if timed_out:
                    if (t := _tail_lines(err)):
                        print("[warn] (host/git) clone stderr (tail):\n" + textwrap.indent(t, "    "))
                    if (t := _tail_lines(out)):
                        print("[warn] (host/git) clone stdout (tail):\n" + textwrap.indent(t, "    "))
                    print(
                        f"[warn] (host/git) clone timed out after {GIT_HOST_CLONE_TIMEOUT_SEC}s (url={clone_url}); retrying next URL..."
                    )
                    continue
                if rc == 0:
                    break
                if (t := _tail_lines(err)):
                    print("[warn] (host/git) clone stderr (tail):\n" + textwrap.indent(t, "    "))
                if (t := _tail_lines(out)):
                    print("[warn] (host/git) clone stdout (tail):\n" + textwrap.indent(t, "    "))
            if last_rc != 0 or not dest.exists():
                raise HarnessGeneratorError(
                    "git clone failed on host. "
                    + (f"Attempted: {attempted}. " if attempted else "")
                    + (f"Last rc={last_rc}." if last_rc is not None else "")
                )

            if spec.ref:
                proxy_overrides = _host_git_proxy_override_args()
                checkout_cmd = ["git", *proxy_overrides, "-C", str(dest), "checkout", spec.ref]
                print(f"[*] ➜  {' '.join(checkout_cmd)}")
                crc, cout, cerr, _ = _run_cmd_capture(checkout_cmd)
                if crc != 0:
                    if (t := _tail_lines(cerr)):
                        print("[warn] (host/git) checkout stderr (tail):\n" + textwrap.indent(t, "    "))
                    if (t := _tail_lines(cout)):
                        print("[warn] (host/git) checkout stdout (tail):\n" + textwrap.indent(t, "    "))
                    fetch_cmd = ["git", *proxy_overrides, "-C", str(dest), "fetch", "origin", spec.ref]
                    print(f"[*] ➜  {' '.join(fetch_cmd)}")
                    frc, fout, ferr, _ = _run_cmd_capture(fetch_cmd)
                    if frc != 0:
                        if (t := _tail_lines(ferr)):
                            print("[warn] (host/git) fetch stderr (tail):\n" + textwrap.indent(t, "    "))
                        if (t := _tail_lines(fout)):
                            print("[warn] (host/git) fetch stdout (tail):\n" + textwrap.indent(t, "    "))
                        raise HarnessGeneratorError(f"git fetch failed on host (rc={frc}).")
                    checkout_fh = ["git", *proxy_overrides, "-C", str(dest), "checkout", "FETCH_HEAD"]
                    print(f"[*] ➜  {' '.join(checkout_fh)}")
                    c2rc, c2out, c2err, _ = _run_cmd_capture(checkout_fh)
                    if c2rc != 0:
                        if (t := _tail_lines(c2err)):
                            print("[warn] (host/git) checkout FETCH_HEAD stderr (tail):\n" + textwrap.indent(t, "    "))
                        if (t := _tail_lines(c2out)):
                            print("[warn] (host/git) checkout FETCH_HEAD stdout (tail):\n" + textwrap.indent(t, "    "))
                        raise HarnessGeneratorError(f"git checkout FETCH_HEAD failed on host (rc={c2rc}).")

            # On Windows, mirrors may yield filemode-only diffs; ignore them.
            _set_git_core_filemode_off_host(dest)

            rev_cmd = ["git", "-C", str(dest), "rev-parse", "HEAD"]
            rev = subprocess.run(rev_cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            commit = (rev.stdout or "").strip() if rev.returncode == 0 else "<unknown>"
            print(f"[*] Checked out commit {commit}")
            return dest

        root = spec.workdir or Path(tempfile.mkdtemp(prefix="sherpa-fuzz-"))
        root = root.resolve()
        if root.exists() and any(root.iterdir()):
            # If provided, allow using an existing working folder (e.g., dev)
            print(f"[*] Using existing working directory: {root}")
            return root

        # If Docker mode is enabled, do clone/checkout via Docker too.
        # This makes the host requirement minimal (no git installation needed).
        if self.docker_image:
            parent = root.parent
            parent.mkdir(parents=True, exist_ok=True)
            name = root.name

            attempted: List[str] = []
            clone_success = False
            last_rc: Optional[int] = None
            for clone_url in _candidate_clone_urls(spec.url):
                attempted.append(clone_url)
                print(f"[*] (docker/git) Cloning {clone_url} → {root}")
                for attempt in range(1, max(1, GIT_CLONE_RETRIES) + 1):
                    temp_name = f"{name}.clone-{uuid.uuid4().hex[:8]}"
                    temp_root = parent / temp_name
                    clone_cmd = [
                        "docker",
                        "run",
                        "--rm",
                        *_docker_proxy_env_args(),
                        "-v",
                        f"{str(parent)}:/out",
                        "-w",
                        "/out",
                        DEFAULT_GIT_DOCKER_IMAGE,
                        "-c",
                        "http.version=HTTP/1.1",
                        "-c",
                        "http.postBuffer=524288000",
                        "clone",
                        "--depth",
                        "1",
                        clone_url,
                        temp_name,
                    ]
                    print(f"[*] ➜  {' '.join(clone_cmd)}")
                    try:
                        if temp_root.exists():
                            shutil.rmtree(temp_root, ignore_errors=True)
                    except Exception:
                        pass
                    try:
                        rc, out, err, timed_out = _run_cmd_capture(clone_cmd, timeout=GIT_DOCKER_CLONE_TIMEOUT_SEC)
                    except Exception as e:
                        rc, out, err, timed_out = 1, "", f"{e}", False

                    last_rc = rc

                    if timed_out:
                        if (t := _tail_lines(err)):
                            print("[warn] (docker/git) clone stderr (tail):\n" + textwrap.indent(t, "    "))
                        if (t := _tail_lines(out)):
                            print("[warn] (docker/git) clone stdout (tail):\n" + textwrap.indent(t, "    "))
                        print(
                            f"[warn] (docker/git) clone timed out (url={clone_url}, attempt {attempt}/{max(1, GIT_CLONE_RETRIES)}, timeout={GIT_DOCKER_CLONE_TIMEOUT_SEC}s); retrying..."
                        )
                        try:
                            if temp_root.exists():
                                shutil.rmtree(temp_root, ignore_errors=True)
                        except Exception:
                            pass
                        time.sleep(2 * attempt)
                        continue

                    if rc == 0:
                        try:
                            if root.exists():
                                shutil.rmtree(root, ignore_errors=True)
                            shutil.move(str(temp_root), str(root))
                        except Exception as move_err:
                            rc = 1
                            err = (err or "") + f"\n[warn] failed to promote temp clone {temp_root} -> {root}: {move_err}"
                            try:
                                if temp_root.exists():
                                    shutil.rmtree(temp_root, ignore_errors=True)
                            except Exception:
                                pass
                        if rc == 0:
                            clone_success = True
                            break

                    try:
                        if temp_root.exists():
                            shutil.rmtree(temp_root, ignore_errors=True)
                    except Exception:
                        pass

                    if (t := _tail_lines(err)):
                        print("[warn] (docker/git) clone stderr (tail):\n" + textwrap.indent(t, "    "))
                    if (t := _tail_lines(out)):
                        print("[warn] (docker/git) clone stdout (tail):\n" + textwrap.indent(t, "    "))
                    print(
                        f"[warn] (docker/git) clone failed (url={clone_url}, attempt {attempt}/{max(1, GIT_CLONE_RETRIES)}, rc={rc}); retrying..."
                    )
                    time.sleep(2 * attempt)

                if clone_success:
                    break

            if not clone_success:
                print("[warn] (docker/git) clone failed after retries; falling back to host git.")
                if attempted:
                    print(f"[warn] (docker/git) attempted clone URLs: {attempted}")
                if last_rc is not None:
                    print(f"[warn] (docker/git) last clone rc={last_rc}")
                try:
                    if root.exists():
                        shutil.rmtree(root, ignore_errors=True)
                except Exception:
                    pass
                return _clone_with_host_git(root)

            if spec.ref:
                checkout_cmd = [
                    "docker",
                    "run",
                    "--rm",
                    *_docker_proxy_env_args(),
                    "-v",
                    f"{str(root)}:/repo",
                    "-w",
                    "/repo",
                    DEFAULT_GIT_DOCKER_IMAGE,
                    "checkout",
                    spec.ref,
                ]
                print(f"[*] ➜  {' '.join(checkout_cmd)}")
                crc, cout, cerr, _ = _run_cmd_capture(checkout_cmd)
                if crc != 0:
                    if (t := _tail_lines(cerr)):
                        print("[warn] (docker/git) checkout stderr (tail):\n" + textwrap.indent(t, "    "))
                    if (t := _tail_lines(cout)):
                        print("[warn] (docker/git) checkout stdout (tail):\n" + textwrap.indent(t, "    "))
                    fetch_cmd = [
                        "docker",
                        "run",
                        "--rm",
                        *_docker_proxy_env_args(),
                        "-v",
                        f"{str(root)}:/repo",
                        "-w",
                        "/repo",
                        DEFAULT_GIT_DOCKER_IMAGE,
                        "-c",
                        "http.version=HTTP/1.1",
                        "-c",
                        "http.postBuffer=524288000",
                        "fetch",
                        "origin",
                        spec.ref,
                    ]
                    print(f"[*] ➜  {' '.join(fetch_cmd)}")
                    fe = None
                    for attempt in range(1, 4):
                        frc, fout, ferr, _ = _run_cmd_capture(fetch_cmd)
                        if frc == 0:
                            fe = type("_FE", (), {"returncode": 0})()  # type: ignore
                            break
                        if (t := _tail_lines(ferr)):
                            print("[warn] (docker/git) fetch stderr (tail):\n" + textwrap.indent(t, "    "))
                        if (t := _tail_lines(fout)):
                            print("[warn] (docker/git) fetch stdout (tail):\n" + textwrap.indent(t, "    "))
                        print(f"[warn] (docker/git) fetch failed (attempt {attempt}/3, rc={frc}); retrying...")
                        time.sleep(2 * attempt)
                        fe = type("_FE", (), {"returncode": frc})()  # type: ignore
                    assert fe is not None
                    if fe.returncode != 0:
                        raise HarnessGeneratorError(f"git fetch failed in docker (rc={fe.returncode}).")

                    checkout_fh = [
                        "docker",
                        "run",
                        "--rm",
                        *_docker_proxy_env_args(),
                        "-v",
                        f"{str(root)}:/repo",
                        "-w",
                        "/repo",
                        DEFAULT_GIT_DOCKER_IMAGE,
                        "checkout",
                        "FETCH_HEAD",
                    ]
                    print(f"[*] ➜  {' '.join(checkout_fh)}")
                    co2 = subprocess.run(checkout_fh, check=False, text=True)
                    if co2.returncode != 0:
                        raise HarnessGeneratorError(f"git checkout FETCH_HEAD failed in docker (rc={co2.returncode}).")

            # On Windows mounts, filemode diffs are common; ignore them.
            _set_git_core_filemode_off_docker(root)

            rev_cmd = [
                "docker",
                "run",
                "--rm",
                *_docker_proxy_env_args(),
                "-v",
                f"{str(root)}:/repo",
                "-w",
                "/repo",
                DEFAULT_GIT_DOCKER_IMAGE,
                "rev-parse",
                "HEAD",
            ]
            rev = subprocess.run(rev_cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            commit = (rev.stdout or "").strip() if rev.returncode == 0 else "<unknown>"
            print(f"[*] Checked out commit {commit}")
            return root

        # Fallback: clone on host.
        # Prefer GitPython if available, otherwise use the git CLI.
        if Repo is None or git_exc is None:
            try:
                return _clone_with_host_git(root)
            except FileNotFoundError:
                raise HarnessGeneratorError(
                    "GitPython is not available and 'git' is not found in PATH. "
                    "Enable Docker mode (--docker-image auto) or install Git."
                )

        print(f"[*] Cloning {spec.url} → {root}")
        repo = Repo.clone_from(spec.url, root)
        if spec.ref:
            try:
                repo.git.checkout(spec.ref)
            except git_exc.GitCommandError:
                repo.git.fetch("origin", spec.ref)
                repo.git.checkout("FETCH_HEAD")
        print(f"[*] Checked out commit {repo.head.commit.hexsha}")
        return root

    def _discover_fuzz_binaries(self) -> List[Path]:
        out = self.fuzz_out_dir
        if not out.is_dir():
            return []
        bins: List[Path] = []
        for p in out.iterdir():
            is_exe = os.access(p, os.X_OK) or p.suffix.lower() == ".exe"
            if p.is_file() and is_exe and FUZZ_BIN_PAT.match(p.name):
                bins.append(p)
        if not bins:
            # Fallback: scan for any executable in fuzz/out
            bins = [
                p
                for p in out.iterdir()
                if p.is_file() and (os.access(p, os.X_OK) or p.suffix.lower() == ".exe")
            ]
        return sorted(bins)

    def _locate_harness_source_for(self, fuzzer_name: str) -> Optional[Path]:
        # Heuristic: any file in fuzz/ with the fuzzer name and C/C++ or Java suffix
        exts = {".c", ".cc", ".cpp", ".cxx", ".java"}
        candidates: List[Path] = []
        for p in (self.repo_root / FUZZ_DIR).rglob("*"):
            if p.suffix.lower() in exts and fuzzer_name.split(".")[0] in p.name:
                candidates.append(p)
        if candidates:
            return sorted(candidates)[0]

        # Fallback: any file containing LLVMFuzzerTestOneInput
        for p in (self.repo_root / FUZZ_DIR).rglob("*"):
            if p.suffix.lower() in {".c", ".cc", ".cpp", ".cxx"}:
                try:
                    if "LLVMFuzzerTestOneInput" in p.read_text(encoding="utf-8", errors="ignore"):
                        return p
                except Exception:
                    continue
        return None

    # Run a command capturing stdout/stderr, optionally passing extra inputs after --
    def _run_cmd(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path,
        env: Optional[Dict[str, str]] = None,
        extra_inputs: Optional[List[str]] = None,
        timeout: int = 7200,
        idle_timeout: int = 0,
    ) -> Tuple[int, str, str]:
        def _redact_cmd(argv: Sequence[str]) -> List[str]:
            """Redact sensitive values from commands before printing.

            We frequently pass secrets (e.g., API keys) via `-e KEY=VALUE` to docker.
            Never echo those values into logs.
            """

            def _is_sensitive_key(k: str) -> bool:
                k_up = k.upper()
                return any(tok in k_up for tok in ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS"))

            redacted: List[str] = []
            i = 0
            while i < len(argv):
                a = str(argv[i])
                if a == "-e" and i + 1 < len(argv):
                    kv = str(argv[i + 1])
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        if _is_sensitive_key(k):
                            redacted += [a, f"{k}=***"]
                        else:
                            redacted += [a, kv]
                    else:
                        redacted += [a, kv]
                    i += 2
                    continue

                if "=" in a:
                    k, v = a.split("=", 1)
                    if _is_sensitive_key(k):
                        redacted.append(f"{k}=***")
                    else:
                        redacted.append(a)
                else:
                    redacted.append(a)
                i += 1
            return redacted

        if extra_inputs:
            # Append corpus/extra inputs directly. libFuzzer treats a bare "--" as an ignored flag
            # and logs noisy warnings that can look like failures in UI streams.
            cmd = list(cmd)
            cmd.extend(extra_inputs)

        effective_env = env or os.environ.copy()
        actual_cmd = self._dockerize_cmd(cmd, cwd=cwd, env=effective_env if self.docker_image else effective_env)

        start_ts = time.time()
        start_mono = time.monotonic()
        print(f"[*] ➜  {' '.join(_redact_cmd(actual_cmd))}", flush=True)
        proc = subprocess.Popen(
            actual_cmd,
            cwd=cwd,
            env=None if self.docker_image else effective_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            bufsize=1,
        )

        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []
        out_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()

        def _reader(pipe: Optional[object], kind: str) -> None:
            try:
                if pipe is None:
                    return
                for line in pipe:
                    out_queue.put((kind, line))
            finally:
                out_queue.put(None)

        t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
        t_out.start()
        t_err.start()

        done_readers = 0
        timed_out = False
        idle_timed_out = False
        heartbeat_sec = 10.0
        last_heartbeat = time.monotonic()
        last_activity = time.monotonic()

        try:
            while done_readers < 2:
                if timeout > 0 and (time.monotonic() - start_mono) > timeout:
                    timed_out = True
                    break

                try:
                    item = out_queue.get(timeout=0.2)
                except queue.Empty:
                    item = None

                if item is None:
                    done_readers += 1
                else:
                    kind, text = item
                    if kind == "stdout":
                        stdout_chunks.append(text)
                        print(text, end="", flush=True)
                    else:
                        stderr_chunks.append(text)
                        # Keep stderr visible in real-time to avoid silent failures.
                        print(text, end="", flush=True)
                    last_activity = time.monotonic()

                if idle_timeout > 0 and (time.monotonic() - last_activity) > idle_timeout:
                    idle_timed_out = True
                    timed_out = True
                    break

                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_sec:
                    elapsed = now - start_mono
                    print(f"[keepalive] command still running... elapsed={elapsed:.0f}s", flush=True)
                    last_heartbeat = now
        finally:
            if timed_out and proc.poll() is None:
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

            try:
                t_out.join(timeout=1)
            except Exception:
                pass
            try:
                t_err.join(timeout=1)
            except Exception:
                pass

            # Drain any remaining queued lines.
            while True:
                try:
                    item = out_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    continue
                kind, text = item
                if kind == "stdout":
                    stdout_chunks.append(text)
                else:
                    stderr_chunks.append(text)

        out = "".join(stdout_chunks)
        err = "".join(stderr_chunks)
        if idle_timed_out:
            err = (err or "") + f"\n[idle-timeout] no output for {idle_timeout}s; process was killed."
        elif timed_out:
            err = (err or "") + "\n[timeout] process exceeded limit and was killed."

        rc = proc.wait() if proc.poll() is None else proc.returncode
        elapsed = time.monotonic() - start_mono
        # Truncate verbose spam in the console but keep full logs if needed.
        print(
            f"[*] Command rc={rc}. elapsed={elapsed:.1f}s. started_at={start_ts:.0f}"
            + ". STDOUT (tail):\n"
            + "\n".join(out.splitlines()[-80:]),
            flush=True,
        )
        if err.strip():
            print("[*] STDERR (tail):\n" + "\n".join(err.splitlines()[-80:]), flush=True)
        return int(rc or 0), out, err

    # ────────────────────────────────────────────────────────────────────
    # Reproducer self-test & iterative repair via Codex
    # ────────────────────────────────────────────────────────────────────

    def _ensure_working_reproducer(self, *, max_retries: int = 3) -> bool:
        """Run reproduce.py/reproduce.sh and ensure it demonstrates the crash.

        Acceptable outcome:
          • The script exits with a *non-zero* status **and** stdout/stderr shows an
            AddressSanitizer, UBSan or similar sanitizer report *not* related to
            out-of-memory shadow allocation (this indicates the bug is hit).

        Failure modes we attempt to auto-repair:
          • Exit status 0 – nothing crashed.
          • Abort with messages like "AddressSanitizer failed to allocate" or
            "ReserveShadowMemoryRange failed" (indicative of incorrect env / ulimit).
          • Generic runtime errors (missing binary, permission denied, etc.).

        For each failure we send the diagnostics alongside the current reproduce.sh
        back to Codex and ask for a minimal fix.  We stop once validation succeeds
        or *max_retries* attempts have been exhausted.
        """

        rp_py = self.repo_root / "reproduce.py"
        rp_sh = self.repo_root / "reproduce.sh"
        if rp_py.exists():
            runner: Sequence[str] = [self._python_runner(), str(rp_py)]
        elif rp_sh.exists():
            runner = ["bash", str(rp_sh)]
            make_executable(rp_sh)
        else:
            raise HarnessGeneratorError("No reproducer script found after agent generation")

        failure_patterns = [
            re.compile(r"AddressSanitizer failed to allocate", re.IGNORECASE),
            re.compile(r"ReserveShadowMemoryRange failed", re.IGNORECASE),
            re.compile(r"usage: .*lib[Ff]uzzer", re.IGNORECASE),
        ]

        for attempt in range(1, max_retries + 1):
            print(f"[*] Validating reproducer (attempt {attempt}/{max_retries}) …")

            rc, out, err = self._run_cmd(list(runner), cwd=self.repo_root, timeout=600)

            combined = out + "\n" + err

            def _is_valid_failure() -> bool:
                """Determine whether the non-zero exit represents the intended bug.

                Accept either:
                  • Sanitizer-detected memory issues (AddressSanitizer, UBSan, etc.)
                  • LibFuzzer timeout/hang (contains "ALARM" or "timeout after")
                """

                if rc == 0:
                    return False

                # memory-bugs via sanitizers
                if re.search(r"==[0-9]+==ERROR: (Address|Undefined)Sanitizer", combined):
                    if any(p.search(combined) for p in failure_patterns):
                        return False
                    return True

                # hangs/timeouts (libFuzzer prints) or our enforced timeout marker
                if re.search(r"ALARM:|timeout after|\[timeout\] process exceeded", combined, re.IGNORECASE):
                    return True

                # General crash keywords
                return bool(re.search(r"Segmentation fault|core dumped|signal", combined, re.IGNORECASE))

            if _is_valid_failure():
                print("[*] Reproducer validation succeeded – bug reproduced (crash or hang).")
                return True

            # If reached, reproducer is faulty → send diagnostics back to Codex.
            if attempt == max_retries:
                print("[!] Reproducer still unreliable after attempts.")
                return False

            print("[!] Reproducer did not reproduce the intended crash. Sending diagnostics back to Codex …")

            current_reproducer = read_text_safely(rp_py if rp_py.exists() else rp_sh)
            diag_context = (
                "=== reproducer (current) ===\n" + current_reproducer +
                "\n\n=== run output ===\n" + strip_ansi(combined)
            )[-20000:]

            fix_prompt = textwrap.dedent(
                f"""
                The reproducer script failed to demonstrate the crash:
                Exit code: {rc}

                Objectives:
                                    • reproduce.py must exit *non-zero* due to the original bug (ASan/UBSan report),
                    a libFuzzer timeout (hang), or similar — but *not* due to allocation failures
                    or script errors.
                  • Ensure `<fuzz_target> -runs=1 <crashing_input>` runs with sane RLIMIT_AS or
                    LIBFUZZER options so that AddressSanitizer can allocate shadow memory.
                                    • If the bug is a **hang**, wrap the fuzzer invocation using Python subprocess timeout,
                                        slightly above any libFuzzer internal `-timeout`, so the script terminates and returns non-zero.
                  • If memory limits are the issue, add `export ASAN_OPTIONS=allow_user_segv_handler=1` or
                    loosen the limit, or run the binary under `ulimit -v unlimited`.

                                Apply the minimal fix to `reproduce.py` (and, only if absolutely required, small tweaks
                                under `fuzz/`).  Do not change unrelated files.  When done, write `reproduce.py` into
                `./done`.
                """
            ).strip()

            self.patcher.run_codex_command(fix_prompt, additional_context=diag_context)

            # loop continues; next iteration will pick up modified reproduce.sh

        # unreachable but mypy safety
        return False



# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and run a local fuzz harness for a generic Git repo (Codex-assisted).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument("--repo", help="Git URL of a single repository to fuzz")
    mx.add_argument("--targets", type=Path, help="YAML file listing multiple git repos to fuzz")

    parser.add_argument("--ref", help="Git ref (branch, tag, or commit) (only with --repo)")
    parser.add_argument("--workdir", type=Path, help="Existing directory to use as working tree (optional)")
    parser.add_argument("--ai-key-path", type=Path, default="./.env", help="Path to file with OPENAI_API_KEY (optional)")
    parser.add_argument("--sanitizer", default=DEFAULT_SANITIZER, help="Sanitizer for C/C++ (address, undefined, etc.)")
    parser.add_argument("--codex-cli", default="opencode", help="OpenCode CLI executable (kept as --codex-cli for compatibility)")
    parser.add_argument(
        "--codex-no-sandbox",
        action="store_true",
        help="Use a broader Codex sandbox (danger-full-access). Use with caution.",
    )
    parser.add_argument(
        "--codex-sandbox-mode",
        choices=["read-only", "workspace-write", "danger-full-access"],
        help="Codex sandbox mode override (default: workspace-write)",
    )
    parser.add_argument("--time-budget", type=int, default=900, help="libFuzzer/Jazzer -max_total_time per target (seconds)")
    parser.add_argument("--rss-limit-mb", type=int, default=8192, help="RSS limit for runs (MB)")
    parser.add_argument("--max-len", type=int, default=1024, help="libFuzzer -max_len")
    parser.add_argument(
        "--docker-image",
        default=None,
        help="If set, run build/fuzz commands inside a Linux Docker image. Use 'auto' to choose per-language images (cpp/java) and auto-build if missing.",
    )
    parser.add_argument("--max-retries", type=int, default=MAX_BUILD_RETRIES, help="Max build-fix rounds")
    parser.add_argument("--max-threads", type=int, default=1, help="Maximum repositories to process in parallel (only with --targets)")
    parser.add_argument("--rounds", type=int, default=1, help="Number of iterative harness-generation rounds to run per repository")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    load_dotenv(os.path.expanduser(str(args.ai_key_path)))

    # helper to allocate unique subdirectories under a base workdir
    from urllib.parse import urlparse

    def _alloc_workdir(base: Path, url: str) -> Path:
        """Return a unique child directory for a repo url inside base."""
        repo_name = os.path.basename(urlparse(url).path)  # e.g., 'foo.git'
        stem = repo_name[:-4] if repo_name.endswith(".git") else repo_name
        cand = base / stem
        if not cand.exists():
            return cand
        for i in range(1, 1000):
            cand_i = base / f"{stem}-{i}"
            if not cand_i.exists():
                return cand_i
        return base / f"{stem}-{uuid.uuid4().hex[:8]}"

    base_workdir: Optional[Path] = None
    if args.workdir:
        base_workdir = args.workdir.expanduser().resolve()
        base_workdir.mkdir(parents=True, exist_ok=True)
    else:
        env_out = os.environ.get("SHERPA_OUTPUT_DIR", "").strip()
        if env_out:
            base_workdir = Path(env_out).expanduser().resolve()
            base_workdir.mkdir(parents=True, exist_ok=True)

    # Build list of RepoSpec objects
    specs: List[RepoSpec] = []
    if args.repo:
        work = _alloc_workdir(base_workdir, args.repo) if base_workdir else None
        specs.append(RepoSpec(url=args.repo, ref=args.ref, workdir=work))
    else:
        import yaml  # lazy import; heavy only if we need it

        targets_path: Path = args.targets.expanduser()
        if not targets_path.is_file():
            print(f"[cli] ERROR: targets file {targets_path} does not exist", file=sys.stderr)
            sys.exit(1)

        try:
            data = yaml.safe_load(targets_path.read_text())
        except Exception as e:
            print(f"[cli] ERROR: failed to parse YAML: {e}", file=sys.stderr)
            sys.exit(1)

        if not isinstance(data, list):
            print("[cli] ERROR: targets YAML must be a list of URLs or dicts", file=sys.stderr)
            sys.exit(1)

        for idx, item in enumerate(data, 1):
            if isinstance(item, str):
                work = _alloc_workdir(base_workdir, item) if base_workdir else None
                specs.append(RepoSpec(url=item, workdir=work))
            elif isinstance(item, dict):
                url = item.get("url") or item.get("repo")
                if not url:
                    print(f"[cli] ERROR: item #{idx} missing 'url' field", file=sys.stderr)
                    sys.exit(1)
                work = _alloc_workdir(base_workdir, url) if base_workdir else None
                specs.append(RepoSpec(url=url, ref=item.get("ref"), workdir=work))
            else:
                print(f"[cli] ERROR: Unsupported YAML item type at #{idx}", file=sys.stderr)
                sys.exit(1)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _worker(spec: RepoSpec) -> Tuple[str, Optional[str]]:
        """Run the generation workflow for a single repo.

        Returns (url, error_message). error_message is None on success.
        """
        try:
            print("=" * 80)
            print(f"[+] Processing repository: {spec.url} (ref={spec.ref or 'default'})")

            repo_root: Optional[Path] = None

            for rnd in range(1, args.rounds + 1):
                print(f"--- Round {rnd}/{args.rounds} for {spec.url} ---")

                # Instantiate generator (reuse same working directory if already cloned).
                round_spec = RepoSpec(url=spec.url, ref=spec.ref, workdir=spec.workdir or repo_root)

                gen = NonOssFuzzHarnessGenerator(
                    repo_spec=round_spec,
                    ai_key_path=args.ai_key_path.expanduser(),
                    sanitizer=args.sanitizer,
                    codex_cli=args.codex_cli,
                    time_budget_per_target=args.time_budget,
                    codex_dangerous=args.codex_no_sandbox,
                    codex_sandbox_mode=args.codex_sandbox_mode,
                    rss_limit_mb=args.rss_limit_mb,
                    max_len=args.max_len,
                    max_build_retries=args.max_retries,
                    docker_image=args.docker_image,
                )
                # If not first round move old artifacts away before generating.
                if rnd > 1:
                    art_dir = gen.fuzz_out_dir / 'artifacts'
                    if art_dir.is_dir():
                        archive = art_dir / 'old'
                        archive.mkdir(exist_ok=True)
                        for p in art_dir.glob('*'):
                            if p.is_file():
                                p.rename(archive / p.name)

                gen.round_index = rnd
                gen.generate()

                repo_root = gen.repo_root  # reuse this for next round
            return (spec.url, None)
        except HarnessGeneratorError as e:
            return (spec.url, str(e))
        except Exception as e:  # generic safety net
            return (spec.url, f"Unhandled exception: {e}")

    max_threads = max(1, int(args.max_threads))

    if max_threads == 1 or len(specs) == 1:
        # Sequential to keep logs readable / if only one repo.
        for spec in specs:
            url, err = _worker(spec)
            if err:
                print(f"[harness_generator] ERROR processing {url}: {err}", file=sys.stderr)
                sys.exit(1)
    else:
        # Parallel execution with simple thread pool.
        print(f"[*] Processing {len(specs)} repositories with up to {max_threads} thread(s)…")
        failures = []
        with ThreadPoolExecutor(max_workers=max_threads) as exe:
            future_map = {exe.submit(_worker, s): s.url for s in specs}
            for fut in as_completed(future_map):
                url = future_map[fut]
                try:
                    _, err = fut.result()
                except Exception as e:
                    err = f"Unhandled exception in thread: {e}"
                if err:
                    failures.append((url, err))

        if failures:
            for url, msg in failures:
                print(f"[harness_generator] ERROR processing {url}: {msg}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
