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
import bz2
import difflib
import gzip
import io
import json
import logging
import lzma
import os
import queue
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
import hashlib
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
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

DEFAULT_GIT_DOCKER_IMAGE = os.environ.get(
    "SHERPA_GIT_DOCKER_IMAGE",
    "m.daocloud.io/docker.io/alpine/git",
)

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


def _default_git_mirror_specs() -> List[str]:
    # Built-in mirror-first defaults for GitHub clone stability.
    return [
        "https://ghfast.top/{url}",
        "https://ghproxy.net/{url}",
    ]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, *, min_value: float = 0.0, max_value: float = 3600.0) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except Exception:
        return default
    return max(min_value, min(max_value, val))


def _retry_backoff_seconds(
    attempt: int,
    *,
    base_env: str,
    cap_env: str,
    default_base: float,
    default_cap: float,
) -> float:
    base = _env_float(base_env, default_base, min_value=0.0, max_value=300.0)
    cap = _env_float(cap_env, default_cap, min_value=0.0, max_value=600.0)
    wait = base * max(1, int(attempt))
    return min(wait, cap)

# Make CodexHelper discoverable in both "package" and "flat script" use.
try:
    from .codex_helper import CodexHelper  # type: ignore
except Exception:  # pragma: no cover
    from codex_helper import CodexHelper  # type: ignore


# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

DEFAULT_SANITIZER = os.environ.get("SHERPA_SANITIZER", "address")
# Supported sanitizer configurations for multi-sanitizer fuzzing
SANITIZER_CONFIGS: Dict[str, Dict[str, str]] = {
    "address": {
        "compile_flags": "-fsanitize=address,undefined,fuzzer",
        "asan_options": "exitcode=76:detect_leaks=0",
        "ubsan_options": "print_stacktrace=1",
    },
    "memory": {
        "compile_flags": "-fsanitize=memory,fuzzer -fno-omit-frame-pointer -fPIE",
        "msan_options": "exitcode=77:halt_on_error=1",
    },
    "undefined": {
        "compile_flags": "-fsanitize=undefined,fuzzer",
        "ubsan_options": "print_stacktrace=1:halt_on_error=1",
    },
    "thread": {
        "compile_flags": "-fsanitize=thread,fuzzer",
        "tsan_options": "exitcode=78",
    },
}
MAX_BUILD_RETRIES = int(os.environ.get("SHERPA_MAX_BUILD_RETRIES", "3"))
CODEX_ANALYSIS_MODEL = os.environ.get("CODEX_ANALYSIS_MODEL", "sonnet")
CODEX_APPROVAL_MODE = os.environ.get("CODEX_APPROVAL_MODE", "full-auto")

FUZZ_DIR = "fuzz"
FUZZ_OUT_DIR = "fuzz/out"
FUZZ_CORPUS_DIR = "fuzz/corpus"
FUZZ_DICT_DIR = "fuzz/dict"
ARTIFACT_PREFIX = "artifacts"
FUZZ_SYSTEM_PACKAGES_FILE = os.environ.get("SHERPA_FUZZ_SYSTEM_PACKAGES_FILE", "fuzz/system_packages.txt")
VCPKG_REPO_DIR = (os.environ.get("SHERPA_VCPKG_REPO_DIR") or "vcpkg").strip() or "vcpkg"
VCPKG_INSTALLED_DIR = (os.environ.get("SHERPA_VCPKG_INSTALLED_DIR") or "vcpkg_installed").strip() or "vcpkg_installed"
VCPKG_PORT_ALIASES: Dict[str, str] = {
    # Common generic/library shorthands -> vcpkg ports
    "z": "zlib",
    "bz2": "bzip2",
    "lzma": "liblzma",
    "xz": "liblzma",
    "ssl": "openssl",
    "crypto": "openssl",
    "libssl": "openssl",
    "libcrypto": "openssl",
    "xml2": "libxml2",
    "libxml": "libxml2",
}
ALLOWED_TARGET_TYPES = {
    "parser",
    "decoder",
    "archive",
    "image",
    "document",
    "network",
    "database",
    "serializer",
    "interpreter",
    "generic",
}

ALLOWED_SEED_PROFILES = {
    "parser-structure",
    "parser-token",
    "parser-format",
    "parser-numeric",
    "decoder-binary",
    "archive-container",
    "serializer-structured",
    "document-text",
    "network-message",
    "generic",
}

YAML_SEED_FAMILIES = {
    "flow_structures",
    "block_scalars",
    "anchors_aliases",
    "tags_directives",
    "document_markers",
    "delimiter_fragments",
    "unterminated_fragments",
    "malformed_separators",
}

FMT_SEED_FAMILIES = {
    "replacement_fields",
    "escaped_braces",
    "positional_arguments",
    "format_specifiers",
    "width_precision",
    "fill_align",
    "type_conversions",
    "malformed_replacement_fields",
}

# ── Per-profile dictionary tokens (libFuzzer -dict= format) ──────────────
# Each entry is a raw byte-string that libFuzzer will use as a mutation hint.
PROFILE_DICTIONARY_TOKENS: Dict[str, List[str]] = {
    "parser-structure": [
        '":"', '"{"', '"}"', '"["', '"]"', '","', '"true"', '"false"', '"null"',
        '"---"', '"..."', '": "', '"- "', '"\\n"', '"\\t"',
    ],
    "parser-token": [
        '"<"', '">"', '"</"', '"/>"', '"="', '"\\""', '"&amp;"', '"&lt;"', '"&gt;"',
        '"<!--"', '"-->"', '"<![CDATA["', '"]]>"',
    ],
    "parser-format": [
        '"%s"', '"%d"', '"%f"', '"%x"', '"%p"', '"%n"', '"%%"', '"%0"',
        '"{}"', '"{0}"', '"{:"', '"}"',
    ],
    "parser-numeric": [
        '"0"', '"-1"', '"2147483647"', '"-2147483648"', '"0x"', '"0b"', '"0o"',
        '"NaN"', '"Inf"', '"-Inf"', '"1e308"', '"-1e308"',
    ],
    "decoder-binary": [
        '"\\x00"', '"\\xff"', '"\\x89PNG"', '"\\xff\\xd8\\xff"', '"GIF89a"',
        '"BM"', '"RIFF"', '"\\x00\\x00\\x01\\x00"',
    ],
    "archive-container": [
        '"PK\\x03\\x04"', '"PK\\x05\\x06"', '"\\x1f\\x8b\\x08"',
        '"BZh"', '"\\xfd7zXZ\\x00"', '"Rar!\\x1a\\x07"',
        '"7z\\xbc\\xaf\\x27\\x1c"', '"\\x75\\x73\\x74\\x61\\x72"',
    ],
    "serializer-structured": [
        '":"', '"{"', '"}"', '"["', '"]"', '","', '"\\n"',
        '"true"', '"false"', '"null"', '"\\""',
    ],
    "document-text": [
        '"<html"', '"<body"', '"<div"', '"<p>"', '"</p>"', '"<a "', '"href="',
        '"<!DOCTYPE"', '"<head"', '"<script"', '"<style"',
    ],
    "network-message": [
        '"GET "', '"POST "', '"HTTP/1."', '"\\r\\n"', '"Host: "',
        '"Content-Length: "', '"Content-Type: "', '"\\r\\n\\r\\n"',
    ],
    "generic": [],
}

# ── Per-profile adaptive max_len (bytes) ─────────────────────────────────
PROFILE_MAX_LEN: Dict[str, int] = {
    "parser-structure": 4096,
    "parser-token": 2048,
    "parser-format": 2048,
    "parser-numeric": 512,
    "decoder-binary": 16384,
    "archive-container": 65536,
    "serializer-structured": 4096,
    "document-text": 8192,
    "network-message": 4096,
    "generic": 1024,
}

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


def _synthesize_opencode_idle_timeout_sec() -> int:
    raw = (os.environ.get("SHERPA_OPENCODE_IDLE_TIMEOUT_SYNTH_SEC") or "900").strip()
    try:
        return max(0, min(int(raw), 86_400))
    except Exception:
        return 900


def _synthesize_activity_watch_paths() -> list[str]:
    return [
        "fuzz/build.py",
        "fuzz/README.md",
        "fuzz/system_packages.txt",
        "fuzz/*.c",
        "fuzz/*.cc",
        "fuzz/*.cpp",
        "fuzz/*.cxx",
        "fuzz/*.java",
        "fuzz/**/*.c",
        "fuzz/**/*.cc",
        "fuzz/**/*.cpp",
        "fuzz/**/*.cxx",
        "fuzz/**/*.java",
    ]


def _run_plateau_pulses() -> int:
    raw = (os.environ.get("SHERPA_RUN_PLATEAU_PULSES") or "3").strip()
    try:
        return max(1, min(int(raw), 20))
    except Exception:
        return 3


def _run_plateau_idle_growth_sec() -> int:
    raw = (os.environ.get("SHERPA_RUN_PLATEAU_IDLE_GROWTH_SEC") or "180").strip()
    try:
        return max(30, min(int(raw), 86_400))
    except Exception:
        return 180


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


def _host_git_proxy_env() -> Dict[str, str]:
    env = os.environ.copy()

    def _pick_env(*names: str) -> str:
        for n in names:
            v = env.get(n)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""

    http_proxy = _pick_env("HTTP_PROXY", "http_proxy", "SHERPA_GIT_HTTP_PROXY", "SHERPA_DOCKER_HTTP_PROXY")
    https_proxy = _pick_env("HTTPS_PROXY", "https_proxy", "SHERPA_GIT_HTTPS_PROXY", "SHERPA_DOCKER_HTTPS_PROXY")
    all_proxy = _pick_env("ALL_PROXY", "all_proxy")
    no_proxy = _pick_env("NO_PROXY", "no_proxy", "SHERPA_GIT_NO_PROXY", "SHERPA_DOCKER_NO_PROXY")

    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy
    if https_proxy:
        env["HTTPS_PROXY"] = https_proxy
        env["https_proxy"] = https_proxy
    if all_proxy:
        env["ALL_PROXY"] = all_proxy
        env["all_proxy"] = all_proxy
    if no_proxy:
        env["NO_PROXY"] = no_proxy
        env["no_proxy"] = no_proxy
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return env


def _candidate_clone_urls(url: str) -> List[str]:
    """Return clone URLs, optionally extended by explicitly configured mirrors."""

    urls: List[str] = []
    if not url.startswith("https://github.com/"):
        # Non-GitHub URLs are returned as-is to avoid breaking custom hosts.
        return [url]

    mirror_specs: List[str] = []
    sherpa_git_mirrors = _get_sherpa_git_mirrors()
    if sherpa_git_mirrors:
        mirror_specs.extend([p.strip() for p in sherpa_git_mirrors.split(",") if p.strip()])
    else:
        mirror_specs.extend(_default_git_mirror_specs())

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

    if url not in urls:
        # Keep official github as the final fallback.
        urls.append(url)
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
    terminal_reason: str = ""
    plateau_detected: bool = False
    plateau_idle_seconds: int = 0
    seed_quality: Dict[str, object] | None = None


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


def parse_libfuzzer_progress_events(text: str) -> List[Dict[str, int | str]]:
    events: List[Dict[str, int | str]] = []
    if not text:
        return events
    for line in text.splitlines():
        m = _LIBFUZZER_PROGRESS_RE.search(line)
        if not m:
            continue
        events.append(
            {
                "iteration": int(m.group("iter") or 0),
                "kind": str(m.group("kind") or "").upper(),
                "cov": int(m.group("cov") or 0),
                "ft": int(m.group("ft") or 0),
                "corpus_files": int(m.group("corp_files") or 0),
                "corpus_size_bytes": _parse_size_token_to_bytes(m.group("corp_size") or ""),
                "execs_per_sec": int(m.group("execs") or 0),
                "rss_mb": int(m.group("rss") or 0),
            }
        )
    return events


def _seed_quality_from_run(
    *,
    log: str,
    initial_corpus_files: int,
    initial_corpus_bytes: int,
    final_stats: Dict[str, int],
    required_families: list[str],
    covered_families: list[str],
    repo_examples_count: int,
    plateau_idle_seconds: int,
    seed_profile: str = "",
    archive_valid_count: int = 0,
    archive_valid_ratio: float = 1.0,
    archive_min_valid_ratio: float = 0.60,
    archive_malformed_ratio: float = 0.0,
    archive_max_malformed_ratio: float = 0.30,
) -> Dict[str, object]:
    events = parse_libfuzzer_progress_events(log)
    inited_cov = 0
    inited_ft = 0
    if events:
        first = events[0]
        inited_cov = int(first.get("cov") or 0)
        inited_ft = int(first.get("ft") or 0)

    def _event_by_iter(iter_limit: int) -> Dict[str, int | str]:
        chosen: Dict[str, int | str] = {}
        for event in events:
            if int(event.get("iteration") or 0) <= iter_limit:
                chosen = event
            else:
                break
        return chosen

    at_30s = _event_by_iter(131072)
    at_60s = _event_by_iter(262144)
    early_new_units_30s = max(0, int(at_30s.get("corpus_files") or 0) - initial_corpus_files)
    early_new_units_60s = max(0, int(at_60s.get("corpus_files") or 0) - initial_corpus_files)
    final_files = int(final_stats.get("corpus_files") or 0)
    final_bytes = int(final_stats.get("corpus_size_bytes") or 0)
    retention_files = (float(final_files) / float(initial_corpus_files)) if initial_corpus_files > 0 else 0.0
    retention_bytes = (float(final_bytes) / float(initial_corpus_bytes)) if initial_corpus_bytes > 0 else 0.0

    def _slope(key: str) -> float:
        if len(events) < 2:
            return 0.0
        start = events[0]
        end = events[-1]
        delta_iter = max(1, int(end.get("iteration") or 0) - int(start.get("iteration") or 0))
        return float(int(end.get(key) or 0) - int(start.get(key) or 0)) / float(delta_iter)

    quality_flags: list[str] = []
    missing_families = [x for x in required_families if x and x not in set(covered_families)]
    if initial_corpus_files >= 16 and retention_files > 0 and retention_files <= 0.25:
        quality_flags.append("low_retention")
    if early_new_units_30s <= 0 and early_new_units_60s <= 0:
        quality_flags.append("low_early_yield")
    if initial_corpus_files >= 16 and final_files <= 12 and int(final_stats.get("cov") or 0) <= max(inited_cov, 1):
        quality_flags.append("high_homogeneity")
    if missing_families:
        quality_flags.append("missing_required_families")
    if repo_examples_count == 0 and any(f in (YAML_SEED_FAMILIES | FMT_SEED_FAMILIES) for f in required_families):
        quality_flags.append("repo_examples_missing")
    if seed_profile == "archive-container":
        if archive_valid_count < 1:
            quality_flags.append("archive_valid_seed_missing")
        if archive_valid_ratio < archive_min_valid_ratio:
            quality_flags.append("archive_seed_validity_low")
        if archive_malformed_ratio > archive_max_malformed_ratio:
            quality_flags.append("archive_seed_malformed_ratio_high")

    required_total = len([x for x in required_families if x])
    covered_required = len([x for x in required_families if x and x in set(covered_families)])
    family_coverage_ratio = (
        float(covered_required) / float(required_total)
        if required_total > 0
        else 1.0
    )
    archive_validity_component = (
        max(0.0, min(1.0, float(archive_valid_ratio)))
        if seed_profile == "archive-container"
        else 1.0
    )
    retention_component = max(0.0, min(1.0, retention_files))
    validity = max(
        0.0,
        min(
            1.0,
            0.60 * family_coverage_ratio
            + 0.25 * archive_validity_component
            + 0.15 * retention_component,
        ),
    )

    cov_gain = max(0, int(final_stats.get("cov") or 0) - inited_cov)
    ft_gain = max(0, int(final_stats.get("ft") or 0) - inited_ft)
    early_units_norm = max(0.0, min(1.0, float(max(early_new_units_30s, early_new_units_60s)) / 16.0))
    coverage_potential = max(
        0.0,
        min(
            1.0,
            0.45 * max(0.0, min(1.0, float(cov_gain) / 24.0))
            + 0.30 * max(0.0, min(1.0, float(ft_gain) / 240.0))
            + 0.25 * early_units_norm,
        ),
    )

    novelty = max(
        0.0,
        min(
            1.0,
            0.50 * max(0.0, min(1.0, float(early_new_units_60s) / 24.0))
            + 0.30 * family_coverage_ratio
            + 0.20 * max(0.0, min(1.0, retention_component)),
        ),
    )

    redundancy_penalty = 0.0
    if "high_homogeneity" in quality_flags:
        redundancy_penalty += 0.45
    if "low_retention" in quality_flags:
        redundancy_penalty += 0.20
    if "low_early_yield" in quality_flags:
        redundancy_penalty += 0.20
    if "missing_required_families" in quality_flags:
        redundancy_penalty += 0.15
    redundancy_penalty = max(0.0, min(1.0, redundancy_penalty))

    alpha, beta, gamma, eta = 0.40, 0.35, 0.25, 0.20
    seed_score = max(
        0.0,
        min(
            1.0,
            alpha * coverage_potential
            + beta * validity
            + gamma * novelty
            - eta * redundancy_penalty,
        ),
    )
    return {
        "initial_corpus_files": initial_corpus_files,
        "initial_corpus_bytes": initial_corpus_bytes,
        "initial_inited_cov": inited_cov,
        "initial_inited_ft": inited_ft,
        "early_new_units_30s": early_new_units_30s,
        "early_new_units_60s": early_new_units_60s,
        "final_corpus_files": final_files,
        "final_corpus_bytes": final_bytes,
        "corpus_retention_ratio_files": retention_files,
        "corpus_retention_ratio_bytes": retention_bytes,
        "cov_growth_slope_pre_plateau": _slope("cov"),
        "ft_growth_slope_pre_plateau": _slope("ft"),
        "plateau_after_sec": plateau_idle_seconds,
        "seed_profile": seed_profile,
        "archive_valid_count": archive_valid_count,
        "archive_valid_ratio": archive_valid_ratio,
        "archive_min_valid_ratio": archive_min_valid_ratio,
        "archive_malformed_ratio": archive_malformed_ratio,
        "archive_max_malformed_ratio": archive_max_malformed_ratio,
        "seed_score": float(seed_score),
        "seed_score_components": {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "eta": eta,
            "coverage_potential": float(coverage_potential),
            "validity": float(validity),
            "novelty": float(novelty),
            "redundancy_penalty": float(redundancy_penalty),
            "family_coverage_ratio": float(family_coverage_ratio),
        },
        "quality_flags": quality_flags,
    }


def _infer_target_type(*parts: str) -> str:
    text = " ".join(p for p in parts if p).lower()
    if any(tok in text for tok in ("format", "printf", "replacement field", "specifier", "brace", "template")):
        return "parser"
    if any(tok in text for tok in ("parse", "parser", "scan", "scanner", "yaml", "json", "xml", "token", "lex", "reader")):
        return "parser"
    if any(tok in text for tok in ("decode", "decoder", "decompress", "inflate", "unpack")):
        return "decoder"
    if any(tok in text for tok in ("archive", "untar", "unzip", "tar", "zip", "rar", "7z")):
        return "archive"
    if any(tok in text for tok in ("png", "jpeg", "jpg", "gif", "bmp", "image", "pixel")):
        return "image"
    if any(tok in text for tok in ("pdf", "doc", "document", "html", "markdown")):
        return "document"
    if any(tok in text for tok in ("socket", "packet", "http", "tls", "dns", "frame", "request", "response")):
        return "network"
    if any(tok in text for tok in ("sql", "query", "db", "database", "sqlite", "record")):
        return "database"
    if any(tok in text for tok in ("emit", "dump", "serialize", "serializer", "write")):
        return "serializer"
    if any(tok in text for tok in ("eval", "vm", "execute", "compile", "bytecode", "script", "interp")):
        return "interpreter"
    return "generic"


def _is_fmt_format_target(*parts: str) -> bool:
    text = " ".join(p for p in parts if p).lower()
    return bool(
        "fmt" in text
        and any(tok in text for tok in ("format", "format_to", "vformat", "println", "print", "replacement field", "specifier"))
    )


def _looks_textual_seed(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except Exception:
        return False
    if not data:
        return True
    printable = 0
    for b in data[:2048]:
        if b in (9, 10, 13) or 32 <= b <= 126:
            printable += 1
    return (float(printable) / float(max(1, min(len(data), 2048)))) >= 0.8


def _normalized_format_shape(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\d+", "#", lowered)
    lowered = re.sub(r"[a-z]+", "a", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


# ── Crash stack signature extraction ─────────────────────────────────

_STACK_FRAME_RE = re.compile(
    r"#\d+\s+(?:0x[0-9a-f]+\s+in\s+)?(\S+)"
    r"|#\d+\s+(\S+)\s+\("
)

_SANITIZER_TYPE_RE = re.compile(
    r"ERROR:\s*(Address|Undefined|Memory|Thread|Leak)Sanitizer:\s*(\S+)"
)


def extract_crash_stack_signature(log: str, top_n: int = 3) -> Dict[str, str]:
    """
    Extract a normalized crash signature from sanitizer output.
    Returns {crash_type, stack_signature, top_frames}.
    """
    crash_type = "unknown"
    m = _SANITIZER_TYPE_RE.search(log)
    if m:
        crash_type = f"{m.group(1)}Sanitizer:{m.group(2)}"

    frames: list[str] = []
    for fm in _STACK_FRAME_RE.finditer(log):
        func_name = fm.group(1) or fm.group(2)
        if func_name and func_name not in ("__asan", "__ubsan", "__interceptor", "<null>"):
            # Strip address offsets and file locations, keep function name only
            clean = re.sub(r"\+0x[0-9a-f]+$", "", func_name)
            clean = re.sub(r"\(.*\)$", "", clean)
            if clean and clean not in frames:
                frames.append(clean)
            if len(frames) >= top_n:
                break

    top_frames = "|".join(frames[:top_n]) if frames else "no_frames"
    sig_input = f"{crash_type}:{top_frames}"
    stack_signature = hashlib.sha256(sig_input.encode("utf-8")).hexdigest()[:16]

    return {
        "crash_type": crash_type,
        "stack_signature": stack_signature,
        "top_frames": top_frames,
    }


def _seed_filter_mode() -> str:
    raw = (os.environ.get("SHERPA_SEED_FILTER_MODE") or "soft").strip().lower()
    if raw in {"strict", "soft", "off"}:
        return raw
    return "soft"


def _seed_families_for_target(seed_profile: str, *parts: str) -> tuple[list[str], list[str]]:
    profile = str(seed_profile or "").strip().lower()
    text = " ".join(p for p in parts if p).lower()
    required: list[str] = []
    optional: list[str] = []
    if profile == "parser-format" and _is_fmt_format_target(text):
        required.extend(
            [
                "replacement_fields",
                "escaped_braces",
                "positional_arguments",
                "format_specifiers",
                "width_precision",
                "fill_align",
                "type_conversions",
                "malformed_replacement_fields",
            ]
        )
        return required, optional
    if profile == "parser-structure":
        required.extend(["document_markers", "block_scalars", "anchors_aliases", "tags_directives"])
        optional.extend(["flow_structures", "unterminated_fragments", "malformed_separators"])
    elif profile == "parser-token":
        required.extend(["delimiter_fragments", "unterminated_fragments", "malformed_separators"])
        optional.extend(["document_markers", "tags_directives", "flow_structures"])
    elif profile == "parser-format":
        required.extend(["delimiter_fragments", "unterminated_fragments", "malformed_separators"])
    elif profile == "parser-numeric":
        required.extend(["delimiter_fragments", "malformed_separators"])
    if any(tok in text for tok in ("yaml", "yml")):
        for family in [
            "flow_structures",
            "block_scalars",
            "anchors_aliases",
            "tags_directives",
            "document_markers",
            "delimiter_fragments",
            "unterminated_fragments",
            "malformed_separators",
        ]:
            if family not in required:
                required.append(family)
    return required, [x for x in optional if x not in required]


def _classify_seed_family(path: Path) -> set[str]:
    name = path.name.lower()
    text = read_text_safely(path)[:2048].lower()
    families: set[str] = set()
    if "{}" in text or re.search(r"\{[^{}]*\}", text):
        families.add("replacement_fields")
    if "{{" in text or "}}" in text:
        families.add("escaped_braces")
    if re.search(r"\{[0-9]+\}", text):
        families.add("positional_arguments")
    if re.search(r"\{[^{}:]+:[^{}]+\}|\{:[^{}]+\}", text):
        families.add("format_specifiers")
    if re.search(r"\{[^{}]*(:[^{}]*[0-9]+\.[0-9]*|:[^{}]*\.[0-9]+)\}", text) or any(tok in text for tok in (".0f", ".1f", ".2f", ":.3", ":#.")):
        families.add("width_precision")
    if re.search(r"\{:[<>=^][^{}]*\}", text) or any(tok in text for tok in (":<", ":>", ":^", "{:*", "{:_", "{:0")):
        families.add("fill_align")
    if re.search(r"\{[^{}]*:[^{}]*[bcdeEfFgGosxXpn?]\}", text):
        families.add("type_conversions")
    if text.count("{") != text.count("}") or any(tok in text for tok in ("{", "}", "{{{", "}}}", "{:", "{0:", "{name")):
        if not {"replacement_fields", "escaped_braces"} <= families:
            families.add("malformed_replacement_fields")
    if any(tok in name for tok in ("delimiter-", "leading-colon", "trailing-space", "indicator-question")):
        families.add("delimiter_fragments")
    if any(tok in name for tok in ("unterminated-",)):
        families.add("unterminated_fragments")
    if any(tok in name for tok in ("malformed-",)):
        families.add("malformed_separators")
    if any(tok in name for tok in ("block-only", "block-scalar")) or any(tok in text for tok in ("\n|", "\n>", "|-", ">-", "|+", ">+")):
        families.add("block_scalars")
    if any(tok in name for tok in ("anchor", "alias")) or any(tok in text for tok in ("&", "*")):
        families.add("anchors_aliases")
    if any(tok in name for tok in ("tag", "directive", "yaml-version")) or any(tok in text for tok in ("%yaml", "%tag", "!!", "!<")):
        families.add("tags_directives")
    if any(tok in name for tok in ("doc-marker", "yaml-version")) or any(tok in text for tok in ("---", "...")):
        families.add("document_markers")
    if any(tok in name for tok in ("flow", "array", "mapping", "json")) or any(tok in text for tok in ("[", "]", "{", "}")):
        families.add("flow_structures")
    return families


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
        rss_limit_mb: int = 131072,
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
        self._active_run_procs_lock = threading.Lock()
        self._active_run_procs: set[subprocess.Popen[str]] = set()
        self.last_seed_profile_by_fuzzer: Dict[str, str] = {}
        self.last_seed_bootstrap_by_fuzzer: Dict[str, Dict[str, object]] = {}
        self.last_selected_target_by_fuzzer: Dict[str, Dict[str, object]] = {}
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

        # In k8s_job mode, Docker client in pod may point to host daemon where
        # pod-internal repo paths are not mountable. Force host-git operations
        # for CodexHelper to avoid "not a git repository" failures.
        git_docker_image = self.docker_image if self.docker_image else None
        if (os.environ.get("SHERPA_EXECUTOR_MODE", "").strip().lower() == "k8s_job"):
            git_docker_image = None

        self.patcher = CodexHelper(
            repo_path=self.repo_root,
            ai_key_path=str(ai_key_path),
            copy_repo=False,                   # operate in-place for determinism
            codex_cli=self.codex_cli,
            codex_model=CODEX_ANALYSIS_MODEL,
            approval_mode=CODEX_APPROVAL_MODE,
            dangerous_bypass=codex_dangerous,
            sandbox_mode=codex_sandbox_mode,
            git_docker_image=git_docker_image,
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
                time.sleep(
                    _retry_backoff_seconds(
                        1,
                        base_env="SHERPA_DOCKER_DAEMON_WAIT_INTERVAL_SEC",
                        cap_env="SHERPA_DOCKER_DAEMON_WAIT_INTERVAL_SEC",
                        default_base=1.0,
                        default_cap=1.0,
                    )
                )

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
                backoff_s = _retry_backoff_seconds(
                    attempt,
                    base_env="SHERPA_DOCKER_BUILD_RETRY_BASE_SEC",
                    cap_env="SHERPA_DOCKER_BUILD_RETRY_MAX_SEC",
                    default_base=1.0,
                    default_cap=6.0,
                )
                print(f"[warn] docker build {reason}; retrying in {backoff_s:.0f}s ({attempt}/{max_tries})")
                time.sleep(backoff_s)
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
                "VCPKG_ROOT",
                "VCPKG_DEFAULT_TRIPLET",
                "VCPKG_INSTALLED_DIR",
                "CMAKE_TOOLCHAIN_FILE",
                "CMAKE_PREFIX_PATH",
                "LD_LIBRARY_PATH",
                "LIBRARY_PATH",
                "PKG_CONFIG_PATH",
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
        dep_rel = (FUZZ_SYSTEM_PACKAGES_FILE or "fuzz/system_packages.txt").replace("\\", "/").strip("/")
        translated_cmd = self._wrap_exec_with_runtime_prelude(
            translated_for_exec,
            dep_file=f"/work/{dep_rel}",
            dep_log_prefix="docker/deps",
        )

        docker_cmd.append(self.docker_image)
        docker_cmd += translated_cmd
        return docker_cmd

    @staticmethod
    def _is_build_entry_arg(a: str) -> bool:
        norm = (a or "").strip().replace("\\", "/")
        if norm in {"build.py", "./build.py", "fuzz/build.py", "build.sh", "./build.sh", "fuzz/build.sh"}:
            return True
        return norm.endswith("/build.py") or norm.endswith("/build.sh")

    def _sanitize_build_py_for_source_build_collision(self) -> bool:
        """Prevent generated build scripts from deleting source-controlled build/ trees.

        Some upstream repos keep real files under REPO_ROOT/build/ (for example,
        build/version and build/cmake/*). If fuzz/build.py reuses REPO_ROOT/build as
        a scratch dir and cleans it, it destroys tracked source files before CMake runs.
        We proactively rewrite that pattern to an isolated fuzz/build-work directory.
        """

        fuzz_dir = Path(getattr(self, "fuzz_dir", self.repo_root / FUZZ_DIR))
        build_py = fuzz_dir / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        uses_repo_build = (
            "BUILD_DIR = REPO_ROOT / \"build\"" in text
            or "BUILD_DIR=REPO_ROOT / \"build\"" in text
            or "BUILD_DIR = REPO_ROOT/'build'" in text
        )
        destructive_clean = ("shutil.rmtree(BUILD_DIR" in text or "rm -rf \"$BUILD_DIR\"" in text)
        if not (uses_repo_build and destructive_clean):
            return False

        new_text = text
        if "BUILD_DIR = REPO_ROOT / \"build\"" in new_text:
            new_text = new_text.replace(
                "BUILD_DIR = REPO_ROOT / \"build\"",
                "BUILD_DIR = REPO_ROOT / \"fuzz\" / \"build-work\"",
            )
        if "BUILD_DIR=REPO_ROOT / \"build\"" in new_text:
            new_text = new_text.replace(
                "BUILD_DIR=REPO_ROOT / \"build\"",
                "BUILD_DIR=REPO_ROOT / \"fuzz\" / \"build-work\"",
            )
        if "BUILD_DIR = REPO_ROOT/'build'" in new_text:
            new_text = new_text.replace(
                "BUILD_DIR = REPO_ROOT/'build'",
                "BUILD_DIR = REPO_ROOT/'fuzz'/'build-work'",
            )

        if new_text == text:
            return False
        try:
            build_py.write_text(new_text, encoding="utf-8", errors="replace")
            print("[*] build preflight: rewrote BUILD_DIR to fuzz/build-work to avoid source build/ collision")
            return True
        except Exception:
            return False

    def _sanitize_build_py_for_non_root_install(self) -> bool:
        """Prevent generated build scripts from installing into system directories.

        Runtime containers run as non-root by default, so any install path that
        writes to `/usr/local` (or similar system prefixes) is fragile and will
        fail with permission denied. Force generated build scripts to link from
        build tree artifacts instead of running install steps.
        """

        fuzz_dir = Path(getattr(self, "fuzz_dir", self.repo_root / FUZZ_DIR))
        build_py = fuzz_dir / "build.py"
        if not build_py.is_file():
            return False
        try:
            text = build_py.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        new_text = text
        # Turn on-tree install toggles off.
        new_text = re.sub(
            r"(-DENABLE_INSTALL=)(ON|on|TRUE|True|1)\b",
            r"\1OFF",
            new_text,
        )
        # Rewrite explicit install targets to normal build targets.
        replacements = [
            ("'--target', 'install'", "'--target', 'all'"),
            ('"--target", "install"', '"--target", "all"'),
            ("'--target','install'", "'--target','all'"),
            ('"--target","install"', '"--target","all"'),
            ("'--install'", "'--build'"),
            ('"--install"', '"--build"'),
            ("cmake --install ", "cmake --build "),
        ]
        for old, new in replacements:
            if old in new_text:
                new_text = new_text.replace(old, new)

        if new_text == text:
            return False
        try:
            build_py.write_text(new_text, encoding="utf-8", errors="replace")
            print("[*] build preflight: disabled install step for non-root runtime")
            return True
        except Exception:
            return False

    @staticmethod
    def _vcpkg_triplet() -> str:
        raw = (os.environ.get("SHERPA_VCPKG_TRIPLET") or "").strip()
        if raw:
            return raw
        arch = (os.environ.get("TARGETARCH") or "").strip().lower()
        if not arch:
            arch = (os.uname().machine or "").strip().lower()
        if arch in {"amd64", "x86_64"}:
            return "x64-linux"
        if arch in {"arm64", "aarch64"}:
            return "arm64-linux"
        return "x64-linux"

    def _declared_vcpkg_ports(self, *, repo_root: Optional[Path] = None) -> list[str]:
        def _normalize_port(raw: str) -> str:
            token = raw.strip().lower()
            if not token:
                return ""
            if not re.fullmatch(r"[a-z0-9][a-z0-9+._-]*", token):
                return ""
            return VCPKG_PORT_ALIASES.get(token, token)
        rr = Path(repo_root or self.repo_root)
        dep_rel = (FUZZ_SYSTEM_PACKAGES_FILE or "fuzz/system_packages.txt").replace("\\", "/").strip("/")
        dep_file = rr / dep_rel
        if not dep_file.is_file():
            return []
        ports: list[str] = []
        seen: set[str] = set()
        try:
            for raw_line in dep_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.split("#", 1)[0].strip().lower()
                norm = _normalize_port(line)
                if not norm:
                    continue
                if norm in seen:
                    continue
                seen.add(norm)
                ports.append(norm)
        except Exception:
            return []
        return ports

    def _compose_vcpkg_runtime_env(self, base_env: Optional[Dict[str, str]] = None, *, repo_root: Optional[Path] = None) -> Dict[str, str]:
        env = dict(base_env or os.environ.copy())
        rr = Path(repo_root or self.repo_root)
        vcpkg_root = rr / VCPKG_REPO_DIR
        installed_root = rr / VCPKG_INSTALLED_DIR
        triplet = self._vcpkg_triplet()
        toolchain = vcpkg_root / "scripts" / "buildsystems" / "vcpkg.cmake"
        declared_ports = self._declared_vcpkg_ports(repo_root=rr)

        env["VCPKG_ROOT"] = str(vcpkg_root)
        env.setdefault("VCPKG_DEFAULT_TRIPLET", triplet)
        env["VCPKG_INSTALLED_DIR"] = str(installed_root)
        if not declared_ports or not toolchain.is_file():
            env.pop("CMAKE_TOOLCHAIN_FILE", None)
            return env
        env["CMAKE_TOOLCHAIN_FILE"] = str(toolchain)

        install_triplet = installed_root / triplet
        include_dir = install_triplet / "include"
        lib_dir = install_triplet / "lib"
        dbg_lib_dir = install_triplet / "debug" / "lib"
        pkgconfig_dir = lib_dir / "pkgconfig"

        def _prepend_env_path(key: str, values: Sequence[Path]) -> None:
            existing = (env.get(key) or "").strip()
            merged: List[str] = []
            seen: set[str] = set()
            for p in values:
                txt = str(p)
                if not txt or txt in seen:
                    continue
                seen.add(txt)
                merged.append(txt)
            if existing:
                for item in existing.split(":"):
                    it = item.strip()
                    if it and it not in seen:
                        seen.add(it)
                        merged.append(it)
            env[key] = ":".join(merged)

        _prepend_env_path("CMAKE_PREFIX_PATH", [install_triplet])
        _prepend_env_path("C_INCLUDE_PATH", [include_dir])
        _prepend_env_path("CPLUS_INCLUDE_PATH", [include_dir])
        _prepend_env_path("CPATH", [include_dir])
        _prepend_env_path("LIBRARY_PATH", [lib_dir, dbg_lib_dir])
        _prepend_env_path("LD_LIBRARY_PATH", [lib_dir, dbg_lib_dir])
        _prepend_env_path("PKG_CONFIG_PATH", [pkgconfig_dir])
        return env

    def _build_system_dep_setup(self, dep_file: str, *, log_prefix: str) -> str:
        return textwrap.dedent(
            f"""
            dep_file={shlex.quote(dep_file)}
            triplet={shlex.quote(self._vcpkg_triplet())}
            repo_root="$(cd "$(dirname "$dep_file")/.." && pwd -P)"
            vcpkg_root="$repo_root/{VCPKG_REPO_DIR}"
            vcpkg_installed="$repo_root/{VCPKG_INSTALLED_DIR}"

            pkgs=""
            if [ -f "$dep_file" ]; then
                while IFS= read -r line || [ -n "$line" ]; do
                    line="${{line%%#*}}"
                    line="$(printf '%s' "$line" | tr -d '\\r' | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
                    [ -n "$line" ] || continue
                    if ! printf '%s' "$line" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9+._-]*$'; then
                        echo "[warn] ({log_prefix}) skip invalid package token: $line"
                        continue
                    fi
                    pkg="$(printf '%s' "$line" | tr '[:upper:]' '[:lower:]')"
                    case "$pkg" in
                        z) mapped="zlib" ;;
                        bz2) mapped="bzip2" ;;
                        lzma|xz) mapped="liblzma" ;;
                        ssl|crypto|libssl|libcrypto) mapped="openssl" ;;
                        xml2|libxml) mapped="libxml2" ;;
                        *) mapped="$pkg" ;;
                    esac
                    if [ "$mapped" != "$pkg" ]; then
                        echo "[warn] ({log_prefix}) normalized package token '$pkg' -> '$mapped' (vcpkg port)"
                    fi
                    case " $pkgs " in
                        *" $mapped "*) ;;
                        *) pkgs="$pkgs $mapped" ;;
                    esac
                done < "$dep_file"
                if [ -n "$pkgs" ]; then
                    {{
                        echo "# Auto-normalized to vcpkg port names."
                        for p in $pkgs; do
                            echo "$p"
                        done
                    }} > "$dep_file" || true
                fi
            fi

            if [ -n "$pkgs" ]; then
                export VCPKG_ROOT="$vcpkg_root"
                export VCPKG_DEFAULT_TRIPLET="$triplet"
                export VCPKG_INSTALLED_DIR="$vcpkg_installed"
                export CMAKE_TOOLCHAIN_FILE="$vcpkg_root/scripts/buildsystems/vcpkg.cmake"
                export CMAKE_PREFIX_PATH="$vcpkg_installed/$triplet${{CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}}"
                export C_INCLUDE_PATH="$vcpkg_installed/$triplet/include${{C_INCLUDE_PATH:+:$C_INCLUDE_PATH}}"
                export CPLUS_INCLUDE_PATH="$vcpkg_installed/$triplet/include${{CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}}"
                export CPATH="$vcpkg_installed/$triplet/include${{CPATH:+:$CPATH}}"
                export LIBRARY_PATH="$vcpkg_installed/$triplet/lib:$vcpkg_installed/$triplet/debug/lib${{LIBRARY_PATH:+:$LIBRARY_PATH}}"
                export LD_LIBRARY_PATH="$vcpkg_installed/$triplet/lib:$vcpkg_installed/$triplet/debug/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
                export PKG_CONFIG_PATH="$vcpkg_installed/$triplet/lib/pkgconfig${{PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}}"
                shared_downloads_default="/shared/tmp/vcpkg-downloads"
                configured_downloads="${{SHERPA_VCPKG_DOWNLOADS_DIR:-$shared_downloads_default}}"
                if mkdir -p "$configured_downloads" 2>/dev/null; then
                    export VCPKG_DOWNLOADS="$configured_downloads"
                else
                    export VCPKG_DOWNLOADS="$repo_root/.vcpkg-downloads"
                    mkdir -p "$VCPKG_DOWNLOADS" || true
                fi

                # Keep vcpkg asset downloads from stalling on first github attempt:
                # install a local curl wrapper with mirror-first URL rewrite + short connect timeout.
                if command -v curl >/dev/null 2>&1; then
                    real_curl="$(command -v curl)"
                    shim_bin="$repo_root/.sherpa-bin"
                    mkdir -p "$shim_bin" || true
                    cat > "$shim_bin/curl" <<'EOF'
#!/bin/sh
set -u
real_curl="${{SHERPA_REAL_CURL_BIN:-/usr/bin/curl}}"
if [ ! -x "$real_curl" ]; then
  real_curl="$(command -v curl 2>/dev/null || true)"
fi
if [ -z "$real_curl" ]; then
  echo "[sherpa/curl] real curl not found" >&2
  exit 127
fi

argc=$#
if [ "$argc" -le 0 ]; then
  exec "$real_curl"
fi

last_arg=""
for a in "$@"; do
  last_arg="$a"
done
case "$last_arg" in
  http://*|https://*) orig_url="$last_arg" ;;
  *) exec "$real_curl" --connect-timeout "${{SHERPA_VCPKG_CURL_CONNECT_TIMEOUT_SEC:-8}}" --max-time "${{SHERPA_VCPKG_CURL_MAX_TIME_SEC:-90}}" --retry-delay 0 "$@" ;;
esac

if [ "$argc" -eq 1 ]; then
  base_eval="set --"
else
  i=1
  base_eval="set --"
  for arg in "$@"; do
    if [ "$i" -ge "$argc" ]; then
      break
    fi
    esc=$(printf "%s" "$arg" | sed "s/'/'\\\\''/g")
    base_eval="$base_eval '$esc'"
    i=$((i + 1))
  done
fi

case "$orig_url" in
  https://github.com/*)
    candidates="https://ghfast.top/$orig_url https://ghproxy.net/$orig_url $orig_url"
    for u in $candidates; do
      eval "$base_eval '$u'"
      "$real_curl" \
        --connect-timeout "${{SHERPA_VCPKG_CURL_CONNECT_TIMEOUT_SEC:-8}}" \
        --max-time "${{SHERPA_VCPKG_CURL_MAX_TIME_SEC:-90}}" \
        --retry-delay 0 \
        "$@"
      rc=$?
      if [ $rc -eq 0 ]; then
        exit 0
      fi
    done
    exit 56
    ;;
  *)
    exec "$real_curl" --connect-timeout "${{SHERPA_VCPKG_CURL_CONNECT_TIMEOUT_SEC:-8}}" --max-time "${{SHERPA_VCPKG_CURL_MAX_TIME_SEC:-90}}" --retry-delay 0 "$@"
    ;;
esac
EOF
                    chmod +x "$shim_bin/curl" || true
                    export SHERPA_REAL_CURL_BIN="$real_curl"
                    export PATH="$shim_bin:$PATH"
                fi

                if [ ! -x "$vcpkg_root/vcpkg" ]; then
                    if [ ! -d "$vcpkg_root/.git" ]; then
                        if [ -d /opt/vcpkg-template/.git ]; then
                            echo "[*] ({log_prefix}) seeding vcpkg from image template"
                            if ! cp -a /opt/vcpkg-template "$vcpkg_root"; then
                                echo "[warn] ({log_prefix}) failed to copy /opt/vcpkg-template"
                            fi
                        fi
                    fi
                    if [ ! -d "$vcpkg_root/.git" ]; then
                        vcpkg_git_bin="${{SHERPA_VCPKG_GIT_BIN:-git}}"
                        if command -v "$vcpkg_git_bin" >/dev/null 2>&1; then
                            echo "[*] ({log_prefix}) cloning vcpkg into $vcpkg_root"
                            clone_urls="https://ghfast.top/https://github.com/microsoft/vcpkg https://ghproxy.net/https://github.com/microsoft/vcpkg https://github.com/microsoft/vcpkg"
                            cloned_ok=0
                            for u in $clone_urls; do
                                echo "[*] ({log_prefix}) trying vcpkg source: $u"
                                if "$vcpkg_git_bin" clone --depth 1 "$u" "$vcpkg_root"; then
                                    cloned_ok=1
                                    break
                                fi
                                rm -rf "$vcpkg_root"
                            done
                            if [ "$cloned_ok" -ne 1 ]; then
                                echo "[warn] ({log_prefix}) unable to clone vcpkg from all configured sources"
                            fi
                        else
                            echo "[warn] ({log_prefix}) git is missing; cannot bootstrap vcpkg"
                        fi
                    fi
                    if [ -x "$vcpkg_root/bootstrap-vcpkg.sh" ]; then
                        if ! (cd "$vcpkg_root" && ./bootstrap-vcpkg.sh -disableMetrics); then
                            echo "[warn] ({log_prefix}) vcpkg bootstrap failed"
                        fi
                    fi
                fi

                if [ ! -x "$vcpkg_root/vcpkg" ]; then
                    echo "[error] ({log_prefix}) vcpkg unavailable while required ports are declared in $dep_file"
                    exit 86
                fi

                missing_pkgs=""
                for p in $pkgs; do
                    if "$vcpkg_root/vcpkg" list "$p:$triplet" 2>/dev/null | grep -Eq "^$p:$triplet\\s"; then
                        continue
                    fi
                    missing_pkgs="$missing_pkgs $p"
                done

                if [ -z "$missing_pkgs" ]; then
                    echo "[*] ({log_prefix}) all requested vcpkg ports already installed; skipping"
                else
                    echo "[*] ({log_prefix}) installing vcpkg ports from $dep_file:$missing_pkgs"
                    if ! "$vcpkg_root/vcpkg" install --triplet "$triplet" $missing_pkgs; then
                        echo "[error] ({log_prefix}) vcpkg install failed for:$missing_pkgs"
                        exit 87
                    fi
                fi
            fi

            if [ -n "$pkgs" ] && [ -f "$vcpkg_root/scripts/buildsystems/vcpkg.cmake" ]; then
                export VCPKG_ROOT="$vcpkg_root"
                export VCPKG_DEFAULT_TRIPLET="$triplet"
                export VCPKG_INSTALLED_DIR="$vcpkg_installed"
                export CMAKE_TOOLCHAIN_FILE="$vcpkg_root/scripts/buildsystems/vcpkg.cmake"
                export CMAKE_PREFIX_PATH="$vcpkg_installed/$triplet${{CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}}"
                export C_INCLUDE_PATH="$vcpkg_installed/$triplet/include${{C_INCLUDE_PATH:+:$C_INCLUDE_PATH}}"
                export CPLUS_INCLUDE_PATH="$vcpkg_installed/$triplet/include${{CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}}"
                export CPATH="$vcpkg_installed/$triplet/include${{CPATH:+:$CPATH}}"
                export LIBRARY_PATH="$vcpkg_installed/$triplet/lib:$vcpkg_installed/$triplet/debug/lib${{LIBRARY_PATH:+:$LIBRARY_PATH}}"
                export LD_LIBRARY_PATH="$vcpkg_installed/$triplet/lib:$vcpkg_installed/$triplet/debug/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
                export PKG_CONFIG_PATH="$vcpkg_installed/$triplet/lib/pkgconfig${{PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}}"
            fi

            if [ -n "$pkgs" ] && [ ! -f "$vcpkg_root/scripts/buildsystems/vcpkg.cmake" ]; then
                echo "[error] ({log_prefix}) missing vcpkg toolchain file: $vcpkg_root/scripts/buildsystems/vcpkg.cmake"
                exit 88
            fi
            """
        ).strip()

    def _wrap_exec_with_runtime_prelude(
        self,
        exec_args: Sequence[str],
        *,
        dep_file: str,
        dep_log_prefix: str,
    ) -> List[str]:
        artifacts_path = ""
        for a in exec_args:
            if str(a).startswith("-artifact_prefix="):
                artifacts_path = str(a).split("=", 1)[1]
                break

        dep_setup = ""
        should_autoinstall = any(self._is_build_entry_arg(str(a)) for a in exec_args)
        if should_autoinstall and _is_truthy_env("SHERPA_AUTO_INSTALL_SYSTEM_DEPS", True):
            dep_setup = self._build_system_dep_setup(dep_file, log_prefix=dep_log_prefix)

        if not artifacts_path and not dep_setup:
            return list(exec_args)

        exec_cmd = " ".join(shlex.quote(str(a)) for a in exec_args)
        shell_parts: List[str] = ["set -u"]
        if artifacts_path:
            shell_parts.append(f"mkdir -p {shlex.quote(artifacts_path)}")
        if dep_setup:
            shell_parts.append(dep_setup)
        shell_parts.append(f"exec {exec_cmd}")
        return ["sh", "-lc", "\n".join(shell_parts)]

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
            Follow global policy from `./.git/sherpa-opencode/opencode_policy.md` when present.
            Goal: produce `{FUZZ_DIR}/PLAN.md` and strict-schema `{FUZZ_DIR}/targets.json`.
            Keep runtime-viable/public targets first. Avoid helper-only targets.
            targets.json requirements:
            - non-empty JSON array
            - each item includes non-empty `name`, `api`, `lang`, `target_type`, `seed_profile`
            - forbidden: `name = LLVMFuzzerTestOneInput`
            Do NOT run build/execute commands; read-only inspection commands are allowed.
            MANDATORY: write `{FUZZ_DIR}/PLAN.md` into `./done`.
            """
        ).strip()

        stdout = self.patcher.run_codex_command(
            instructions,
            stage_skill="plan",
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
            Follow global policy from `./.git/sherpa-opencode/opencode_policy.md` when present.
            Goal: synthesize a complete fuzz scaffold under `{FUZZ_DIR}`.
            Required outputs:
            - harness source file(s)
            - `fuzz/build.py` or `fuzz/build.sh`
            - `fuzz/repo_understanding.json`
            - `fuzz/build_strategy.json`
            - `fuzz/build_runtime_facts.json`
            - `fuzz/README.md`
            Build constraints:
            - keep `DEFAULT_CMAKE_ARGS` with `-DENABLE_TEST=OFF` and `-DENABLE_INSTALL=OFF`
            - do not hardcode a single artifact path; use discovery
            - if external deps are required, write canonical vcpkg port names to `{FUZZ_SYSTEM_PACKAGES_FILE}`
            Keep selected/observed target alignment and record drift reasons.
            Do NOT run build/execute commands; read-only inspection commands are allowed.
            MANDATORY: write `{FUZZ_OUT_DIR}` into `./done`.
            """
        ).strip()

        context = (
            "=== fuzz/PLAN.md ===\n" + plan_text +
            "\n\n=== fuzz/targets.json ===\n" + targets_text
        )
        stdout = self.patcher.run_codex_command(
            instructions,
            additional_context=context,
            stage_skill="synthesize",
            timeout=timeout,
            max_attempts=1,
            max_cli_retries=_workflow_opencode_cli_retries(),
            idle_timeout_override=_synthesize_opencode_idle_timeout_sec(),
            activity_watch_paths=_synthesize_activity_watch_paths(),
        )
        if stdout is None:
            partial_outputs = False
            try:
                for p in self.fuzz_dir.rglob("*"):
                    if not p.is_file():
                        continue
                    rel = p.relative_to(self.fuzz_dir).as_posix()
                    if rel.startswith("out/") or rel.startswith("corpus/"):
                        continue
                    if (
                        p.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".java"}
                        or rel in {"build.py", "build.sh", "README.md", "system_packages.txt"}
                    ):
                        partial_outputs = True
                        break
            except Exception:
                partial_outputs = False
            if partial_outputs:
                print("[warn] Codex exited without done sentinel, but partial synth outputs exist; deferring completeness check")
                return
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

    def _resolve_seed_target_metadata(self, fuzzer_name: str, harness_text: str) -> tuple[str, str]:
        observed = self._resolve_observed_target(fuzzer_name, harness_text)
        if observed:
            observed_markers = "\n".join(
                [
                    str(observed.get("observed_target_api") or ""),
                    str(observed.get("selected_target_api") or ""),
                    str(observed.get("observed_harness") or ""),
                    harness_text,
                ]
            )
            target_type = str(observed.get("target_type") or "").strip().lower()
            if target_type not in ALLOWED_TARGET_TYPES:
                target_type = _infer_target_type(fuzzer_name, observed_markers)
            seed_profile = str(observed.get("seed_profile") or "").strip().lower()
            if seed_profile not in ALLOWED_SEED_PROFILES:
                seed_profile = self._infer_seed_profile(fuzzer_name, observed_markers, target_type)
            if target_type in ALLOWED_TARGET_TYPES and seed_profile in ALLOWED_SEED_PROFILES:
                return target_type, seed_profile

        selected = self._resolve_selected_target(fuzzer_name, harness_text)
        if selected:
            target_type = str(selected.get("target_type") or "").strip().lower()
            seed_profile = str(selected.get("seed_profile") or "").strip().lower()
            if target_type in ALLOWED_TARGET_TYPES and seed_profile in ALLOWED_SEED_PROFILES:
                return target_type, seed_profile

        targets_path = self.fuzz_dir / "targets.json"
        candidates: list[dict[str, object]] = []
        try:
            if targets_path.is_file():
                raw = json.loads(targets_path.read_text(encoding="utf-8", errors="replace"))
                if isinstance(raw, list):
                    candidates = [item for item in raw if isinstance(item, dict)]
        except Exception:
            candidates = []

        normalized_fuzzer = re.sub(r"_fuzz(?:er)?$", "", fuzzer_name.lower())
        for item in candidates:
            name = str(item.get("name") or "").strip().lower()
            api = str(item.get("api") or "").strip().lower()
            target_type = str(item.get("target_type") or "").strip().lower()
            seed_profile = str(item.get("seed_profile") or "").strip().lower()
            if target_type not in ALLOWED_TARGET_TYPES:
                continue
            if name and (name in normalized_fuzzer or normalized_fuzzer in name):
                return target_type, (seed_profile if seed_profile in ALLOWED_SEED_PROFILES else self._infer_seed_profile(fuzzer_name, harness_text, target_type))
            if api and (api in normalized_fuzzer or normalized_fuzzer in api):
                return target_type, (seed_profile if seed_profile in ALLOWED_SEED_PROFILES else self._infer_seed_profile(fuzzer_name, harness_text, target_type))

        if len(candidates) == 1:
            target_type = str(candidates[0].get("target_type") or "").strip().lower()
            if target_type in ALLOWED_TARGET_TYPES:
                seed_profile = str(candidates[0].get("seed_profile") or "").strip().lower()
                return target_type, (seed_profile if seed_profile in ALLOWED_SEED_PROFILES else self._infer_seed_profile(fuzzer_name, harness_text, target_type))

        target_type = _infer_target_type(fuzzer_name, harness_text)
        return target_type, self._infer_seed_profile(fuzzer_name, harness_text, target_type)

    def _load_selected_targets_doc(self) -> list[dict[str, object]]:
        path = self.fuzz_dir / "selected_targets.json"
        try:
            if not path.is_file():
                return []
            raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        return [item for item in raw if isinstance(item, dict)]

    def _load_observed_target_doc(self) -> dict[str, object]:
        path = self.fuzz_dir / "observed_target.json"
        try:
            if not path.is_file():
                return {}
            raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return {}
        return dict(raw) if isinstance(raw, dict) else {}

    def _resolve_observed_target(self, fuzzer_name: str, harness_text: str) -> dict[str, object]:
        doc = self._load_observed_target_doc()
        if not doc:
            return {}
        observed_api = str(doc.get("observed_target_api") or "").strip().lower()
        observed_harness = str(doc.get("observed_harness") or "").strip().lower()
        selected_api = str(doc.get("selected_target_api") or "").strip().lower()
        normalized_fuzzer = re.sub(r"_fuzz(?:er)?$", "", fuzzer_name.lower())
        lowered_harness = harness_text.lower()
        harness_name = Path(observed_harness).stem.lower() if observed_harness else ""
        if observed_harness and harness_name and (harness_name == fuzzer_name.lower() or harness_name == normalized_fuzzer):
            return dict(doc)
        if observed_api and observed_api in lowered_harness:
            return dict(doc)
        if harness_name and (harness_name in normalized_fuzzer or normalized_fuzzer in harness_name):
            return dict(doc)
        if selected_api and selected_api in lowered_harness:
            return dict(doc)
        return dict(doc) if len(doc) > 0 else {}

    def _resolve_selected_target(self, fuzzer_name: str, harness_text: str) -> dict[str, object]:
        doc = self._load_selected_targets_doc()
        if not doc:
            return {}
        normalized_fuzzer = re.sub(r"_fuzz(?:er)?$", "", fuzzer_name.lower())
        lowered_harness = harness_text.lower()
        for item in doc:
            api = str(item.get("api") or "").strip().lower()
            target_name = str(item.get("target_name") or item.get("name") or "").strip().lower()
            wrapper_name = str(item.get("wrapper_fuzzer_name") or "").strip().lower()
            if wrapper_name and wrapper_name == fuzzer_name.lower():
                return dict(item)
            if target_name and (target_name in normalized_fuzzer or normalized_fuzzer in target_name):
                return dict(item)
            if api and (api in normalized_fuzzer or api in lowered_harness):
                return dict(item)
        return dict(doc[0]) if len(doc) == 1 else {}

    def _seed_family_coverage(self, corpus_dir: Path, required_families: list[str]) -> dict[str, object]:
        files = sorted(p for p in corpus_dir.iterdir() if p.is_file()) if corpus_dir.is_dir() else []
        covered: set[str] = set()
        family_examples: dict[str, list[str]] = {}
        for path in files:
            for family in _classify_seed_family(path):
                covered.add(family)
                family_examples.setdefault(family, [])
                if len(family_examples[family]) < 3:
                    family_examples[family].append(path.name)
        required = [x for x in required_families if x]
        missing = [x for x in required if x not in covered]
        return {
            "required": required,
            "covered": sorted(covered),
            "missing": missing,
            "family_examples": family_examples,
        }

    def _infer_seed_profile(self, fuzzer_name: str, harness_text: str, target_type: str) -> str:
        lowered = f"{fuzzer_name}\n{harness_text}".lower()
        if target_type == "parser":
            if any(tok in lowered for tok in ("arg_id", "argument id", "positional", "named arg", "named argument", "numeric", "number")):
                return "parser-numeric"
            if any(tok in lowered for tok in ("format", "replacement field", "specifier", "fmt::", "brace", "printf")):
                return "parser-format"
            if any(tok in lowered for tok in ("token", "lexer", "lex", "scan", "scanner")):
                return "parser-token"
            return "parser-structure"
        mapping = {
            "decoder": "decoder-binary",
            "archive": "archive-container",
            "serializer": "serializer-structured",
            "document": "document-text",
            "network": "network-message",
        }
        return mapping.get(target_type, "generic")

    def _seed_generation_guidance(self, target_type: str, seed_profile: str, fuzzer_name: str, harness_text: str) -> str:
        common = (
            "Seed goal: maximize early coverage, not readability. Mix valid, boundary, and malformed inputs. "
            "Prefer small files that drive distinct parser or state-machine behaviors."
        )
        guidance = {
            "parser-structure": (
                "Create seeds that exercise parser state transitions: empty/minimal input, valid structured input, "
                "deep nesting, alternate syntactic forms, truncated/incomplete forms, invalid tokens, long scalars/strings, "
                "duplicate keys or repeated sections, and if applicable directives/tags/anchors/aliases."
            ),
            "parser-token": (
                "Create token-oriented parser seeds: single token, repeated tokens, delimiter-only inputs, "
                "whitespace-prefixed tokens, unterminated tokens, malformed separator placement, and short malformed fragments."
            ),
            "parser-format": (
                "Create format-string parser seeds: bare text, single replacement fields, nested or mismatched braces, "
                "width/precision markers, invalid specifier suffixes, mixed named/positional fields, and truncated directives."
            ),
            "parser-numeric": (
                "Create numeric/token parser seeds: `0`, `1`, `42`, leading zeros, extremely long numeric ids, mixed alpha-numeric boundaries, "
                "truncated ids, invalid starter characters, separator-boundary tokens such as `:`, `}`, `]`, `,`, `{`, `[`, and high-byte cases."
            ),
            "decoder-binary": (
                "Create binary decoder seeds: shortest valid frame, maximal headers, truncated payloads, bad checksums/lengths, "
                "nested containers, malformed trailers, and magic-byte variations."
            ),
            "archive-container": (
                "Create archive/container seeds with a real-sample-first strategy: first ensure valid archive samples exist from "
                "repository sources (for example `contrib/oss-fuzz/corpus.zip`, `contrib/oss-fuzz/**`, `test/**`, `tests/**`), "
                "then add boundary or malformed variants only when needed for coverage growth. Prefer real archives over hand-crafted malformed bytes."
            ),
            "serializer-structured": (
                "Create structured serializer seeds: empty object, nested object, repeated fields, large strings, invalid tags, "
                "alternate encodings, and partially malformed structures."
            ),
            "document-text": (
                "Create text document seeds: minimal valid document, nested sections, metadata blocks, embedded markup, "
                "malformed closing markers, and truncated bodies."
            ),
            "network-message": (
                "Create packet/message seeds: minimal valid message, alternate message types, length mismatches, partial frames, "
                "repeated headers, malformed fields, and boundary numeric values."
            ),
            "generic": (
                "Create diverse seeds: empty input, smallest valid sample, largest small sample, malformed/truncated input, "
                "repeated delimiters, and boundary numeric/text values."
            ),
        }
        extra = guidance.get(seed_profile, guidance["generic"])
        yaml_hint = ""
        lowered = f"{fuzzer_name}\n{harness_text}".lower()
        if seed_profile == "parser-format" and _is_fmt_format_target(fuzzer_name, harness_text):
            extra = (
                "Create fmt-style format-string seeds by family bucket: replacement fields (`{}`), escaped braces (`{{`/`}}`), "
                "positional arguments (`{0}`), format specifiers (`{:x}`/`{:s}`), width/precision (`{:10.3f}`), fill/alignment (`{:_>8}`), "
                "type conversions, and malformed replacement fields (mismatched braces, truncated specs, mixed bad fields). "
                "Prefer textual UTF-8 seeds. Do not flood the corpus with random binary noise."
            )
        if any(tok in lowered for tok in ("yaml", "yml")):
            yaml_hint = (
                " Include YAML-specific cases: document markers (`---`/`...`), anchors and aliases, block scalars (`|`/`>`), "
                "flow collections (`[]`/`{}`), tags, directives, malformed indentation, and truncated nested mappings."
            )
        return (
            f"Target type for `{fuzzer_name}` is `{target_type}` and seed_profile is `{seed_profile}`. {common} {extra}"
            + yaml_hint
        )

    def _collect_repo_seed_examples(
        self,
        seed_profile: str,
        fuzzer_name: str,
        corpus_dir: Path,
        *,
        required_families: list[str] | None = None,
    ) -> tuple[list[Path], dict[str, Any]]:
        search_roots = [
            "contrib/oss-fuzz",
            "tests/examples/testdata",
            "tests",
            "examples",
            "regression-inputs",
            "testdata",
            "samples",
            "docs",
        ]
        structured_text_suffixes = {".txt", ".yaml", ".yml", ".json", ".xml", ".ini", ".cfg", ".conf", ".toml"}
        binary_suffixes = {".bin", ".dat", ".arc", ".zip", ".tar", ".gz", ".xz", ".png", ".jpg", ".jpeg", ".gif"}
        source_blacklist = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".html", ".md", ".rst", ".cmake", ".py", ".js", ".ts"}
        sample_dir_tokens = ("testdata", "tests", "examples", "regression", "samples", "corpus", "data", "inputs")
        selected: list[Path] = []
        seen_hashes: set[str] = set()
        accepted = 0
        rejected = 0
        lowered_name = fuzzer_name.lower()
        family_limits: dict[str, int] = {}
        required_set = set(required_families or [])
        max_seed_file = self._seed_max_file_bytes()
        imported_zip_count = 0
        imported_zip_rejected = 0

        if seed_profile == "archive-container":
            imported_zip_count, imported_zip_rejected = self._import_repo_corpus_zip(
                corpus_dir=corpus_dir,
                seen_hashes=seen_hashes,
                max_seed_file=max_seed_file,
                max_keep=12,
            )
            if imported_zip_count > 0:
                accepted += imported_zip_count

        def _under_sample_dir(path: Path) -> bool:
            parts = [part.lower() for part in path.parts]
            return any(tok in sample_dir_tokens for tok in parts)

        def _yaml_like_candidate(path: Path, size: int) -> bool:
            suffix = path.suffix.lower()
            if suffix not in {".yaml", ".yml", ".txt", ".in"}:
                return False
            if size > max_seed_file:
                return False
            try:
                snippet = path.read_text(encoding="utf-8", errors="replace")[:512].lower()
            except Exception:
                return False
            if any(tok in snippet for tok in ("---", "%yaml", "%tag", "&", "*", "[", "]", "{", "}")):
                return True
            if any(tok in path.name.lower() for tok in ("yaml", "tag", "anchor", "alias", "directive", "flow", "mapping", "scalar")):
                return True
            return False

        def _allow_path(path: Path, size: int) -> bool:
            nonlocal rejected
            suffix = path.suffix.lower()
            if suffix in source_blacklist:
                rejected += 1
                return False
            under_sample = _under_sample_dir(path)
            name_hint = path.name.lower()
            haystack = f"{name_hint} {lowered_name} {' '.join(part.lower() for part in path.parts)}"
            if seed_profile in {"parser-structure", "document-text", "serializer-structured"}:
                if suffix not in structured_text_suffixes:
                    rejected += 1
                    return False
                return True
            if seed_profile in {"decoder-binary", "archive-container"}:
                if suffix and suffix in binary_suffixes:
                    return True
                rejected += 1
                return False
            if seed_profile == "parser-format":
                if not under_sample or size > 4096:
                    rejected += 1
                    return False
                if suffix and suffix not in structured_text_suffixes and suffix not in {".fmt", ".in", ".tmpl"}:
                    rejected += 1
                    return False
                if _is_fmt_format_target(fuzzer_name, path.name, " ".join(path.parts)):
                    return True
                if not any(tok in haystack for tok in ("fmt", "format", "printf", "arg", "spec", "brace", "replacement")):
                    rejected += 1
                    return False
                return True
            if seed_profile == "parser-numeric":
                if not under_sample or size > 2048:
                    rejected += 1
                    return False
                if suffix and suffix not in structured_text_suffixes and suffix not in {".dat", ".in", ".txt"}:
                    rejected += 1
                    return False
                if not any(tok in haystack for tok in ("id", "arg", "number", "numeric", "token", "name", "index", "field")):
                    rejected += 1
                    return False
                return True
            if seed_profile == "parser-token":
                if not under_sample or size > 4096:
                    rejected += 1
                    return False
                if suffix and suffix not in structured_text_suffixes and suffix not in {".dat", ".in", ".txt"}:
                    rejected += 1
                    return False
                if any(tok in lowered_name for tok in ("yaml", "yml")) and _yaml_like_candidate(path, size):
                    return True
                if not any(tok in haystack for tok in ("token", "scan", "lex", "arg", "field", "name", "spec")):
                    rejected += 1
                    return False
                return True
            if seed_profile == "network-message":
                if suffix and suffix not in {".bin", ".dat", ".msg", ".pkt", ".txt", ".json", ".xml"}:
                    rejected += 1
                    return False
                return under_sample
            if seed_profile == "generic":
                if not under_sample:
                    rejected += 1
                    return False
                if size > max_seed_file:
                    rejected += 1
                    return False
                if suffix and suffix in source_blacklist:
                    rejected += 1
                    return False
                if suffix and suffix not in structured_text_suffixes and suffix not in binary_suffixes and suffix not in {".dat", ".in", ".raw"}:
                    rejected += 1
                    return False
                return True
            return False

        for rel_root in search_roots:
            root = self.repo_root / rel_root
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    size = path.stat().st_size
                except Exception:
                    rejected += 1
                    continue
                if size <= 0 or size > max_seed_file:
                    rejected += 1
                    continue
                if not _allow_path(path, size):
                    continue
                try:
                    data = path.read_bytes()
                except Exception:
                    rejected += 1
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen_hashes:
                    rejected += 1
                    continue
                seen_hashes.add(digest)
                dest = corpus_dir / f"repo_{len(selected)+1:02d}{path.suffix.lower()}"
                try:
                    dest.write_bytes(data)
                except Exception:
                    rejected += 1
                    continue
                selected.append(dest)
                accepted += 1
                for family in _classify_seed_family(dest):
                    family_limits[family] = family_limits.get(family, 0) + 1
                if len(selected) >= 12:
                    break
            if len(selected) >= 12:
                break
        if seed_profile == "archive-container" and len(selected) < 12:
            for suffix, data in self._default_archive_seed_samples():
                if len(selected) >= 12:
                    break
                size = len(data)
                if size <= 0 or size > max_seed_file:
                    rejected += 1
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen_hashes:
                    rejected += 1
                    continue
                dest = corpus_dir / f"repo_{len(selected)+1:02d}{suffix}"
                try:
                    dest.write_bytes(data)
                except Exception:
                    rejected += 1
                    continue
                seen_hashes.add(digest)
                selected.append(dest)
                accepted += 1
                for family in _classify_seed_family(dest):
                    family_limits[family] = family_limits.get(family, 0) + 1
        if imported_zip_rejected > 0:
            rejected += imported_zip_rejected
        return selected, {
            "sources": ["repo_examples"] if (selected or imported_zip_count > 0) else [],
            "accepted_count": accepted,
            "rejected_count": rejected,
            "filtered": True,
            "family_limits": family_limits,
            "required_families": sorted(required_set),
            "imported_corpus_zip_count": imported_zip_count,
            "imported_corpus_zip_rejected": imported_zip_rejected,
        }

    def _default_archive_seed_samples(self) -> list[tuple[str, bytes]]:
        samples: list[tuple[str, bytes]] = []
        payload = b"seed\n"

        try:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("a.txt", payload)
            samples.append((".zip", zip_buf.getvalue()))
        except Exception:
            pass

        try:
            tar_buf = io.BytesIO()
            info = tarfile.TarInfo(name="a.txt")
            info.size = len(payload)
            with tarfile.open(fileobj=tar_buf, mode="w") as tf:
                tf.addfile(info, io.BytesIO(payload))
            samples.append((".tar", tar_buf.getvalue()))
        except Exception:
            pass

        try:
            samples.append((".gz", gzip.compress(payload)))
        except Exception:
            pass

        try:
            samples.append((".bz2", bz2.compress(payload)))
        except Exception:
            pass

        try:
            samples.append((".xz", lzma.compress(payload)))
        except Exception:
            pass

        return samples

    def _archive_seed_is_semantically_valid(self, path: Path, data: bytes | None = None) -> bool:
        suffix = path.suffix.lower()
        raw = data
        if raw is None:
            try:
                raw = path.read_bytes()
            except Exception:
                return False
        if not raw:
            return False
        inferred_suffix = suffix
        if inferred_suffix not in {".zip", ".tar", ".gz", ".bz2", ".xz"}:
            if raw.startswith(b"PK"):
                inferred_suffix = ".zip"
            elif len(raw) > 264 and raw[257:262] == b"ustar":
                inferred_suffix = ".tar"
            elif raw.startswith(b"\x1f\x8b"):
                inferred_suffix = ".gz"
            elif raw.startswith(b"BZh"):
                inferred_suffix = ".bz2"
            elif raw.startswith(b"\xFD7zXZ\x00"):
                inferred_suffix = ".xz"
            else:
                return False
        try:
            if inferred_suffix == ".zip":
                if len(raw) < 4 or raw[:2] != b"PK":
                    return False
                with zipfile.ZipFile(io.BytesIO(raw), mode="r") as zf:
                    return len(zf.namelist()) > 0
            if inferred_suffix == ".tar":
                with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
                    members = tf.getmembers()
                    return len(members) > 0
            if inferred_suffix == ".gz":
                _ = gzip.decompress(raw)
                return True
            if inferred_suffix == ".bz2":
                _ = bz2.decompress(raw)
                return True
            if inferred_suffix == ".xz":
                _ = lzma.decompress(raw)
                return True
        except Exception:
            return False
        return True

    def _is_magic_only_archive_seed(self, data: bytes | None) -> bool:
        raw = data or b""
        if not raw:
            return False
        signature_threshold = 24
        signatures = (
            b"PK\x03\x04",
            b"PK\x05\x06",
            b"PK\x07\x08",
            b"\x1f\x8b",
            b"BZh",
            b"\xFD7zXZ\x00",
            b"7z\xBC\xAF\x27\x1C",
            b"Rar!\x1A\x07",
        )
        return len(raw) <= signature_threshold and any(raw.startswith(sig) for sig in signatures)

    def _is_archive_malformed_candidate(self, path: Path, data: bytes | None = None) -> bool:
        raw = data
        if raw is None:
            try:
                raw = path.read_bytes()
            except Exception:
                raw = b""
        if self._is_magic_only_archive_seed(raw):
            return True
        suffix = path.suffix.lower()
        if suffix in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".cpio"}:
            return not self._archive_seed_is_semantically_valid(path, raw)
        name = path.name.lower()
        malformed_tokens = ("malformed", "invalid", "corrupt", "broken", "trunc", "magic_only")
        if any(tok in name for tok in malformed_tokens) and len(raw or b"") <= 256:
            return True
        return False

    def _import_repo_corpus_zip(
        self,
        *,
        corpus_dir: Path,
        seen_hashes: set[str],
        max_seed_file: int,
        max_keep: int,
    ) -> tuple[int, int]:
        zip_path = self.repo_root / "contrib" / "oss-fuzz" / "corpus.zip"
        if not zip_path.is_file():
            return 0, 0
        imported = 0
        rejected = 0
        try:
            with zipfile.ZipFile(zip_path, mode="r") as zf:
                for info in zf.infolist():
                    if imported >= max_keep:
                        break
                    if info.is_dir():
                        continue
                    if info.file_size <= 0 or info.file_size > max_seed_file:
                        rejected += 1
                        continue
                    try:
                        data = zf.read(info.filename)
                    except Exception:
                        rejected += 1
                        continue
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in seen_hashes:
                        rejected += 1
                        continue
                    ext = Path(info.filename).suffix.lower()
                    if ext not in {".zip", ".tar", ".gz", ".bz2", ".xz", ".bin", ".dat"}:
                        ext = ".bin"
                    dest = corpus_dir / f"repo_{imported+1:02d}{ext}"
                    try:
                        dest.write_bytes(data)
                    except Exception:
                        rejected += 1
                        continue
                    seen_hashes.add(digest)
                    imported += 1
        except Exception:
            return 0, 0
        return imported, rejected

    def _summarize_seed_corpus(self, corpus_dir: Path) -> str:
        files = sorted(p for p in corpus_dir.iterdir() if p.is_file()) if corpus_dir.is_dir() else []
        if not files:
            return "No existing corpus files."
        parts: list[str] = []
        for path in files[:12]:
            try:
                size = path.stat().st_size
            except Exception:
                size = 0
            parts.append(f"- {path.name} ({size} bytes)")
        return "Existing corpus files:\n" + "\n".join(parts)

    def _infer_seed_gaps(self, seed_profile: str, corpus_dir: Path) -> str:
        names = " ".join(p.name.lower() for p in corpus_dir.iterdir() if p.is_file()) if corpus_dir.is_dir() else ""
        gaps: list[str] = []
        if seed_profile == "parser-structure":
            if not any(tok in names for tok in ("trunc", "invalid", "malformed")):
                gaps.append("missing malformed/truncated parser cases")
            if not any(tok in names for tok in ("deep", "nested", "alias", "anchor", "flow")):
                gaps.append("missing deep nesting / alternate syntax forms")
        elif seed_profile == "parser-numeric":
            gaps.extend([
                "missing long numeric ids and leading-zero cases",
                "missing separator-boundary and non-ASCII cases",
            ])
        elif seed_profile == "parser-format":
            if any(tok in names for tok in ("fmt", "format", "print", "println")):
                gaps.extend([
                    "missing replacement field / escaped brace coverage",
                    "missing width/precision, fill/align, and type conversion cases",
                    "missing malformed replacement field cases",
                ])
            else:
                gaps.extend([
                    "missing mismatched brace and truncated directive cases",
                    "missing mixed named/positional field cases",
                ])
        elif seed_profile == "parser-token":
            gaps.extend([
                "missing delimiter-only and unterminated token cases",
                "missing malformed separator placement cases",
            ])
        elif seed_profile == "decoder-binary":
            gaps.append("missing malformed length/checksum and truncated binary frames")
        elif seed_profile == "archive-container":
            gaps.append("ensure at least one valid archive sample exists first")
        return "; ".join(gaps[:4]) or "cover valid, malformed, truncation, and boundary-value cases"

    def _seed_exploration_path(self, fuzzer_name: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(fuzzer_name or "").strip()) or "seed"
        return self.fuzz_dir / f"seed_exploration_{safe_name}.json"

    def _seed_check_path(self, fuzzer_name: str) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(fuzzer_name or "").strip()) or "seed"
        return self.fuzz_dir / f"seed_check_{safe_name}.json"

    def _seed_max_file_bytes(self) -> int:
        raw = (os.environ.get("SHERPA_SEED_MAX_FILE_BYTES") or "8192").strip()
        try:
            return max(512, min(int(raw), 262144))
        except Exception:
            return 8192

    def _seed_radamsa_max_file_bytes(self) -> int:
        raw = (os.environ.get("SHERPA_RADAMSA_MAX_FILE_BYTES") or "4096").strip()
        if not raw:
            return self._seed_max_file_bytes()
        try:
            return max(512, min(int(raw), 262144))
        except Exception:
            return self._seed_max_file_bytes()

    def _seed_archive_validate_enabled(self) -> bool:
        raw = (os.environ.get("SHERPA_SEED_ARCHIVE_VALIDATE_ENABLED") or "1").strip().lower()
        if not raw:
            return True
        return raw in {"1", "true", "yes", "on"}

    def _seed_archive_min_valid_ratio(self) -> float:
        raw = (os.environ.get("SHERPA_SEED_ARCHIVE_MIN_VALID_RATIO") or "0.60").strip()
        try:
            ratio = float(raw)
        except Exception:
            return 0.60
        return max(0.10, min(ratio, 1.0))

    def _seed_archive_max_malformed_ratio(self) -> float:
        raw = (os.environ.get("SHERPA_SEED_ARCHIVE_MAX_MALFORMED_RATIO") or "0.30").strip()
        try:
            ratio = float(raw)
        except Exception:
            return 0.30
        return max(0.05, min(ratio, 0.90))

    def _seed_corpus_min_per_target(self) -> int:
        raw = (os.environ.get("SHERPA_SEED_CORPUS_MIN_PER_TARGET") or "16").strip()
        try:
            return max(4, min(int(raw), 128))
        except Exception:
            return 16

    def _seed_max_total_bytes(self) -> int:
        raw = (os.environ.get("SHERPA_SEED_MAX_TOTAL_BYTES") or "524288").strip()
        try:
            return max(16384, min(int(raw), 8 * 1024 * 1024))
        except Exception:
            return 524288

    def _run_radamsa_bootstrap(self, corpus_dir: Path, *, seed_profile: str = "") -> int:
        radamsa = which("radamsa")
        if not radamsa:
            print("[warn] radamsa not found; skipping corpus mutation")
            return 0
        base_files = [p for p in sorted(corpus_dir.iterdir()) if p.is_file()][:12]
        if seed_profile == "archive-container":
            base_files = [
                p for p in base_files
                if p.suffix.lower() in {".zip", ".tar", ".gz", ".bz2", ".xz"} and self._archive_seed_is_semantically_valid(p)
            ]
        if not base_files:
            return 0
        created = 0
        attempted = 0
        invalid = 0
        total_bytes = sum((p.stat().st_size for p in corpus_dir.iterdir() if p.is_file()), 0)
        max_total = self._seed_max_total_bytes()
        max_file = self._seed_radamsa_max_file_bytes()
        for path in base_files:
            for variant in range(2):
                if created >= 24 or total_bytes >= max_total:
                    return created
                dest = corpus_dir / f"radamsa_{created+1:02d}{path.suffix.lower()}"
                proc = subprocess.run([radamsa, str(path)], capture_output=True)
                if proc.returncode != 0 or not proc.stdout:
                    continue
                attempted += 1
                data = proc.stdout[: min(len(proc.stdout), max_file)]
                if not data:
                    continue
                if seed_profile == "archive-container" and path.suffix.lower() in {".zip", ".tar", ".gz", ".bz2", ".xz"}:
                    if not self._archive_seed_is_semantically_valid(dest, data):
                        invalid += 1
                        invalid_ratio = float(invalid) / float(max(1, attempted))
                        if invalid_ratio > 0.40:
                            return created
                        continue
                if total_bytes + len(data) > max_total:
                    return created
                dest.write_bytes(data)
                total_bytes += len(data)
                created += 1
        return created

    def _filter_seed_corpus(
        self,
        corpus_dir: Path,
        *,
        seed_profile: str,
        required_families: list[str],
        target_markers: list[str] | None = None,
    ) -> dict[str, object]:
        filter_mode = _seed_filter_mode()
        files = sorted(p for p in corpus_dir.iterdir() if p.is_file()) if corpus_dir.is_dir() else []
        raw_counts = {"repo_examples": 0, "ai": 0, "radamsa": 0, "total": len(files)}
        for path in files:
            if path.name.startswith("repo_"):
                raw_counts["repo_examples"] += 1
            elif path.name.startswith("radamsa_"):
                raw_counts["radamsa"] += 1
            else:
                raw_counts["ai"] += 1
        filtered_counts = dict(raw_counts)
        noise_rejected = 0
        oversized_rejected = 0
        total_pruned_count = 0
        total_pruned_bytes = 0
        archive_valid_count = 0
        archive_invalid_count = 0
        archive_magic_only_rejected_count = 0
        archive_malformed_pruned_count = 0
        family_caps: dict[str, int] = {}
        family_members: dict[str, int] = {}
        path_families: dict[Path, set[str]] = {}
        content_hashes: set[str] = set()
        shape_hashes: set[str] = set()
        shape_group_counts: dict[str, int] = {}
        archive_malformed_marks: dict[Path, bool] = {}
        textual_mode = seed_profile in {
            "parser-format",
            "parser-token",
            "parser-structure",
            "parser-numeric",
            "document-text",
            "serializer-structured",
        }
        max_file = self._seed_max_file_bytes()
        max_radamsa_file = self._seed_radamsa_max_file_bytes()
        max_total = self._seed_max_total_bytes()
        required_family_set = set(required_families)
        base_total = max(1, len(files))
        strict_required_cap = 3
        strict_optional_cap = 2
        soft_required_cap = max(6, min(24, base_total // 3))
        soft_optional_cap = max(4, min(16, base_total // 4))
        filtered_by_rule_breakdown = {
            "size": 0,
            "hash": 0,
            "shape": 0,
            "family": 0,
            "noise": 0,
            "total_bytes": 0,
        }
        kept: list[Path] = []
        for path in files:
            reject = False
            reject_reason = ""
            try:
                size = int(path.stat().st_size)
            except Exception:
                size = 0
            try:
                data = path.read_bytes()
            except Exception:
                data = b""
            if size <= 0:
                reject = True
                oversized_rejected += 1
                reject_reason = "size"
            if not reject and size > max_file:
                reject = True
                oversized_rejected += 1
                reject_reason = "size"
            if not reject and path.name.startswith("radamsa_") and size > max_radamsa_file:
                reject = True
                oversized_rejected += 1
                reject_reason = "size"
            if (
                not reject
                and textual_mode
                and filter_mode == "strict"
                and not _looks_textual_seed(path)
            ):
                reject = True
                noise_rejected += 1
                reject_reason = "noise"
            elif (
                not reject
                and textual_mode
                and filter_mode == "soft"
                and not _looks_textual_seed(path)
            ):
                # Soft mode: keep weak textual samples unless they are mostly binary noise.
                printable = 0
                for b in data[:2048]:
                    if b in (9, 10, 13) or 32 <= b <= 126:
                        printable += 1
                printable_ratio = float(printable) / float(max(1, min(len(data), 2048)))
                if printable_ratio < 0.25:
                    reject = True
                    noise_rejected += 1
                    reject_reason = "noise"
            if (
                not reject
                and seed_profile == "archive-container"
                and self._seed_archive_validate_enabled()
            ):
                if self._is_magic_only_archive_seed(data):
                    reject = True
                    noise_rejected += 1
                    archive_magic_only_rejected_count += 1
                    reject_reason = "noise"
                elif path.suffix.lower() in {".zip", ".tar", ".gz", ".bz2", ".xz", ".bin", ".dat"}:
                    if self._archive_seed_is_semantically_valid(path, data):
                        archive_valid_count += 1
                    else:
                        archive_invalid_count += 1
                        reject = True
                        noise_rejected += 1
                        reject_reason = "noise"
            digest = hashlib.sha256(data).hexdigest()
            if not reject and digest in content_hashes:
                reject = True
                reject_reason = "hash"
            if not reject:
                content_hashes.add(digest)
            families = _classify_seed_family(path)
            if (
                textual_mode
                and seed_profile not in {"parser-format", "parser-numeric"}
                and not reject
                and filter_mode != "off"
            ):
                shape = _normalized_format_shape(data.decode("utf-8", errors="replace")[:512])
                if shape:
                    if filter_mode == "strict":
                        if shape in shape_hashes:
                            reject = True
                            reject_reason = "shape"
                        shape_hashes.add(shape)
                    else:
                        source_key = "repo" if path.name.startswith("repo_") else ("radamsa" if path.name.startswith("radamsa_") else "ai")
                        family_key = ",".join(sorted(families)) if families else "_none"
                        group_key = f"{source_key}|{family_key}|{shape}"
                        group_count = shape_group_counts.get(group_key, 0)
                        if group_count >= 3:
                            reject = True
                            reject_reason = "shape"
                        shape_group_counts[group_key] = group_count + 1
            if not reject and families and filter_mode != "off":
                cap_hit = False
                for family in sorted(families):
                    current = family_caps.get(family, 0)
                    if filter_mode == "strict":
                        cap = strict_required_cap if family in required_family_set else strict_optional_cap
                    else:
                        cap = soft_required_cap if family in required_family_set else soft_optional_cap
                    if current >= cap:
                        cap_hit = True
                        break
                if cap_hit:
                    reject = True
                    reject_reason = "family"
            if reject:
                if reject_reason in filtered_by_rule_breakdown:
                    filtered_by_rule_breakdown[reject_reason] += 1
                try:
                    path.unlink()
                except Exception:
                    pass
                if path.name.startswith("repo_"):
                    filtered_counts["repo_examples"] = max(0, int(filtered_counts["repo_examples"]) - 1)
                elif path.name.startswith("radamsa_"):
                    filtered_counts["radamsa"] = max(0, int(filtered_counts["radamsa"]) - 1)
                else:
                    filtered_counts["ai"] = max(0, int(filtered_counts["ai"]) - 1)
                filtered_counts["total"] = max(0, int(filtered_counts["total"]) - 1)
                continue
            kept.append(path)
            path_families[path] = set(families)
            archive_malformed_marks[path] = (
                seed_profile == "archive-container" and self._is_archive_malformed_candidate(path, data)
            )
            for family in families:
                family_caps[family] = family_caps.get(family, 0) + 1
                family_members[family] = family_members.get(family, 0) + 1
        if seed_profile == "archive-container" and kept:
            max_malformed_ratio = self._seed_archive_max_malformed_ratio()

            def _current_malformed_paths() -> list[Path]:
                return [p for p in kept if archive_malformed_marks.get(p, False)]

            malformed_paths = _current_malformed_paths()
            while kept and malformed_paths and (float(len(malformed_paths)) / float(len(kept))) > max_malformed_ratio:
                def _prune_priority(p: Path) -> tuple[int, int]:
                    if p.name.startswith("radamsa_"):
                        source_prio = 0
                    elif p.name.startswith("repo_"):
                        source_prio = 2
                    else:
                        source_prio = 1
                    try:
                        sz = int(p.stat().st_size)
                    except Exception:
                        sz = 0
                    return (source_prio, -sz)

                victim = sorted(malformed_paths, key=_prune_priority)[0]
                try:
                    victim_size = int(victim.stat().st_size)
                except Exception:
                    victim_size = 0
                try:
                    victim.unlink()
                except Exception:
                    break
                archive_malformed_pruned_count += 1
                total_pruned_count += 1
                total_pruned_bytes += max(0, victim_size)
                filtered_by_rule_breakdown["total_bytes"] += 1
                kept = [p for p in kept if p != victim]
                if victim.name.startswith("repo_"):
                    filtered_counts["repo_examples"] = max(0, int(filtered_counts["repo_examples"]) - 1)
                elif victim.name.startswith("radamsa_"):
                    filtered_counts["radamsa"] = max(0, int(filtered_counts["radamsa"]) - 1)
                else:
                    filtered_counts["ai"] = max(0, int(filtered_counts["ai"]) - 1)
                filtered_counts["total"] = max(0, int(filtered_counts["total"]) - 1)
                archive_malformed_marks.pop(victim, None)
                for fam in path_families.get(victim, set()):
                    family_members[fam] = max(0, family_members.get(fam, 0) - 1)
                path_families.pop(victim, None)
                malformed_paths = _current_malformed_paths()
        if kept:
            total_bytes = 0
            sized: list[tuple[Path, int]] = []
            for path in kept:
                try:
                    sz = int(path.stat().st_size)
                except Exception:
                    sz = 0
                total_bytes += max(0, sz)
                sized.append((path, max(0, sz)))
            if total_bytes > max_total:
                # Prune least-useful large seeds first: radamsa > ai > repo examples.
                def _priority(item: tuple[Path, int]) -> tuple[int, int]:
                    p, sz = item
                    if p.name.startswith("radamsa_"):
                        prio = 0
                    elif p.name.startswith("repo_"):
                        prio = 2
                    else:
                        prio = 1
                    return (prio, -sz)

                for path, sz in sorted(sized, key=_priority):
                    if total_bytes <= max_total:
                        break
                    path_family_set = path_families.get(path, set())
                    if required_family_set and any(
                        fam in required_family_set and family_members.get(fam, 0) <= 1
                        for fam in path_family_set
                    ):
                        continue
                    try:
                        path.unlink()
                    except Exception:
                        continue
                    total_bytes = max(0, total_bytes - sz)
                    total_pruned_count += 1
                    total_pruned_bytes += sz
                    filtered_by_rule_breakdown["total_bytes"] += 1
                    if path.name.startswith("repo_"):
                        filtered_counts["repo_examples"] = max(0, int(filtered_counts["repo_examples"]) - 1)
                    elif path.name.startswith("radamsa_"):
                        filtered_counts["radamsa"] = max(0, int(filtered_counts["radamsa"]) - 1)
                    else:
                        filtered_counts["ai"] = max(0, int(filtered_counts["ai"]) - 1)
                    filtered_counts["total"] = max(0, int(filtered_counts["total"]) - 1)
                    for fam in path_family_set:
                        family_members[fam] = max(0, family_members.get(fam, 0) - 1)
                    path_families.pop(path, None)
        retention_ratio_ai = float(filtered_counts.get("ai") or 0) / float(max(1, int(raw_counts.get("ai") or 0)))
        retention_ratio_radamsa = float(filtered_counts.get("radamsa") or 0) / float(max(1, int(raw_counts.get("radamsa") or 0)))
        retention_ratio_repo = float(filtered_counts.get("repo_examples") or 0) / float(max(1, int(raw_counts.get("repo_examples") or 0)))
        archive_malformed_count = sum(1 for p in kept if archive_malformed_marks.get(p, False))
        return {
            "seed_filter_mode": filter_mode,
            "seed_counts_raw": raw_counts,
            "seed_counts_filtered": filtered_counts,
            "seed_noise_rejected_count": noise_rejected,
            "seed_oversized_rejected_count": oversized_rejected,
            "seed_total_pruned_count": total_pruned_count,
            "seed_total_pruned_bytes": total_pruned_bytes,
            "filtered_by_rule_breakdown": filtered_by_rule_breakdown,
            "retention_ratio_ai": retention_ratio_ai,
            "retention_ratio_radamsa": retention_ratio_radamsa,
            "retention_ratio_repo": retention_ratio_repo,
            "seed_max_file_bytes": max_file,
            "seed_radamsa_max_file_bytes": max_radamsa_file,
            "seed_max_total_bytes": max_total,
            "archive_valid_count": archive_valid_count,
            "archive_invalid_count": archive_invalid_count,
            "archive_magic_only_rejected_count": archive_magic_only_rejected_count,
            "archive_malformed_count": archive_malformed_count,
            "archive_malformed_pruned_count": archive_malformed_pruned_count,
            "archive_malformed_ratio": (
                float(archive_malformed_count) / float(max(1, int(filtered_counts.get("total") or 0)))
                if seed_profile == "archive-container"
                else 0.0
            ),
            "archive_max_malformed_ratio": self._seed_archive_max_malformed_ratio(),
            "archive_valid_ratio": (
                float(archive_valid_count) / float(max(1, archive_valid_count + archive_invalid_count))
                if seed_profile == "archive-container"
                else 1.0
            ),
            "seed_family_coverage": self._seed_family_coverage(corpus_dir, required_families),
        }

    def _pass_generate_seeds(self, fuzzer_name: str) -> None:
        harness_src = self._locate_harness_source_for(fuzzer_name)
        harness_text = read_text_safely(harness_src) if harness_src else ""
        readme_text = read_text_safely(self.fuzz_dir / "README.md")
        corpus_dir = self.fuzz_corpus_dir / fuzzer_name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        seed_exploration_path = self._seed_exploration_path(fuzzer_name)
        seed_check_path = self._seed_check_path(fuzzer_name)
        selected_target = self._resolve_selected_target(fuzzer_name, harness_text)
        observed_target = self._resolve_observed_target(fuzzer_name, harness_text)
        target_type = ""
        seed_profile = ""
        seed_profile_source = "fallback"
        if selected_target:
            selected_type = str(selected_target.get("target_type") or "").strip().lower()
            selected_profile = str(selected_target.get("seed_profile") or "").strip().lower()
            if selected_type in ALLOWED_TARGET_TYPES and selected_profile in ALLOWED_SEED_PROFILES:
                target_type = selected_type
                seed_profile = selected_profile
                seed_profile_source = "selected_targets"
        if not target_type or not seed_profile:
            target_type, seed_profile = self._resolve_seed_target_metadata(fuzzer_name, harness_text)
            if observed_target:
                seed_profile_source = "observed_or_inferred"
        execution_target = dict(observed_target or selected_target)
        if selected_target:
            self.last_selected_target_by_fuzzer[fuzzer_name] = dict(selected_target)
        required_families, optional_families = _seed_families_for_target(
            seed_profile,
            fuzzer_name,
            harness_text,
            readme_text,
            str(execution_target.get("observed_target_api") or ""),
            str(execution_target.get("selected_target_api") or ""),
            str(execution_target.get("target_name") or ""),
            str(execution_target.get("api") or ""),
        )
        seed_guidance = self._seed_generation_guidance(target_type, seed_profile, fuzzer_name, harness_text)
        self.last_seed_profile_by_fuzzer[fuzzer_name] = seed_profile
        repo_seed_files, repo_meta = self._collect_repo_seed_examples(
            seed_profile,
            fuzzer_name,
            corpus_dir,
            required_families=required_families,
        )
        sources = list(repo_meta.get("sources") or [])
        family_coverage = self._seed_family_coverage(corpus_dir, required_families)
        target_corpus_files = max(self._seed_corpus_min_per_target(), len(required_families) * 2)
        per_family_target = 2 if required_families else 1

        instructions = textwrap.dedent(
            f"""
            First explore repository facts for the harness `{fuzzer_name}`, then add or refine **warm-up seed files by family bucket**
            inside `{corpus_dir.relative_to(self.repo_root)}`. Reuse the existing corpus files as grounding and prioritize missing families.
            Use appropriate file extensions if known. If binary, you may write contents via hex bytes.

            {seed_guidance}

            Required seed families:
            {", ".join(required_families) if required_families else "none"}

            Optional seed families:
            {", ".join(optional_families) if optional_families else "none"}

            Current corpus summary:
            {self._summarize_seed_corpus(corpus_dir)}

            Target metadata source:
            - target_type/seed_profile source: {seed_profile_source}
            - selected seed_profile from `fuzz/selected_targets.json`: {str(selected_target.get("seed_profile") or "(missing)") if selected_target else "(missing)"}
            - active seed_profile for this run: {seed_profile}

            Current seed family coverage:
            covered={", ".join(family_coverage.get("covered") or []) if family_coverage.get("covered") else "none"}
            missing={", ".join(family_coverage.get("missing") or []) if family_coverage.get("missing") else "none"}

            Active seed filter mode:
            - `{_seed_filter_mode()}` (default soft; keep semantic diversity while still removing exact duplicates and oversized files)

            Corpus size goal:
            - Aim for at least {target_corpus_files} total seed files in `{corpus_dir.relative_to(self.repo_root)}` after your edits.
            - Aim for at least {per_family_target} semantically different seed files for each required family where feasible.

            Coverage-oriented gap hints:
            {self._infer_seed_gaps(seed_profile, corpus_dir)}

            Rules:
            - Before writing new seeds, inspect repository files relevant to target inputs: tests, examples, fuzz directories, build files, `fuzz/PLAN.md`, and target metadata files.
            - For `archive-container`, real archive samples must come first: import/use repository examples from `contrib/oss-fuzz/corpus.zip`, `contrib/oss-fuzz/**`, `test/**`, or `tests/**` before adding synthetic variants.
            - For `archive-container`, avoid hand-crafted magic-only files (for example a few header bytes without a valid structure).
            - For `archive-container`, keep malformed/truncated seeds <= 30% of the corpus. Prioritize valid archive samples.
            - For `archive-container`, ensure at least one semantically valid archive sample exists in the corpus.
            - Write a concise exploration summary to `{seed_exploration_path.relative_to(self.repo_root)}` before or alongside seed creation.
            - `{seed_exploration_path.relative_to(self.repo_root)}` must be plain JSON with these keys only: `chosen_target_api`, `observed_target_api`, `seed_profile`, `required_families`, `missing_families`, `repo_paths_reviewed`, `sample_inputs_found`, `summary`.
            - Keep `repo_paths_reviewed` concrete and short. It should list the actual repository files or directories inspected for seed design.
            - Keep `sample_inputs_found` concrete. List real repo examples, existing corpus files, or note that none were found.
            - Before finishing, write a seed self-check file to `{seed_check_path.relative_to(self.repo_root)}`.
            - `{seed_check_path.relative_to(self.repo_root)}` must be plain JSON with these keys only: `seed_profile`, `required_families`, `covered_families`, `missing_families`, `family_counts`, `corpus_files`, `target_corpus_files`, `per_family_target`, `planned_additions`, `summary`.
            - Use `{seed_check_path.relative_to(self.repo_root)}` to self-check whether the current corpus is sufficient. If required families are still missing, or if the corpus is still much smaller than the target size, add more seeds before finishing.
            - Before finishing, write a seed self-check file to `{seed_check_path.relative_to(self.repo_root)}`.
            - `{seed_check_path.relative_to(self.repo_root)}` must be plain JSON with these keys only: `seed_profile`, `required_families`, `covered_families`, `missing_families`, `family_counts`, `corpus_files`, `target_corpus_files`, `per_family_target`, `planned_additions`, `summary`.
            - Use `{seed_check_path.relative_to(self.repo_root)}` to self-check whether the current corpus is sufficient. If required families are still missing, or if the corpus is still much smaller than the target size, add more seeds before finishing.
            - Treat `fuzz/observed_target.json` as the execution truth source when present; do not generate seeds only for the originally selected target if the actual harness drifted.
            - Each missing required family should have at least one representative seed after your edits.
            - Do not stop after creating only one tiny seed per family. Build a thicker warm-up corpus with multiple semantically different seeds per required family.
            - Prefer missing families over adding more variants to already-covered malformed cases.
            - Do not only generate malformed separator variants if structure families are missing.
            - If this is a textual DSL or textual parser target, prefer readable text seeds that directly exercise the observed target grammar/path.
            - For textual targets, avoid random binary noise, large opaque blobs, or mostly non-printable bytes unless the harness clearly expects binary input.
            - Keep seeds semantically distinct by family bucket; do not create many near-duplicate seeds that only change one random byte.
            - Soft filtering keeps diverse seeds; do not assume near variants will always be removed.
            - Each seed file must stay small (<= {self._seed_max_file_bytes()} bytes by default). Prefer concise high-signal seeds over large blobs.
            - Only create seed files plus `{seed_exploration_path.relative_to(self.repo_root)}` and `{seed_check_path.relative_to(self.repo_root)}` (no code changes).
            - Only create seed files plus `{seed_exploration_path.relative_to(self.repo_root)}` and `{seed_check_path.relative_to(self.repo_root)}` (no code changes).
            - When finished, write the path to one seed file into `./done`.
            """
        ).strip()

        seed_timeout = getattr(self, "seed_generation_timeout_sec", None)
        patcher_kwargs: Dict[str, object] = {}
        if isinstance(seed_timeout, int) and seed_timeout > 0:
            patcher_kwargs["timeout"] = seed_timeout
        selected_targets_text = read_text_safely(self.fuzz_dir / "selected_targets.json")
        observed_target_text = read_text_safely(self.fuzz_dir / "observed_target.json")
        target_analysis_text = read_text_safely(self.fuzz_dir / "target_analysis.json")
        antlr_text = read_text_safely(self.fuzz_dir / "antlr_plan_context.json")
        plan_text = read_text_safely(self.fuzz_dir / "PLAN.md")
        repo_understanding_text = read_text_safely(self.fuzz_dir / "repo_understanding.json")
        build_strategy_text = read_text_safely(self.fuzz_dir / "build_strategy.json")
        additional_context_parts = [
            "=== fuzz/observed_target.json ===\n" + (observed_target_text or "(missing)"),
            "=== fuzz/selected_targets.json ===\n" + (selected_targets_text or "(missing)"),
            "=== fuzz/target_analysis.json ===\n" + (target_analysis_text or "(missing)"),
            "=== fuzz/antlr_plan_context.json ===\n" + (antlr_text or "(missing)"),
            "=== fuzz/PLAN.md ===\n" + (plan_text or "(missing)"),
            "=== fuzz/repo_understanding.json ===\n" + (repo_understanding_text or "(missing)"),
            "=== fuzz/build_strategy.json ===\n" + (build_strategy_text or "(missing)"),
            "=== harness source ===\n" + (harness_text or "(no harness found)"),
            "=== fuzz/README.md ===\n" + (readme_text or "(missing)"),
            "=== seed family coverage ===\n" + json.dumps(family_coverage, ensure_ascii=False, indent=2),
        ]
        stdout = self.patcher.run_codex_command(
            instructions,
            additional_context="\n\n".join(additional_context_parts),
            stage_skill="seed_generation",
            **patcher_kwargs,
        )
        if stdout is None:
            raise HarnessGeneratorError("Codex did not generate any seed files.")
        ai_seed_count = len([p for p in corpus_dir.iterdir() if p.is_file()]) - len(repo_seed_files)
        if ai_seed_count < 0:
            ai_seed_count = 0
        radamsa_count = self._run_radamsa_bootstrap(corpus_dir, seed_profile=seed_profile)
        if radamsa_count > 0:
            sources.append("radamsa")
        if ai_seed_count > 0:
            sources.append("ai")
        filtered_meta = self._filter_seed_corpus(
            corpus_dir,
            seed_profile=seed_profile,
            required_families=required_families,
            target_markers=[
                fuzzer_name,
                harness_text,
                readme_text,
                str(execution_target.get("observed_target_api") or ""),
                str(execution_target.get("selected_target_api") or ""),
                str(execution_target.get("target_name") or ""),
                str(execution_target.get("api") or ""),
            ],
        )
        archive_valid_ratio = float(filtered_meta.get("archive_valid_ratio") or 1.0)
        archive_malformed_ratio = float(filtered_meta.get("archive_malformed_ratio") or 0.0)
        archive_valid_count = int(filtered_meta.get("archive_valid_count") or 0)
        archive_max_malformed_ratio = float(filtered_meta.get("archive_max_malformed_ratio") or self._seed_archive_max_malformed_ratio())
        seed_family_cov = dict(filtered_meta.get("seed_family_coverage") or {})
        required_total = max(0, int(seed_family_cov.get("required_total") or 0))
        covered_required = max(0, int(seed_family_cov.get("covered_required") or 0))
        family_coverage_ratio = (
            float(covered_required) / float(required_total)
            if required_total > 0
            else 1.0
        )
        retention_ai = float(filtered_meta.get("retention_ratio_ai") or 0.0)
        retention_repo = float(filtered_meta.get("retention_ratio_repo") or 0.0)
        retention_radamsa = float(filtered_meta.get("retention_ratio_radamsa") or 0.0)
        retention_blended = max(
            0.0,
            min(
                1.0,
                0.50 * retention_ai + 0.35 * retention_repo + 0.15 * retention_radamsa,
            ),
        )
        validity_component = (
            max(0.0, min(1.0, archive_valid_ratio))
            if seed_profile == "archive-container"
            else 1.0
        )
        novelty_component = max(
            0.0,
            min(
                1.0,
                0.70 * family_coverage_ratio + 0.30 * retention_blended,
            ),
        )
        redundancy_penalty = 0.0
        if int(filtered_meta.get("seed_noise_rejected_count") or 0) > 0:
            redundancy_penalty += 0.25
        if int((filtered_meta.get("filtered_by_rule_breakdown") or {}).get("shape") or 0) > 0:
            redundancy_penalty += 0.35
        if int((filtered_meta.get("filtered_by_rule_breakdown") or {}).get("family") or 0) > 0:
            redundancy_penalty += 0.25
        redundancy_penalty = max(0.0, min(1.0, redundancy_penalty))
        alpha, beta, gamma, eta = 0.40, 0.35, 0.25, 0.20
        seed_score_prefuzz = max(
            0.0,
            min(
                1.0,
                alpha * novelty_component
                + beta * validity_component
                + gamma * retention_blended
                - eta * redundancy_penalty,
            ),
        )
        if (
            seed_profile == "archive-container"
            and self._seed_archive_validate_enabled()
            and archive_valid_ratio < self._seed_archive_min_valid_ratio()
        ):
            print(
                "[warn] archive seed valid ratio below threshold "
                f"for {fuzzer_name}: ratio={archive_valid_ratio:.2f} "
                f"threshold={self._seed_archive_min_valid_ratio():.2f}"
            )
        if seed_profile == "archive-container" and archive_valid_count < 1:
            print(f"[warn] archive seed corpus has no semantically valid sample for {fuzzer_name}")
        if seed_profile == "archive-container" and archive_malformed_ratio > archive_max_malformed_ratio:
            print(
                "[warn] archive malformed ratio above threshold "
                f"for {fuzzer_name}: ratio={archive_malformed_ratio:.2f} "
                f"threshold={archive_max_malformed_ratio:.2f}"
            )
        seed_quality_path = self.fuzz_dir / f"seed_quality_{re.sub(r'[^A-Za-z0-9_.-]+', '_', fuzzer_name)}.json"
        seed_quality_doc = {
            "fuzzer": fuzzer_name,
            "seed_profile": seed_profile,
            "target_type": target_type,
            "seed_profile_source": seed_profile_source,
            "required_families": required_families,
            "optional_families": optional_families,
            "seed_filter_mode": str(filtered_meta.get("seed_filter_mode") or _seed_filter_mode()),
            "seed_counts_raw": dict(filtered_meta.get("seed_counts_raw") or {}),
            "seed_counts_filtered": dict(filtered_meta.get("seed_counts_filtered") or {}),
            "seed_family_coverage": dict(filtered_meta.get("seed_family_coverage") or {}),
            "seed_noise_rejected_count": int(filtered_meta.get("seed_noise_rejected_count") or 0),
            "seed_oversized_rejected_count": int(filtered_meta.get("seed_oversized_rejected_count") or 0),
            "filtered_by_rule_breakdown": dict(filtered_meta.get("filtered_by_rule_breakdown") or {}),
            "retention_ratio_ai": float(filtered_meta.get("retention_ratio_ai") or 0.0),
            "retention_ratio_radamsa": float(filtered_meta.get("retention_ratio_radamsa") or 0.0),
            "retention_ratio_repo": float(filtered_meta.get("retention_ratio_repo") or 0.0),
            "archive_valid_count": int(filtered_meta.get("archive_valid_count") or 0),
            "archive_invalid_count": int(filtered_meta.get("archive_invalid_count") or 0),
            "archive_magic_only_rejected_count": int(filtered_meta.get("archive_magic_only_rejected_count") or 0),
            "archive_malformed_count": int(filtered_meta.get("archive_malformed_count") or 0),
            "archive_malformed_pruned_count": int(filtered_meta.get("archive_malformed_pruned_count") or 0),
            "archive_malformed_ratio": archive_malformed_ratio,
            "archive_max_malformed_ratio": archive_max_malformed_ratio,
            "archive_valid_ratio": archive_valid_ratio,
            "seed_score": float(seed_score_prefuzz),
            "seed_score_components": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "eta": eta,
                "coverage_potential": float(novelty_component),
                "validity": float(validity_component),
                "novelty": float(retention_blended),
                "redundancy_penalty": float(redundancy_penalty),
                "family_coverage_ratio": float(family_coverage_ratio),
            },
        }
        try:
            seed_quality_path.write_text(json.dumps(seed_quality_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        self.last_seed_bootstrap_by_fuzzer[fuzzer_name] = {
            "counts": {
                "repo_examples": len(repo_seed_files),
                "ai": ai_seed_count,
                "radamsa": radamsa_count,
                "total": len([p for p in corpus_dir.iterdir() if p.is_file()]),
            },
            "sources": sorted(set(sources)),
            "seed_profile": seed_profile,
            "seed_profile_source": seed_profile_source,
            "target_type": target_type,
            "selected_target": dict(selected_target),
            "observed_target": dict(observed_target),
            "seed_families_required": required_families,
            "seed_families_optional": optional_families,
            "seed_family_coverage": dict(filtered_meta.get("seed_family_coverage") or self._seed_family_coverage(corpus_dir, required_families)),
            "seed_counts_raw": dict(filtered_meta.get("seed_counts_raw") or {}),
            "seed_counts_filtered": dict(filtered_meta.get("seed_counts_filtered") or {}),
            "seed_noise_rejected_count": int(filtered_meta.get("seed_noise_rejected_count") or 0),
            "seed_oversized_rejected_count": int(filtered_meta.get("seed_oversized_rejected_count") or 0),
            "seed_filter_mode": str(filtered_meta.get("seed_filter_mode") or _seed_filter_mode()),
            "filtered_by_rule_breakdown": dict(filtered_meta.get("filtered_by_rule_breakdown") or {}),
            "retention_ratio_ai": float(filtered_meta.get("retention_ratio_ai") or 0.0),
            "retention_ratio_radamsa": float(filtered_meta.get("retention_ratio_radamsa") or 0.0),
            "retention_ratio_repo": float(filtered_meta.get("retention_ratio_repo") or 0.0),
            "seed_total_pruned_count": int(filtered_meta.get("seed_total_pruned_count") or 0),
            "seed_total_pruned_bytes": int(filtered_meta.get("seed_total_pruned_bytes") or 0),
            "seed_max_file_bytes": int(filtered_meta.get("seed_max_file_bytes") or self._seed_max_file_bytes()),
            "seed_radamsa_max_file_bytes": int(filtered_meta.get("seed_radamsa_max_file_bytes") or self._seed_radamsa_max_file_bytes()),
            "seed_max_total_bytes": int(filtered_meta.get("seed_max_total_bytes") or self._seed_max_total_bytes()),
            "archive_valid_count": int(filtered_meta.get("archive_valid_count") or 0),
            "archive_invalid_count": int(filtered_meta.get("archive_invalid_count") or 0),
            "archive_magic_only_rejected_count": int(filtered_meta.get("archive_magic_only_rejected_count") or 0),
            "archive_malformed_count": int(filtered_meta.get("archive_malformed_count") or 0),
            "archive_malformed_pruned_count": int(filtered_meta.get("archive_malformed_pruned_count") or 0),
            "archive_malformed_ratio": archive_malformed_ratio,
            "archive_max_malformed_ratio": archive_max_malformed_ratio,
            "archive_valid_ratio": archive_valid_ratio,
            "seed_score": float(seed_quality_doc.get("seed_score") or 0.0),
            "seed_score_components": dict(seed_quality_doc.get("seed_score_components") or {}),
            "repo_examples_filtered": bool(repo_meta.get("filtered") or False),
            "repo_examples_rejected_count": int(repo_meta.get("rejected_count") or 0),
            "repo_examples_accepted_count": int(repo_meta.get("accepted_count") or 0),
            "seed_exploration_path": str(seed_exploration_path.relative_to(self.repo_root)) if seed_exploration_path.is_file() else "",
            "seed_check_path": str(seed_check_path.relative_to(self.repo_root)) if seed_check_path.is_file() else "",
            "seed_quality_path": str(seed_quality_path.relative_to(self.repo_root)) if seed_quality_path.is_file() else "",
        }
        if not seed_exploration_path.is_file():
            print(f"[warn] seed exploration summary missing for {fuzzer_name}: {seed_exploration_path.relative_to(self.repo_root)}")
        if not seed_check_path.is_file():
            print(f"[warn] seed self-check summary missing for {fuzzer_name}: {seed_check_path.relative_to(self.repo_root)}")
        print(f"[*] Codex seed creation done (truncated):\n{stdout[:600]}")

    # ────────────────────────────────────────────────────────────────────
    # Step E-pre – Dictionary generation & corpus minimization
    # ────────────────────────────────────────────────────────────────────

    def _generate_dictionary(self, bin_path: Path, seed_profile: str) -> Optional[Path]:
        """
        Auto-generate a libFuzzer dictionary file for the given fuzzer binary.
        Sources:
          1. Built-in tokens from PROFILE_DICTIONARY_TOKENS keyed by seed_profile
          2. Existing .dict files under fuzz/
          3. String literals extracted from the harness source file
        Returns the path to the generated .dict file, or None if no tokens found.
        """
        tokens: list[str] = []

        # 1. Profile-based tokens
        profile_tokens = PROFILE_DICTIONARY_TOKENS.get(seed_profile or "generic", [])
        tokens.extend(profile_tokens)

        # 2. Collect tokens from existing .dict files in fuzz/
        dict_dir = self.repo_root / FUZZ_DICT_DIR
        dict_dir.mkdir(parents=True, exist_ok=True)
        for existing_dict in self.fuzz_dir.rglob("*.dict"):
            try:
                for line in existing_dict.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Accept lines like: keyword="value" or just "value"
                        if "=" in line:
                            line = line.split("=", 1)[1].strip()
                        if line.startswith('"') and line.endswith('"'):
                            tokens.append(line)
            except Exception:
                pass

        # 3. Extract string literals from harness source
        harness_extensions = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")
        for src in self.fuzz_dir.iterdir():
            if src.is_file() and src.suffix in harness_extensions:
                try:
                    content = src.read_text(encoding="utf-8", errors="replace")
                    # Extract C string literals (simple heuristic)
                    for m in re.finditer(r'"([^"\\]{1,64}(?:\\.[^"\\]{0,64})*)"', content):
                        literal = m.group(0)
                        # Skip very common/useless strings
                        if len(m.group(1)) >= 2 and literal not in {'"\\n"', '"\\0"', '""'}:
                            tokens.append(literal)
                except Exception:
                    pass

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_tokens: list[str] = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        if not unique_tokens:
            return None

        # Write dictionary file
        dict_path = dict_dir / f"{bin_path.stem}.dict"
        lines = [f"# Auto-generated dictionary for {bin_path.name} (profile: {seed_profile})"]
        for i, tok in enumerate(unique_tokens[:256]):  # Cap at 256 entries
            lines.append(f"token_{i}={tok}")
        write_text_safely(dict_path, "\n".join(lines) + "\n")
        print(f"[*] Generated dictionary: {dict_path.relative_to(self.repo_root)} ({len(unique_tokens[:256])} tokens)")
        return dict_path

    def _minimize_corpus(self, bin_path: Path, corpus_dir: Path) -> Dict[str, int]:
        """
        Minimize the corpus using libFuzzer -merge=1 mode.
        Replaces the corpus dir contents with the minimized set.
        Returns a dict with before/after file counts and sizes.
        """
        if not corpus_dir.exists():
            return {"before_files": 0, "after_files": 0, "before_bytes": 0, "after_bytes": 0}

        before_files = 0
        before_bytes = 0
        for p in corpus_dir.rglob("*"):
            if p.is_file():
                before_files += 1
                try:
                    before_bytes += int(p.stat().st_size)
                except Exception:
                    pass

        if before_files < 4:
            return {"before_files": before_files, "after_files": before_files,
                    "before_bytes": before_bytes, "after_bytes": before_bytes}

        merged_dir = corpus_dir.parent / f"{corpus_dir.name}_merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env = self._compose_vcpkg_runtime_env(env)
        env.setdefault("ASAN_OPTIONS", "exitcode=76:detect_leaks=0")
        env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1")
        env.setdefault("LLVM_SYMBOLIZER_PATH", which("llvm-symbolizer") or "")

        cmd = [str(bin_path), "-merge=1", str(merged_dir), str(corpus_dir)]
        try:
            rc, out, err = self._run_cmd(
                cmd, cwd=self.repo_root, env=env, timeout=120,
            )
            if rc == 0 and any(merged_dir.iterdir()):
                # Replace old corpus with merged
                for p in corpus_dir.iterdir():
                    if p.is_file():
                        p.unlink()
                for p in merged_dir.iterdir():
                    if p.is_file():
                        shutil.move(str(p), str(corpus_dir / p.name))
        except Exception as exc:
            print(f"[warn] corpus minimization failed: {exc}")
        finally:
            shutil.rmtree(merged_dir, ignore_errors=True)

        after_files = 0
        after_bytes = 0
        for p in corpus_dir.rglob("*"):
            if p.is_file():
                after_files += 1
                try:
                    after_bytes += int(p.stat().st_size)
                except Exception:
                    pass

        if before_files != after_files:
            print(f"[*] Corpus minimized: {before_files} -> {after_files} files, "
                  f"{before_bytes} -> {after_bytes} bytes")
        return {"before_files": before_files, "after_files": after_files,
                "before_bytes": before_bytes, "after_bytes": after_bytes}

    def collect_source_coverage(self, bin_path: Path) -> Optional[Dict[str, object]]:
        """
        Collect source-level coverage using llvm-cov after a fuzzer run.
        Requires the binary to have been compiled with
        -fprofile-instr-generate -fcoverage-mapping.

        Returns a dict with function-level coverage summary, or None if unavailable.
        """
        corpus_dir = self.fuzz_corpus_dir / bin_path.name
        if not corpus_dir.exists():
            return None

        llvm_profdata = which("llvm-profdata") or which("llvm-profdata-18")
        llvm_cov = which("llvm-cov") or which("llvm-cov-18")
        if not llvm_profdata or not llvm_cov:
            print("[warn] llvm-profdata/llvm-cov not found, skipping source coverage")
            return None

        # Check if binary has coverage instrumentation
        profraw_dir = self.repo_root / "fuzz" / "out"
        profraw_files = list(profraw_dir.glob("*.profraw")) if profraw_dir.exists() else []
        # Also check default location
        default_profraw = self.repo_root / "default.profraw"
        if default_profraw.exists():
            profraw_files.append(default_profraw)

        if not profraw_files:
            # Try running the binary against the corpus to generate profraw
            env = os.environ.copy()
            env["LLVM_PROFILE_FILE"] = str(self.fuzz_dir / "coverage.profraw")
            env.setdefault("ASAN_OPTIONS", "exitcode=76:detect_leaks=0")
            env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1")
            env.setdefault("LLVM_SYMBOLIZER_PATH", which("llvm-symbolizer") or "")

            corpus_files = sorted(corpus_dir.rglob("*"))[:50]  # sample up to 50
            if not corpus_files:
                return None

            for cf in corpus_files:
                if cf.is_file():
                    try:
                        self._run_cmd(
                            [str(bin_path), str(cf)],
                            cwd=self.repo_root, env=env, timeout=5,
                        )
                    except Exception:
                        pass  # Crashes are expected

            profraw_path = Path(env["LLVM_PROFILE_FILE"])
            if not profraw_path.exists():
                return None
            profraw_files = [profraw_path]

        # Merge profraw files
        profdata_path = self.fuzz_dir / "coverage.profdata"
        merge_cmd = [llvm_profdata, "merge", "-sparse"] + [str(p) for p in profraw_files] + ["-o", str(profdata_path)]
        try:
            rc, _, err = self._run_cmd(merge_cmd, cwd=self.repo_root, timeout=60)
            if rc != 0:
                print(f"[warn] llvm-profdata merge failed: {err[:200]}")
                return None
        except Exception as exc:
            print(f"[warn] llvm-profdata merge error: {exc}")
            return None

        # Generate function-level report
        report_cmd = [llvm_cov, "report", str(bin_path), f"-instr-profile={profdata_path}"]
        try:
            rc, report_out, report_err = self._run_cmd(report_cmd, cwd=self.repo_root, timeout=60)
            if rc != 0:
                print(f"[warn] llvm-cov report failed: {report_err[:200]}")
                return None
        except Exception as exc:
            print(f"[warn] llvm-cov report error: {exc}")
            return None

        # Parse function coverage from report
        uncovered_functions: list[str] = []
        total_functions = 0
        covered_functions = 0
        for line in report_out.splitlines():
            parts = line.split()
            if len(parts) >= 6:
                try:
                    funcs_total = int(parts[-4])
                    funcs_missed = int(parts[-3])
                    total_functions += funcs_total
                    covered_functions += funcs_total - funcs_missed
                except (ValueError, IndexError):
                    pass

        # Generate uncovered function list using llvm-cov export
        export_cmd = [
            llvm_cov, "export", str(bin_path),
            f"-instr-profile={profdata_path}",
            "-format=text", "-summary-only",
        ]
        try:
            rc, export_out, _ = self._run_cmd(export_cmd, cwd=self.repo_root, timeout=60)
            if rc == 0:
                try:
                    export_data = json.loads(export_out)
                    for file_data in export_data.get("data", []):
                        for fn in file_data.get("functions", []):
                            if fn.get("count", 0) == 0:
                                uncovered_functions.append(fn.get("name", ""))
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception:
            pass

        # Write coverage report file
        report_path = self.fuzz_dir / "coverage_report.txt"
        write_text_safely(report_path, report_out)

        coverage_pct = (float(covered_functions) / float(max(1, total_functions))) * 100.0

        result: Dict[str, object] = {
            "total_functions": total_functions,
            "covered_functions": covered_functions,
            "coverage_pct": round(coverage_pct, 1),
            "uncovered_functions": uncovered_functions[:20],  # top 20
            "report_path": str(report_path),
        }
        print(f"[*] Source coverage: {covered_functions}/{total_functions} functions "
              f"({coverage_pct:.1f}%), {len(uncovered_functions)} uncovered")
        return result

    @staticmethod
    def _adaptive_max_len(seed_profile: str, configured_max_len: int) -> int:
        """Return adaptive max_len based on seed_profile. If user explicitly
        configured a non-default value, respect it."""
        if configured_max_len > 0:
            return configured_max_len
        return PROFILE_MAX_LEN.get(seed_profile or "generic", 1024)

    @staticmethod
    def _adaptive_plateau_idle_sec(
        depth_class: str,
        execs_per_sec: int,
        initial_cov_growth_rate: float,
        configured: int,
    ) -> int:
        """Return adaptive plateau idle threshold based on target characteristics.
        If user explicitly configured a value via env var, respect it."""
        if configured > 0:
            return configured
        base = 180
        if depth_class == "shallow":
            base = 90
        elif depth_class == "deep":
            base = 360
        if execs_per_sec > 0 and execs_per_sec < 100:
            base = int(base * 1.5)
        if initial_cov_growth_rate > 2.0:
            base = int(base * 0.7)
        return max(60, min(600, base))

    def seed_completeness_pre_check(self, bin_path: Path) -> Dict[str, object]:
        """
        Pre-check seed corpus completeness before a full run.
        Returns a dict with check results and any issues found.
        """
        corpus_dir = self.fuzz_corpus_dir / bin_path.name
        issues: list[str] = []
        seed_profile = str(self.last_seed_profile_by_fuzzer.get(bin_path.name) or "generic")
        bootstrap = dict(self.last_seed_bootstrap_by_fuzzer.get(bin_path.name) or {})
        family_coverage = dict(bootstrap.get("seed_family_coverage") or {})
        required_families = list(bootstrap.get("seed_families_required") or [])
        covered_families = list(family_coverage.get("covered") or [])

        # Check corpus file count
        file_count = 0
        total_bytes = 0
        if corpus_dir.exists():
            for p in corpus_dir.rglob("*"):
                if p.is_file():
                    file_count += 1
                    try:
                        total_bytes += int(p.stat().st_size)
                    except Exception:
                        pass
        if file_count < 8:
            issues.append(f"corpus_too_small (only {file_count} files, need >= 8)")

        # Check family coverage
        missing_families = [f for f in required_families if f and f not in set(covered_families)]
        if missing_families:
            issues.append(f"missing_families: {', '.join(missing_families)}")

        # Check archive validity
        if seed_profile == "archive-container" and file_count > 0:
            valid = 0
            for p in corpus_dir.rglob("*"):
                if p.is_file():
                    try:
                        data = p.read_bytes()[:4]
                        if data[:2] == b'PK' or data[:3] == b'\x1f\x8b\x08' or data[:3] == b'BZh':
                            valid += 1
                    except Exception:
                        pass
            if valid < 1:
                issues.append("archive_no_valid_samples")

        result: Dict[str, object] = {
            "seed_profile": seed_profile,
            "corpus_files": file_count,
            "corpus_bytes": total_bytes,
            "required_families": required_families,
            "covered_families": covered_families,
            "missing_families": missing_families,
            "issues": issues,
            "passed": len(issues) == 0,
        }
        if issues:
            print(f"[warn] Seed pre-check issues for {bin_path.name}: {'; '.join(issues)}")
        else:
            print(f"[*] Seed pre-check passed for {bin_path.name}: {file_count} files, "
                  f"profile={seed_profile}")
        return result

    def dry_run_fuzzer(self, bin_path: Path, duration_sec: int = 10) -> Dict[str, object]:
        """
        Run the fuzzer for a short duration to validate basic functionality.
        Returns diagnostics: cov, ft, execs_per_sec, and any issues.
        """
        corpus_dir = self.fuzz_corpus_dir / bin_path.name
        corpus_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env = self._compose_vcpkg_runtime_env(env)
        env.setdefault("ASAN_OPTIONS", "exitcode=76:detect_leaks=0")
        env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1")
        env.setdefault("LLVM_SYMBOLIZER_PATH", which("llvm-symbolizer") or "")

        cmd = [
            str(bin_path),
            "-print_final_stats=1",
            f"-max_total_time={duration_sec}",
            f"-max_len={self._adaptive_max_len(str(self.last_seed_profile_by_fuzzer.get(bin_path.name) or 'generic'), self.max_len)}",
            f"-rss_limit_mb={self.rss_limit_mb}",
            str(corpus_dir),
        ]

        issues: list[str] = []
        print(f"[*] Dry-run: {' '.join(cmd)}")
        try:
            rc, out, err = self._run_cmd(
                cmd, cwd=self.repo_root, env=env,
                timeout=max(30, duration_sec + 20),
            )
            log = (out + "\n" + err).replace("\r", "\n")
            stats = parse_libfuzzer_final_stats(log)
            cov = int(stats.get("cov", 0))
            ft = int(stats.get("ft", 0))
            execs = int(stats.get("execs_per_sec", 0))

            if cov == 0:
                issues.append("zero_coverage")
            if cov > 0 and ft == 0:
                issues.append("zero_features")
            if execs > 0 and execs < 10:
                issues.append(f"very_slow ({execs} exec/s)")
            if rc != 0 and "ERROR" in log:
                issues.append("runtime_error")

            result: Dict[str, object] = {
                "rc": rc,
                "cov": cov,
                "ft": ft,
                "execs_per_sec": execs,
                "issues": issues,
                "passed": len(issues) == 0 or (cov > 0 and "runtime_error" not in issues),
            }
        except Exception as exc:
            result = {
                "rc": -1, "cov": 0, "ft": 0, "execs_per_sec": 0,
                "issues": [f"dry_run_failed: {exc}"], "passed": False,
            }

        if issues:
            print(f"[warn] Dry-run issues for {bin_path.name}: {'; '.join(issues)}")
        else:
            print(f"[*] Dry-run passed for {bin_path.name}: cov={result['cov']}, ft={result['ft']}, "
                  f"exec/s={result['execs_per_sec']}")
        return result

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
        env = self._compose_vcpkg_runtime_env(env)
        # Apply sanitizer-specific runtime options
        sanitizer_config = SANITIZER_CONFIGS.get(self.sanitizer, SANITIZER_CONFIGS["address"])
        for key, val in sanitizer_config.items():
            if key.endswith("_options"):
                env_key = key.upper()  # e.g. asan_options -> ASAN_OPTIONS
                env.setdefault(env_key, val)
        env.setdefault("ASAN_OPTIONS", "exitcode=76:detect_leaks=0")
        env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1")
        env.setdefault("LLVM_SYMBOLIZER_PATH", which("llvm-symbolizer") or "")

        corpus_dir = self.fuzz_corpus_dir / bin_path.name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        initial_corpus_files = 0
        initial_corpus_bytes = 0
        try:
            for p in corpus_dir.rglob("*"):
                if p.is_file():
                    initial_corpus_files += 1
                    try:
                        initial_corpus_bytes += int(p.stat().st_size)
                    except Exception:
                        pass
        except Exception:
            initial_corpus_files = 0
            initial_corpus_bytes = 0

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
        plateau_pulses = _run_plateau_pulses()
        plateau_idle_growth_sec = _run_plateau_idle_growth_sec()
        best_cov = 0
        best_ft = 0
        last_growth_at = time.monotonic()
        plateau_pulse_hits = 0
        callback_stop_reason = ""

        def _line_callback(_kind: str, text: str) -> Optional[str]:
            nonlocal best_cov, best_ft, last_growth_at, plateau_pulse_hits, callback_stop_reason
            m = _LIBFUZZER_PROGRESS_RE.search(text or "")
            if not m:
                return None
            cov = int(m.group("cov") or 0)
            ft = int(m.group("ft") or 0)
            kind = str(m.group("kind") or "").upper()
            grew = cov > best_cov or ft > best_ft
            if grew:
                best_cov = max(best_cov, cov)
                best_ft = max(best_ft, ft)
                last_growth_at = time.monotonic()
                plateau_pulse_hits = 0
                return None
            if kind == "PULSE":
                if (time.monotonic() - last_growth_at) >= plateau_idle_growth_sec:
                    plateau_pulse_hits += 1
                    if plateau_pulse_hits >= plateau_pulses:
                        callback_stop_reason = (
                            "coverage_plateau "
                            f"(idle_no_growth={plateau_idle_growth_sec}s pulse_hits={plateau_pulse_hits})"
                        )
                        return callback_stop_reason
            return None

        cmd = [
            str(bin_path),
            "-artifact_prefix=" + str(artifacts_dir) + "/",
            "-print_final_stats=1",
            f"-rss_limit_mb={self.rss_limit_mb}",
        ]

        # Adaptive max_len based on seed_profile
        seed_profile = str(self.last_seed_profile_by_fuzzer.get(bin_path.name) or "generic")
        effective_max_len = self._adaptive_max_len(seed_profile, self.max_len)
        cmd.append(f"-max_len={effective_max_len}")
        if effective_max_len != 1024:
            print(f"[*] Adaptive max_len={effective_max_len} for profile={seed_profile}")

        # Auto-generate and attach dictionary
        dict_path = self._generate_dictionary(bin_path, seed_profile)
        if dict_path and dict_path.is_file():
            cmd.append(f"-dict={dict_path}")

        # Fork mode for parallel exploration
        fork_count_raw = os.environ.get("SHERPA_FUZZ_FORK", "0")
        try:
            fork_count = max(0, min(int(fork_count_raw), os.cpu_count() or 1))
        except (ValueError, TypeError):
            fork_count = 0
        if fork_count > 1:
            cmd.append(f"-fork={fork_count}")
            cmd.append("-ignore_crashes=1")
            print(f"[*] Fork mode enabled: {fork_count} workers")

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
            line_callback=_line_callback,
            track_for_early_stop=True,
        )

        # Dump the tail for quick reading.
        log = (out + "\n=== STDERR ===\n" + err).replace("\r", "\n")
        tail = "\n".join(log.splitlines()[-200:])
        print(tail)

        libfuzzer_stats = parse_libfuzzer_final_stats(log)
        seed_bootstrap = dict(self.last_seed_bootstrap_by_fuzzer.get(bin_path.name) or {})
        seed_family_coverage = dict(seed_bootstrap.get("seed_family_coverage") or {})
        required_families = list(seed_bootstrap.get("seed_families_required") or [])
        covered_families = list(seed_family_coverage.get("covered") or [])

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
            oom_like_artifacts = [
                p for p in sorted_artifacts if p.name.startswith(("oom-", "oom-alloc-"))
            ]
            timeout_artifact_count = len(timeout_like_artifacts)
            crash_like_artifacts = [
                p for p in sorted_artifacts if p not in timeout_like_artifacts and p not in oom_like_artifacts
            ]

            if crash_like_artifacts:
                crash_evidence = "artifact"
                first_artifact = str(crash_like_artifacts[0])
            elif timeout_like_artifacts:
                # timeout-/slow-unit-* are performance/hang signals, not stable crash proof.
                crash_evidence = "timeout_artifact"
                first_artifact = str(timeout_like_artifacts[0])
            elif oom_like_artifacts:
                # oom-* indicates resource exhaustion, not a memory safety crash proof.
                crash_evidence = "oom_artifact"
                first_artifact = str(oom_like_artifacts[0])

        def _is_sanitizer_crash(text: str) -> bool:
            if not text:
                return False
            if re.search(r"ERROR:\s*libFuzzer:\s*out-of-memory", text, re.IGNORECASE):
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
            # Timeout-like artifacts are not memory-safety crash proof.
            # Keep strict mode opt-in for users who want to fail fast on hangs.
            strict_timeout_artifacts = _env_bool("SHERPA_STRICT_TIMEOUT_ARTIFACTS", False)
            run_error_kind = "run_timeout" if strict_timeout_artifacts else ""
            if strict_timeout_artifacts:
                error = (
                    f"fuzzer produced timeout-like artifacts for {bin_path.name} "
                    f"(count={timeout_artifact_count})"
                )
            else:
                print(
                    f"[warn] timeout-like artifacts found for {bin_path.name} "
                    f"(count={timeout_artifact_count}); continuing as non-fatal"
                )
        if crash_evidence == "oom_artifact":
            run_error_kind = "run_resource_exhaustion"
            error = f"fuzzer produced oom-like artifacts for {bin_path.name}"
        if rc != 0 and not crash_found:
            lowered = log.lower()
            if "[callback-stop] coverage_plateau" in lowered:
                rc = 0
            elif "error: libfuzzer: out-of-memory" in lowered:
                if not run_error_kind:
                    run_error_kind = "run_resource_exhaustion"
                if not error:
                    error = f"fuzzer hit resource exhaustion (out-of-memory) for {bin_path.name}"
            elif "idle-timeout" in lowered:
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

        plateau_detected = "[callback-stop] coverage_plateau" in log.lower()
        plateau_idle_seconds = plateau_idle_growth_sec if plateau_detected else 0

        # Corpus minimization after run (non-fatal)
        corpus_min_stats: Dict[str, int] = {}
        if not crash_found and corpus_files >= 4:
            try:
                corpus_min_stats = self._minimize_corpus(bin_path, corpus_dir)
            except Exception as exc:
                print(f"[warn] corpus minimization skipped: {exc}")

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
            terminal_reason="coverage_plateau" if plateau_detected else "",
            plateau_detected=plateau_detected,
            plateau_idle_seconds=plateau_idle_seconds,
            seed_quality=_seed_quality_from_run(
                log=log,
                initial_corpus_files=initial_corpus_files,
                initial_corpus_bytes=initial_corpus_bytes,
                final_stats=libfuzzer_stats,
                required_families=required_families,
                covered_families=covered_families,
                repo_examples_count=int((seed_bootstrap.get("counts") or {}).get("repo_examples") or 0),
                plateau_idle_seconds=plateau_idle_seconds,
                seed_profile=str(seed_bootstrap.get("seed_profile") or ""),
                archive_valid_count=int(seed_bootstrap.get("archive_valid_count") or 0),
                archive_valid_ratio=float(seed_bootstrap.get("archive_valid_ratio") or 1.0),
                archive_min_valid_ratio=self._seed_archive_min_valid_ratio(),
                archive_malformed_ratio=float(seed_bootstrap.get("archive_malformed_ratio") or 0.0),
                archive_max_malformed_ratio=float(
                    seed_bootstrap.get("archive_max_malformed_ratio") or self._seed_archive_max_malformed_ratio()
                ),
            ),
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
                print(f"[*] (host/git) Cloning {clone_url} → {dest}")
                clone_success = False
                for attempt in range(1, max(1, GIT_CLONE_RETRIES) + 1):
                    try:
                        if dest.exists():
                            shutil.rmtree(dest, ignore_errors=True)
                    except Exception:
                        pass

                    proxy_overrides = _host_git_proxy_override_args()
                    if proxy_overrides:
                        print("[warn] (host/git) detected broken localhost proxy; disabling git http(s).proxy for this operation")
                    clone_cmd = ["git", *proxy_overrides, "clone", "--depth", "1", clone_url, str(dest)]
                    print(f"[*] ➜  {' '.join(clone_cmd)}")
                    rc, out, err, timed_out = _run_cmd_capture(
                        clone_cmd,
                        timeout=GIT_HOST_CLONE_TIMEOUT_SEC,
                        env=_host_git_proxy_env(),
                    )
                    last_rc = rc

                    if timed_out:
                        if (t := _tail_lines(err)):
                            print("[warn] (host/git) clone stderr (tail):\n" + textwrap.indent(t, "    "))
                        if (t := _tail_lines(out)):
                            print("[warn] (host/git) clone stdout (tail):\n" + textwrap.indent(t, "    "))
                        print(
                            f"[warn] (host/git) clone timed out (url={clone_url}, attempt {attempt}/{max(1, GIT_CLONE_RETRIES)}, timeout={GIT_HOST_CLONE_TIMEOUT_SEC}s); retrying..."
                        )
                        time.sleep(
                            _retry_backoff_seconds(
                                attempt,
                                base_env="SHERPA_GIT_RETRY_BASE_SEC",
                                cap_env="SHERPA_GIT_RETRY_MAX_SEC",
                                default_base=0.5,
                                default_cap=3.0,
                            )
                        )
                        continue

                    if rc == 0:
                        clone_success = True
                        break

                    if (t := _tail_lines(err)):
                        print("[warn] (host/git) clone stderr (tail):\n" + textwrap.indent(t, "    "))
                    if (t := _tail_lines(out)):
                        print("[warn] (host/git) clone stdout (tail):\n" + textwrap.indent(t, "    "))
                    print(
                        f"[warn] (host/git) clone failed (url={clone_url}, attempt {attempt}/{max(1, GIT_CLONE_RETRIES)}, rc={rc}); retrying..."
                    )
                    time.sleep(
                        _retry_backoff_seconds(
                            attempt,
                            base_env="SHERPA_GIT_RETRY_BASE_SEC",
                            cap_env="SHERPA_GIT_RETRY_MAX_SEC",
                            default_base=0.5,
                            default_cap=3.0,
                        )
                    )

                if clone_success:
                    break
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
                crc, cout, cerr, _ = _run_cmd_capture(checkout_cmd, env=_host_git_proxy_env())
                if crc != 0:
                    if (t := _tail_lines(cerr)):
                        print("[warn] (host/git) checkout stderr (tail):\n" + textwrap.indent(t, "    "))
                    if (t := _tail_lines(cout)):
                        print("[warn] (host/git) checkout stdout (tail):\n" + textwrap.indent(t, "    "))
                    fetch_cmd = ["git", *proxy_overrides, "-C", str(dest), "fetch", "origin", spec.ref]
                    print(f"[*] ➜  {' '.join(fetch_cmd)}")
                    frc, fout, ferr, _ = _run_cmd_capture(fetch_cmd, env=_host_git_proxy_env())
                    if frc != 0:
                        if (t := _tail_lines(ferr)):
                            print("[warn] (host/git) fetch stderr (tail):\n" + textwrap.indent(t, "    "))
                        if (t := _tail_lines(fout)):
                            print("[warn] (host/git) fetch stdout (tail):\n" + textwrap.indent(t, "    "))
                        raise HarnessGeneratorError(f"git fetch failed on host (rc={frc}).")
                    checkout_fh = ["git", *proxy_overrides, "-C", str(dest), "checkout", "FETCH_HEAD"]
                    print(f"[*] ➜  {' '.join(checkout_fh)}")
                    c2rc, c2out, c2err, _ = _run_cmd_capture(checkout_fh, env=_host_git_proxy_env())
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
                        time.sleep(
                            _retry_backoff_seconds(
                                attempt,
                                base_env="SHERPA_GIT_RETRY_BASE_SEC",
                                cap_env="SHERPA_GIT_RETRY_MAX_SEC",
                                default_base=0.5,
                                default_cap=3.0,
                            )
                        )
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
                    time.sleep(
                        _retry_backoff_seconds(
                            attempt,
                            base_env="SHERPA_GIT_RETRY_BASE_SEC",
                            cap_env="SHERPA_GIT_RETRY_MAX_SEC",
                            default_base=0.5,
                            default_cap=3.0,
                        )
                    )

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
                        time.sleep(
                            _retry_backoff_seconds(
                                attempt,
                                base_env="SHERPA_GIT_RETRY_BASE_SEC",
                                cap_env="SHERPA_GIT_RETRY_MAX_SEC",
                                default_base=0.5,
                                default_cap=3.0,
                            )
                        )
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

        # Fallback: clone on host using git CLI.
        # This keeps clone behaviour consistent with retries/mirror/proxy logic
        # used in docker clone path and avoids direct GitPython github.com calls.
        try:
            return _clone_with_host_git(root)
        except FileNotFoundError:
            raise HarnessGeneratorError(
                "'git' is not found in PATH. "
                "Enable Docker mode (--docker-image auto) or install Git."
            )

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

    def _register_active_run_process(self, proc: subprocess.Popen[str]) -> None:
        with self._active_run_procs_lock:
            self._active_run_procs.add(proc)

    def _unregister_active_run_process(self, proc: subprocess.Popen[str]) -> None:
        with self._active_run_procs_lock:
            self._active_run_procs.discard(proc)

    def terminate_active_run_processes(self, *, reason: str = "") -> int:
        """Terminate currently active run subprocesses (best effort)."""
        with self._active_run_procs_lock:
            procs = list(self._active_run_procs)
        if not procs:
            return 0
        stopped = 0
        for proc in procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    stopped += 1
            except Exception:
                continue
        # Escalate to kill for stubborn processes.
        deadline = time.monotonic() + 2.0
        for proc in procs:
            while time.monotonic() < deadline:
                try:
                    if proc.poll() is not None:
                        break
                except Exception:
                    break
                time.sleep(0.05)
            try:
                if proc.poll() is None:
                    proc.kill()
            except Exception:
                pass
        if reason:
            print(f"[*] terminate_active_run_processes reason={reason} count={stopped}", flush=True)
        return stopped

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
        line_callback: Optional[Callable[[str, str], Optional[str]]] = None,
        track_for_early_stop: bool = False,
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

        def _redact_text(text: str) -> str:
            if not text:
                return text
            out = text
            for key in (
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
                "DEEPSEEK_API_KEY",
                "MINIMAX_API_KEY",
                "ANTHROPIC_API_KEY",
                "DATABASE_URL",
                "POSTGRES_PASSWORD",
            ):
                val = str((env or {}).get(key) or os.environ.get(key) or "").strip()
                if val:
                    out = out.replace(val, "***")
            out = re.sub(
                r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASS))\s*=\s*([^\s,;]+)",
                lambda m: f"{m.group(1)}=***",
                out,
            )
            out = re.sub(r"(?i)\b(Authorization\s*:\s*Bearer\s+)([^\s]+)", r"\1***", out)
            return out

        if extra_inputs:
            # Append corpus/extra inputs directly. libFuzzer treats a bare "--" as an ignored flag
            # and logs noisy warnings that can look like failures in UI streams.
            cmd = list(cmd)
            cmd.extend(extra_inputs)

        effective_env = env or os.environ.copy()
        exec_args = list(cmd)
        if any(self._is_build_entry_arg(str(a)) for a in exec_args):
            self._sanitize_build_py_for_source_build_collision()
            self._sanitize_build_py_for_non_root_install()
        dep_rel = (FUZZ_SYSTEM_PACKAGES_FILE or "fuzz/system_packages.txt").replace("\\", "/").strip("/")
        if self.docker_image:
            actual_cmd = self._dockerize_cmd(exec_args, cwd=cwd, env=effective_env)
        else:
            actual_cmd = self._wrap_exec_with_runtime_prelude(
                exec_args,
                dep_file=str((self.repo_root / dep_rel).resolve()),
                dep_log_prefix="native/deps",
            )

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
        if track_for_early_stop:
            self._register_active_run_process(proc)

        stdout_chunks: List[str] = []
        stderr_chunks: List[str] = []
        reader_eof = object()
        out_queue: queue.Queue[tuple[str, str] | object] = queue.Queue()

        def _reader(pipe: Optional[object], kind: str) -> None:
            try:
                if pipe is None:
                    return
                for line in pipe:
                    out_queue.put((kind, line))
            finally:
                out_queue.put(reader_eof)

        t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
        t_out.start()
        t_err.start()

        done_readers = 0
        timed_out = False
        idle_timed_out = False
        callback_stop_reason = ""
        heartbeat_raw = (os.environ.get("SHERPA_CMD_KEEPALIVE_SEC") or "0").strip()
        try:
            heartbeat_sec = max(0.0, min(float(heartbeat_raw), 3600.0))
        except Exception:
            heartbeat_sec = 0.0
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
                    # Queue timeout only means "no new output yet", not reader EOF.
                    continue

                if item is reader_eof:
                    done_readers += 1
                else:
                    kind, text = item
                    safe_text = _redact_text(text)
                    if kind == "stdout":
                        stdout_chunks.append(safe_text)
                        print(safe_text, end="", flush=True)
                    else:
                        stderr_chunks.append(safe_text)
                        # Keep stderr visible in real-time to avoid silent failures.
                        print(safe_text, end="", flush=True)
                    last_activity = time.monotonic()
                    if line_callback is not None:
                        try:
                            callback_stop_reason = str(line_callback(kind, safe_text) or "").strip()
                        except Exception:
                            callback_stop_reason = ""
                        if callback_stop_reason:
                            timed_out = True
                            break

                if idle_timeout > 0 and (time.monotonic() - last_activity) > idle_timeout:
                    idle_timed_out = True
                    timed_out = True
                    break

                if heartbeat_sec > 0:
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
                if item is reader_eof:
                    continue
                kind, text = item
                safe_text = _redact_text(text)
                if kind == "stdout":
                    stdout_chunks.append(safe_text)
                else:
                    stderr_chunks.append(safe_text)
            if track_for_early_stop:
                self._unregister_active_run_process(proc)

        out = "".join(stdout_chunks)
        err = "".join(stderr_chunks)
        if callback_stop_reason:
            err = (err or "") + f"\n[callback-stop] {callback_stop_reason}"
        elif idle_timed_out:
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
            # Do not hard-fail crash analysis when agent skipped reproducer generation.
            # Downstream packaging can still classify this run as unreproducible.
            print("[warn] No reproducer script found after agent generation; mark as unreproducible")
            return False

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
