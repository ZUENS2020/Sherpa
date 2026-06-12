from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

# main.py pulls fastapi; import the function in isolation via the module only if
# available, otherwise re-implement the contract test against the source. Here we
# import lazily and skip if the web deps are unavailable.
import importlib

import pytest

try:
    main = importlib.import_module("main")
except Exception:  # pragma: no cover - depends on optional web deps
    main = None

pytestmark = pytest.mark.skipif(main is None, reason="web deps (fastapi) unavailable")


# Echoed deps-bootstrap script SOURCE lines that previously polluted the error
# stream because they contain literal 'error'/'failed'/'warn' substrings.
SCRIPT_SOURCE = [
    '                        echo "[error] (native/deps) vcpkg unavailable while required ports are declared in $dep_file"',
    '                    echo "[warn] (native/deps) vcpkg bootstrap failed"',
    '                        _vcpkg_degrade "vcpkg install failed for:$missing_pkgs"',
    '                    echo "[error] (native/deps) missing vcpkg toolchain file: $vcpkg_root/scripts/buildsystems/vcpkg.cmake"',
    '                install_max="${SHERPA_VCPKG_INSTALL_RETRIES:-3}"',
    "                if grep -Eiq 'vcpkg-running[.]lock|failed to take lock' \"$install_log\"; then",
]

# Real runtime log lines that MUST still classify as error/warn.
REAL_ERRORS = [
    "Traceback (most recent call last):",
    "RuntimeError: unknown_error: synthesize failed",
    "fuzz/libarchive_fuzzer.cc:22:10: error: no member named 'unique_ptr'",
    "[error] (native/deps) vcpkg unavailable while required ports are declared in /work/fuzz/system_packages.txt",
    "[job d379] stage synthesize failed (k8s_job_failed): unknown_error",
]
REAL_WARN = [
    "libpng warning: iCCP: known incorrect sRGB profile",
    "[wf] retrying after transient failure",
]


def test_script_source_not_misclassified_as_error():
    for line in SCRIPT_SOURCE:
        assert main._classify_log_level(line) == "info", line


def test_real_errors_still_error():
    for line in REAL_ERRORS:
        assert main._classify_log_level(line) == "error", line


def test_real_warnings_still_warn():
    for line in REAL_WARN:
        assert main._classify_log_level(line) == "warn", line


def test_plain_info_unchanged():
    assert main._classify_log_level("[wf step=11] synthesize completed dt=12s") == "info"
