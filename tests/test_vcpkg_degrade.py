from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "harness_generator" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fuzz_unharnessed_repo as m  # noqa: E402


def _render_dep_setup(dep_file: str) -> str:
    cls = m.NonOssFuzzHarnessGenerator
    inst = cls.__new__(cls)  # bypass __init__ (no repo needed for the template)
    inst._vcpkg_triplet = lambda: "x64-linux"  # type: ignore[attr-defined]
    return cls._build_system_dep_setup(inst, dep_file=dep_file, log_prefix="native/deps")


def _run(script: str, env_extra: dict[str, str]):
    import os

    env = dict(os.environ)
    # ensure vcpkg cannot be bootstrapped during the test (no git, no network)
    env["SHERPA_VCPKG_GIT_BIN"] = "/nonexistent-git-binary"
    env.update(env_extra)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)


@pytest.fixture()
def declared_repo(tmp_path: Path) -> str:
    fuzz = tmp_path / "fuzz"
    fuzz.mkdir(parents=True)
    dep = fuzz / "system_packages.txt"
    dep.write_text("libfoo-nonexistent\n", encoding="utf-8")
    return str(dep)


def test_shell_is_valid(declared_repo: str):
    sh = _render_dep_setup(declared_repo)
    r = subprocess.run(["bash", "-n", "-c", sh], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_degrades_to_best_effort_by_default(declared_repo: str):
    # vcpkg cannot be provisioned + a port is declared -> must NOT hard-fail;
    # toolchain env must be cleared so the project's own build can proceed.
    sh = _render_dep_setup(declared_repo)
    probe = sh + '\necho "TOOLCHAIN=[${CMAKE_TOOLCHAIN_FILE:-UNSET}]"; echo "PKGS=[$pkgs]"\n'
    r = _run(probe, {})  # SHERPA_VCPKG_STRICT unset -> default 0
    assert r.returncode == 0, r.stdout + r.stderr
    assert "continuing without vcpkg" in r.stdout
    assert "TOOLCHAIN=[UNSET]" in r.stdout
    assert "PKGS=[]" in r.stdout


def test_strict_mode_hard_fails(declared_repo: str):
    sh = _render_dep_setup(declared_repo)
    r = _run(sh, {"SHERPA_VCPKG_STRICT": "1"})
    assert r.returncode == 86
    assert "vcpkg unavailable" in r.stdout


def test_no_ports_is_noop(tmp_path: Path):
    # absent system_packages.txt -> no vcpkg work, clean exit
    sh = _render_dep_setup(str(tmp_path / "fuzz" / "system_packages.txt"))
    r = _run(sh, {})
    assert r.returncode == 0, r.stdout + r.stderr
