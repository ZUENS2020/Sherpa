from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "harness_generator" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fuzz_unharnessed_repo import NonOssFuzzHarnessGenerator


def _fake_generator(repo_root: Path) -> NonOssFuzzHarnessGenerator:
    gen = NonOssFuzzHarnessGenerator.__new__(NonOssFuzzHarnessGenerator)
    gen.repo_root = repo_root
    gen.docker_image = None
    return gen


def test_run_cmd_keeps_stream_loop_open_while_process_is_silent(tmp_path: Path):
    gen = _fake_generator(tmp_path)
    script = (
        "import sys,time;"
        "time.sleep(1.7);"
        "print('late-out', flush=True);"
        "print('late-err', file=sys.stderr, flush=True)"
    )

    rc, out, err = gen._run_cmd(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=10,
        idle_timeout=0,
    )

    assert rc == 0
    assert "late-out" in out
    assert "late-err" in err


def test_run_cmd_native_autoinstalls_declared_system_packages_for_build_entry(tmp_path: Path):
    gen = _fake_generator(tmp_path)
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "system_packages.txt").write_text("cmake-data\n", encoding="utf-8")

    log_path = tmp_path / "apt.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    apt_script = bin_dir / "apt-get"
    apt_script.write_text(
        "#!/bin/sh\n"
        f"echo \"$@\" >> {log_path}\n"
        "exit 0\n",
        encoding="utf-8",
    )
    apt_script.chmod(0o755)

    dpkg_query = bin_dir / "dpkg-query"
    dpkg_query.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    dpkg_query.chmod(0o755)

    build_script = fuzz_dir / "build.sh"
    build_script.write_text("#!/bin/sh\necho native-build-ok\n", encoding="utf-8")
    build_script.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["SHERPA_AUTO_INSTALL_SYSTEM_DEPS"] = "1"

    rc, out, err = gen._run_cmd(
        ["./build.sh"],
        cwd=fuzz_dir,
        env=env,
        timeout=10,
        idle_timeout=0,
    )

    assert rc == 0
    assert "native-build-ok" in out
    log_text = log_path.read_text(encoding="utf-8")
    assert "update -o Acquire::Retries=3 -o Acquire::ForceIPv4=true" in log_text
    assert "install -y --no-install-recommends cmake-data" in log_text
