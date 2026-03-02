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
