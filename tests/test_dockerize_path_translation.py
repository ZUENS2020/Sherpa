from __future__ import annotations

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
    gen.docker_image = "sherpa-fuzz-cpp:latest"
    return gen


def test_dockerize_translates_fuzzer_binary_and_artifact_prefix_before_exec(tmp_path: Path):
    repo_root = tmp_path / "repo"
    bin_path = repo_root / "fuzz" / "out" / "fuzzer"
    corpus_dir = repo_root / "fuzz" / "corpus" / "fuzzer"
    artifacts_dir = repo_root / "fuzz" / "out" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    gen = _fake_generator(repo_root)
    cmd = [
        str(bin_path),
        f"-artifact_prefix={artifacts_dir}/",
        "-print_final_stats=1",
        "-max_total_time=5",
        "--",
        str(corpus_dir),
    ]

    docker_cmd = gen._dockerize_cmd(cmd, cwd=repo_root, env={"ASAN_OPTIONS": "exitcode=76"})
    joined = " ".join(docker_cmd)

    assert str(bin_path) not in joined
    assert str(corpus_dir) not in joined
    assert str(artifacts_dir) not in joined

    assert "/work/fuzz/out/fuzzer" in joined
    assert "-artifact_prefix=/work/fuzz/out/artifacts/" in joined
    assert "/work/fuzz/corpus/fuzzer" in joined


def test_dockerize_autoinstall_triggers_for_build_py_from_fuzz_cwd(tmp_path: Path, monkeypatch):
    repo_root = tmp_path / "repo"
    fuzz_dir = repo_root / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)

    gen = _fake_generator(repo_root)
    monkeypatch.setenv("SHERPA_AUTO_INSTALL_SYSTEM_DEPS", "1")

    docker_cmd = gen._dockerize_cmd(["python3", "build.py"], cwd=fuzz_dir, env={})
    joined = " ".join(docker_cmd)

    assert "-w /work/fuzz" in joined
    assert "dep_file=/work/fuzz/system_packages.txt" in joined
    assert "set -u" in joined
    assert "(docker/deps) installing from" in joined
    assert "continuing without auto-install" in joined
    assert "exec python3 build.py" in joined
