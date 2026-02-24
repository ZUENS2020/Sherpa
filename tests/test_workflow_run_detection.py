from __future__ import annotations

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph
from fuzz_unharnessed_repo import FuzzerRunResult


class _FakeRunGenerator:
    def __init__(self, tmp_path: Path, run_results: list[FuzzerRunResult]) -> None:
        self.repo_root = tmp_path
        self.fuzz_out_dir = tmp_path / "fuzz" / "out"
        self.fuzz_out_dir.mkdir(parents=True, exist_ok=True)
        self._bin = self.fuzz_out_dir / "demo_fuzz"
        self._bin.write_text("", encoding="utf-8")
        self._run_results = list(run_results)
        self.analysis_calls: list[tuple[str, Path]] = []

    def _discover_fuzz_binaries(self) -> list[Path]:
        return [self._bin]

    def _pass_generate_seeds(self, _fuzzer_name: str) -> None:
        return

    def _run_fuzzer(self, _bin_path: Path) -> FuzzerRunResult:
        if not self._run_results:
            raise AssertionError("unexpected _run_fuzzer call")
        return self._run_results.pop(0)

    def _analyze_and_package(self, fuzzer_name: str, artifact: Path) -> None:
        self.analysis_calls.append((fuzzer_name, artifact))


class _SlowSeedGenerator(_FakeRunGenerator):
    def __init__(self, tmp_path: Path, run_results: list[FuzzerRunResult], *, seed_sleep_sec: float) -> None:
        super().__init__(tmp_path, run_results)
        self._bins = [self.fuzz_out_dir / "demo_fuzz_1", self.fuzz_out_dir / "demo_fuzz_2"]
        for p in self._bins:
            p.write_text("", encoding="utf-8")
        self._seed_sleep_sec = seed_sleep_sec

    def _discover_fuzz_binaries(self) -> list[Path]:
        return list(self._bins)

    def _pass_generate_seeds(self, _fuzzer_name: str) -> None:
        time.sleep(self._seed_sleep_sec)


def test_node_run_marks_error_when_fuzzer_exits_nonzero_without_crash(tmp_path: Path):
    gen = _FakeRunGenerator(
        tmp_path,
        run_results=[
            FuzzerRunResult(
                rc=127,
                new_artifacts=[],
                crash_found=False,
                crash_evidence="none",
                first_artifact="",
                log_tail="sh: exec fuzzer: not found",
                error="fuzzer run failed rc=127 for demo_fuzz; no crash artifact/sanitizer evidence found",
                run_error_kind="nonzero_exit_without_crash",
            )
        ],
    )

    out = workflow_graph._node_run({"generator": gen, "crash_fix_attempts": 0})

    assert out["last_step"] == "run"
    assert out["crash_found"] is False
    assert "rc=127" in out["last_error"]
    assert out["run_rc"] == 127
    assert out["crash_evidence"] == "none"
    assert out["run_error_kind"] == "nonzero_exit_without_crash"


def test_node_run_accepts_sanitizer_log_crash_without_native_artifact(tmp_path: Path):
    artifact = tmp_path / "fuzz" / "out" / "artifacts" / "crash-log-1.txt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("ERROR: AddressSanitizer: heap-use-after-free", encoding="utf-8")

    gen = _FakeRunGenerator(
        tmp_path,
        run_results=[
            FuzzerRunResult(
                rc=76,
                new_artifacts=[artifact],
                crash_found=True,
                crash_evidence="sanitizer_log",
                first_artifact=str(artifact),
                log_tail="ERROR: AddressSanitizer: heap-use-after-free",
                error="",
                run_error_kind="",
            )
        ],
    )

    out = workflow_graph._node_run({"generator": gen, "crash_fix_attempts": 0})

    assert out["last_step"] == "run"
    assert out["last_error"] == ""
    assert out["crash_found"] is True
    assert out["run_rc"] == 76
    assert out["crash_evidence"] == "sanitizer_log"
    assert out["last_crash_artifact"] == str(artifact)
    assert gen.analysis_calls == [("demo_fuzz", artifact)]


def test_node_run_emits_run_details_metrics(tmp_path: Path):
    gen = _FakeRunGenerator(
        tmp_path,
        run_results=[
            FuzzerRunResult(
                rc=0,
                new_artifacts=[],
                crash_found=False,
                crash_evidence="none",
                first_artifact="",
                log_tail="ok",
                error="",
                run_error_kind="",
                final_cov=321,
                final_ft=654,
                final_corpus_files=12,
                final_corpus_size_bytes=2048,
                final_execs_per_sec=777,
                final_rss_mb=88,
                final_iteration=999,
                corpus_files=10,
                corpus_size_bytes=1024,
            )
        ],
    )

    out = workflow_graph._node_run({"generator": gen, "crash_fix_attempts": 0})

    assert out["last_step"] == "run"
    assert out["last_error"] == ""
    details = out.get("run_details") or []
    assert len(details) == 1
    detail = details[0]
    assert detail["fuzzer"] == "demo_fuzz"
    assert detail["final_cov"] == 321
    assert detail["final_ft"] == 654
    assert detail["final_corpus_files"] == 12
    assert detail["final_execs_per_sec"] == 777


def test_node_run_stops_when_total_budget_exhausted_during_seed_generation(tmp_path: Path):
    gen = _SlowSeedGenerator(
        tmp_path,
        run_results=[
            FuzzerRunResult(
                rc=0,
                new_artifacts=[],
                crash_found=False,
                crash_evidence="none",
                first_artifact="",
                log_tail="ok",
                error="",
                run_error_kind="",
            ),
            FuzzerRunResult(
                rc=0,
                new_artifacts=[],
                crash_found=False,
                crash_evidence="none",
                first_artifact="",
                log_tail="ok",
                error="",
                run_error_kind="",
            ),
        ],
        seed_sleep_sec=1.2,
    )
    started = time.time()
    out = workflow_graph._node_run(
        {
            "generator": gen,
            "crash_fix_attempts": 0,
            "workflow_started_at": started,
            "time_budget": 1,
            "run_time_budget": 300,
        }
    )
    assert out["last_step"] == "run"
    assert out["failed"] is True
    assert "time budget exceeded" in out["last_error"]
    assert out["message"] == "workflow stopped (time budget exceeded)"
