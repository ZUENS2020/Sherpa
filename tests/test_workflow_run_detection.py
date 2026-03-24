from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace


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
        self.seed_calls: int = 0

    def _discover_fuzz_binaries(self) -> list[Path]:
        return [self._bin]

    def _pass_generate_seeds(self, _fuzzer_name: str) -> None:
        self.seed_calls += 1
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
        self.seed_calls += 1
        time.sleep(self._seed_sleep_sec)


class _MultiRunGenerator(_FakeRunGenerator):
    def __init__(self, tmp_path: Path, run_results: list[FuzzerRunResult], *, run_sleep_sec: float = 0.0) -> None:
        super().__init__(tmp_path, run_results)
        self._bins = [self.fuzz_out_dir / "demo_fuzz_1", self.fuzz_out_dir / "demo_fuzz_2", self.fuzz_out_dir / "demo_fuzz_3"]
        for p in self._bins:
            p.write_text("", encoding="utf-8")
        self._run_sleep_sec = run_sleep_sec

    def _discover_fuzz_binaries(self) -> list[Path]:
        return list(self._bins)

    def _run_fuzzer(self, _bin_path: Path) -> FuzzerRunResult:
        if self._run_sleep_sec > 0:
            time.sleep(self._run_sleep_sec)
        return super()._run_fuzzer(_bin_path)


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


def test_node_run_writes_repro_context_on_crash(tmp_path: Path):
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
                log_tail="asan log",
                error="",
                run_error_kind="",
            )
        ],
    )

    out = workflow_graph._node_run(
        {
            "generator": gen,
            "repo_url": "https://github.com/fmtlib/fmt.git",
            "crash_fix_attempts": 0,
        }
    )

    ctx = workflow_graph._read_repro_context(tmp_path)
    assert out["crash_found"] is True
    assert ctx["repo_url"] == "https://github.com/fmtlib/fmt.git"
    assert ctx["last_fuzzer"] == "demo_fuzz"
    assert ctx["last_crash_artifact"] == str(artifact)
    assert ctx["crash_signature"]
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
    assert isinstance(out.get("coverage_seed_feedback"), dict)
    assert isinstance(out.get("coverage_harness_feedback"), dict)


def test_node_run_stops_when_total_budget_exhausted_during_seed_generation(tmp_path: Path, monkeypatch):
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
            "time_budget": 2,
            "run_time_budget": 300,
        }
    )
    assert out["last_step"] == "run"
    assert out["failed"] is True
    assert "time budget exceeded" in out["last_error"]
    assert out["message"] == "workflow stopped (time budget exceeded)"


def test_node_run_default_generates_ai_seeds(tmp_path: Path):
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
        seed_sleep_sec=1.5,
    )
    out = workflow_graph._node_run({"generator": gen, "crash_fix_attempts": 0})
    assert out["last_step"] == "run"
    assert out.get("failed") is not True
    assert gen.seed_calls == 2


def test_node_run_records_stable_parallel_batch_plan(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SHERPA_PARALLEL_FUZZERS", "2")
    monkeypatch.setenv("SHERPA_RUN_STOP_ON_FIRST_CRASH", "0")
    gen = _MultiRunGenerator(
        tmp_path,
        run_results=[
            FuzzerRunResult(rc=0, new_artifacts=[], crash_found=False, crash_evidence="none", first_artifact="", log_tail="ok", error="", run_error_kind=""),
            FuzzerRunResult(rc=0, new_artifacts=[], crash_found=False, crash_evidence="none", first_artifact="", log_tail="ok", error="", run_error_kind=""),
            FuzzerRunResult(rc=0, new_artifacts=[], crash_found=False, crash_evidence="none", first_artifact="", log_tail="ok", error="", run_error_kind=""),
        ],
    )

    out = workflow_graph._node_run(
        {"generator": gen, "crash_fix_attempts": 0, "workflow_started_at": time.time(), "time_budget": 120, "run_time_budget": 120}
    )
    plan = out.get("run_batch_plan") or []

    assert len(plan) == 2
    assert plan[0]["batch_size"] == 2
    assert plan[0]["pending_before"] == 3
    assert plan[0]["rounds_left"] == 2
    # First-round budget is derived from remaining total budget; allow runtime jitter.
    assert 1 <= int(plan[0]["round_budget_sec"]) <= 120
    assert plan[1]["batch_size"] == 1
    assert plan[1]["rounds_left"] == 1
    assert plan[1]["round_budget_sec"] >= plan[0]["round_budget_sec"]


def test_node_run_stops_after_first_crash_by_default(tmp_path: Path):
    artifact = tmp_path / "fuzz" / "out" / "artifacts" / "crash-1"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("asan", encoding="utf-8")

    gen = _MultiRunGenerator(
        tmp_path,
        run_results=[
            FuzzerRunResult(
                rc=76,
                new_artifacts=[artifact],
                crash_found=True,
                crash_evidence="artifact",
                first_artifact=str(artifact),
                log_tail="asan",
                error="",
                run_error_kind="",
            ),
        ],
    )

    out = workflow_graph._node_run({"generator": gen, "crash_fix_attempts": 0})

    assert out["crash_found"] is True
    assert out["last_crash_artifact"] == str(artifact)
    assert out["last_fuzzer"] == "demo_fuzz_1"
    assert len(out.get("run_details") or []) == 1
    assert gen.analysis_calls == [("demo_fuzz_1", artifact)]
    assert len(gen._run_results) == 0


def test_node_run_marks_budget_exhausted_when_run_phase_times_out(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SHERPA_PARALLEL_FUZZERS", "1")
    gen = _MultiRunGenerator(
        tmp_path,
        run_results=[
            FuzzerRunResult(rc=0, new_artifacts=[], crash_found=False, crash_evidence="none", first_artifact="", log_tail="ok", error="", run_error_kind=""),
        ],
        run_sleep_sec=2.2,
    )

    started = time.time()
    out = workflow_graph._node_run(
        {
            "generator": gen,
            "crash_fix_attempts": 0,
            "workflow_started_at": started,
            "time_budget": 2,
            "run_time_budget": 300,
        }
    )

    assert out["last_step"] == "run"
    assert out["failed"] is True
    assert out["run_error_kind"] == "workflow_time_budget_exceeded"
    assert out["message"] == "workflow stopped (time budget exceeded)"
    details = out.get("run_details") or []
    assert len(details) == 3
    assert details[1]["run_error_kind"] == "run_exception"
    assert details[1]["error"].startswith("skipped: workflow total time budget exhausted")


def test_node_run_marks_no_progress_for_execs_zero_with_warning(tmp_path: Path):
    gen = _FakeRunGenerator(
        tmp_path,
        run_results=[
            FuzzerRunResult(
                rc=0,
                new_artifacts=[],
                crash_found=False,
                crash_evidence="none",
                first_artifact="",
                log_tail=(
                    "INFO: seed corpus: files: 5 min: 8b max: 19b total: 67b rss: 27Mb\n"
                    "#6\tINITED exec/s: 0 rss: 27Mb\n"
                    "WARNING: no interesting inputs were found so far."
                ),
                error="",
                run_error_kind="",
                final_execs_per_sec=0,
            )
        ],
    )

    out = workflow_graph._node_run({"generator": gen, "crash_fix_attempts": 0})

    assert out["last_step"] == "run"
    assert out["crash_found"] is False
    assert out["run_error_kind"] == "run_no_progress"
    assert "no measurable progress" in out["last_error"]


def test_route_after_run_routes_recoverable_run_errors_to_plan():
    route = workflow_graph._route_after_run_state(
        {"run_error_kind": "run_no_progress", "failed": False, "crash_found": False}
    )
    assert route == "plan"


def test_route_after_run_routes_crash_to_repro_stage():
    route = workflow_graph._route_after_run_state(
        {"run_error_kind": "", "failed": False, "crash_found": True}
    )
    assert route == "crash-triage"


def test_route_after_run_routes_clean_result_to_coverage_analysis():
    route = workflow_graph._route_after_run_state(
        {"run_error_kind": "", "failed": False, "crash_found": False}
    )
    assert route == "coverage-analysis"


def test_route_after_run_routes_idle_timeout_to_plan():
    route = workflow_graph._route_after_run_state(
        {"run_error_kind": "run_idle_timeout", "failed": False, "crash_found": False}
    )
    assert route == "plan"


def test_route_after_run_routes_resource_exhaustion_to_plan():
    route = workflow_graph._route_after_run_state(
        {"run_error_kind": "run_resource_exhaustion", "failed": False, "crash_found": False}
    )
    assert route == "plan"


def test_route_after_coverage_analysis_routes_to_improve_harness():
    route = workflow_graph._route_after_coverage_analysis_state(
        {"failed": False, "last_error": "", "coverage_should_improve": True}
    )
    assert route == "improve-harness"


def test_route_after_improve_harness_routes_back_to_plan():
    route = workflow_graph._route_after_improve_harness_state(
        {"failed": False, "last_error": "", "coverage_should_improve": True}
    )
    assert route == "plan"


def test_route_after_improve_harness_stops_on_ineffective_replan():
    route = workflow_graph._route_after_improve_harness_state(
        {
            "failed": False,
            "last_error": "",
            "coverage_should_improve": True,
            "coverage_improve_mode": "replan",
            "coverage_replan_effective": False,
        }
    )
    assert route == "stop"


def test_route_after_improve_harness_routes_to_build_for_in_place_improve():
    route = workflow_graph._route_after_improve_harness_state(
        {
            "failed": False,
            "last_error": "",
            "coverage_should_improve": True,
            "coverage_improve_mode": "in_place",
        }
    )
    assert route == "build"


def test_route_after_improve_harness_stops_when_round_budget_exhausted():
    route = workflow_graph._route_after_improve_harness_state(
        {
            "failed": False,
            "last_error": "",
            "coverage_should_improve": True,
            "coverage_improve_mode": "replan",
            "coverage_round_budget_exhausted": True,
        }
    )
    assert route == "stop"


def test_node_coverage_analysis_keeps_first_plateau_in_place():
    out = workflow_graph._node_coverage_analysis(
        {
            "coverage_loop_max_rounds": 3,
            "coverage_loop_round": 0,
            "coverage_history": [],
            "coverage_target_name": "yaml_parser_parse_fuzz",
            "coverage_seed_profile": "parser-structure",
            "run_details": [
                {
                    "fuzzer": "yaml_parser_parse_fuzz",
                    "final_cov": 7,
                    "final_ft": 28,
                    "plateau_detected": True,
                    "plateau_idle_seconds": 180,
                }
            ],
            "crash_found": False,
            "failed": False,
            "run_error_kind": "",
        }
    )

    assert out["coverage_should_improve"] is True
    assert out["coverage_improve_mode"] == "in_place"
    assert out["coverage_replan_required"] is False
    assert out["coverage_plateau_streak"] == 1


def test_node_coverage_analysis_replans_after_second_plateau_without_gain():
    out = workflow_graph._node_coverage_analysis(
        {
            "coverage_loop_max_rounds": 3,
            "coverage_loop_round": 1,
            "coverage_history": [],
            "coverage_target_name": "yaml_parser_parse_fuzz",
            "coverage_seed_profile": "parser-structure",
            "coverage_plateau_streak": 1,
            "coverage_last_max_cov": 7,
            "coverage_last_ft": 28,
            "run_details": [
                {
                    "fuzzer": "yaml_parser_parse_fuzz",
                    "final_cov": 7,
                    "final_ft": 28,
                    "plateau_detected": True,
                    "plateau_idle_seconds": 240,
                }
            ],
            "crash_found": False,
            "failed": False,
            "run_error_kind": "",
        }
    )

    assert out["coverage_should_improve"] is True
    assert out["coverage_improve_mode"] == "replan"
    assert out["coverage_replan_required"] is True
    assert out["coverage_plateau_streak"] == 2


def test_node_coverage_analysis_stops_when_replan_budget_exhausted():
    out = workflow_graph._node_coverage_analysis(
        {
            "coverage_loop_max_rounds": 3,
            "coverage_loop_round": 2,
            "coverage_history": [],
            "coverage_target_name": "yaml_parser_parse_fuzz",
            "coverage_seed_profile": "parser-structure",
            "coverage_plateau_streak": 1,
            "coverage_last_max_cov": 7,
            "coverage_last_ft": 28,
            "run_details": [
                {
                    "fuzzer": "yaml_parser_parse_fuzz",
                    "final_cov": 7,
                    "final_ft": 28,
                    "plateau_detected": True,
                    "plateau_idle_seconds": 240,
                }
            ],
            "crash_found": False,
            "failed": False,
            "run_error_kind": "",
        }
    )

    assert out["coverage_should_improve"] is False
    assert out["coverage_improve_mode"] == ""
    assert out["coverage_replan_required"] is False
    assert out["coverage_round_budget_exhausted"] is True
    assert out["coverage_stop_reason"] == "coverage_loop_budget_exhausted"
    assert "budget exhausted" in out["coverage_improve_reason"]


def test_node_coverage_analysis_allows_resource_exhaustion_to_improve():
    out = workflow_graph._node_coverage_analysis(
        {
            "coverage_loop_max_rounds": 3,
            "coverage_loop_round": 0,
            "coverage_history": [],
            "coverage_target_name": "yaml_parser_parse_fuzz",
            "coverage_seed_profile": "parser-structure",
            "run_details": [
                {
                    "fuzzer": "yaml_parser_parse_fuzz",
                    "final_cov": 5,
                    "final_ft": 12,
                    "plateau_detected": False,
                    "plateau_idle_seconds": 0,
                }
            ],
            "crash_found": False,
            "failed": False,
            "run_error_kind": "run_resource_exhaustion",
        }
    )

    assert out["coverage_should_improve"] is True
    assert out["coverage_improve_mode"] == "in_place"


def test_node_coverage_analysis_prioritizes_seed_quality_issue_over_replan():
    out = workflow_graph._node_coverage_analysis(
        {
            "coverage_loop_max_rounds": 3,
            "coverage_loop_round": 1,
            "coverage_history": [],
            "coverage_target_name": "yaml_parser_parse_fuzz",
            "coverage_target_api": "fmt::println",
            "coverage_seed_profile": "parser-structure",
            "coverage_seed_quality": {"quality_flags": ["missing_required_families", "repo_examples_missing", "target_runtime_mismatch"]},
            "coverage_quality_flags": ["missing_required_families", "repo_examples_missing", "target_runtime_mismatch"],
            "coverage_seed_families_required": ["flow_structures", "anchors_aliases"],
            "coverage_seed_families_covered": ["anchors_aliases"],
            "coverage_seed_families_missing": ["flow_structures"],
            "coverage_plateau_streak": 1,
            "coverage_last_max_cov": 5,
            "coverage_last_ft": 19,
            "run_details": [
                {
                    "fuzzer": "yaml_parser_parse_fuzz",
                    "final_cov": 5,
                    "final_ft": 19,
                    "plateau_detected": True,
                    "plateau_idle_seconds": 180,
                    "seed_quality": {"quality_flags": ["missing_required_families", "repo_examples_missing"]},
                }
            ],
            "crash_found": False,
            "failed": False,
            "run_error_kind": "",
        }
    )
    assert out["coverage_should_improve"] is True
    assert out["coverage_improve_mode"] == "in_place"
    assert "seed_quality_flags" in out["coverage_improve_reason"]
    assert out["coverage_target_api"] == "fmt::println"
    assert out["coverage_quality_oracle"] == "quality_degraded"
    assert isinstance(out.get("coverage_seed_feedback"), dict)
    assert isinstance(out.get("coverage_harness_feedback"), dict)


def test_route_after_re_build_routes_to_re_run_on_success():
    route = workflow_graph._route_after_re_build_state(
        {
            "failed": False,
            "crash_found": True,
            "re_build_done": True,
            "re_build_ok": True,
            "restart_to_plan": False,
        }
    )
    assert route == "re-run"


def test_route_after_re_build_routes_to_plan_on_failure():
    route = workflow_graph._route_after_re_build_state(
        {
            "failed": False,
            "crash_found": True,
            "re_build_done": True,
            "re_build_ok": False,
            "restart_to_plan": True,
            "restart_to_plan_count": 1,
        }
    )
    assert route == "plan"


def test_route_after_re_run_routes_to_crash_analysis_on_success():
    route = workflow_graph._route_after_re_run_state(
        {
            "failed": False,
            "crash_found": True,
            "crash_repro_done": True,
            "crash_repro_ok": True,
            "restart_to_plan": False,
        }
    )
    assert route == "crash-analysis"


def test_route_after_re_run_routes_to_plan_on_failure():
    route = workflow_graph._route_after_re_run_state(
        {
            "failed": False,
            "crash_found": True,
            "crash_repro_done": True,
            "crash_repro_ok": False,
            "restart_to_plan": True,
            "restart_to_plan_count": 1,
        }
    )
    assert route == "plan"


def test_route_after_crash_analysis_routes_to_plan_on_false_positive():
    route = workflow_graph._route_after_crash_analysis_state(
        {
            "failed": False,
            "restart_to_plan": True,
            "restart_to_plan_count": 1,
            "crash_analysis_verdict": "false_positive",
        }
    )
    assert route == "plan"


def test_route_after_crash_analysis_routes_to_stop_on_real_bug():
    route = workflow_graph._route_after_crash_analysis_state(
        {
            "failed": False,
            "restart_to_plan": False,
            "crash_analysis_verdict": "real_bug",
        }
    )
    assert route == "stop"


def test_apply_stage_stop_guard_always_stops_when_targeted():
    assert workflow_graph._apply_stage_stop_guard({"stop_after_step": "run"}, "run", "re-build") == "stop"
    assert workflow_graph._apply_stage_stop_guard({"stop_after_step": "re-build"}, "re-build", "plan") == "stop"
    assert workflow_graph._apply_stage_stop_guard({"stop_after_step": "crash-triage"}, "crash-triage", "fix-harness") == "stop"
    assert workflow_graph._apply_stage_stop_guard({"stop_after_step": "run"}, "crash-triage", "fix-harness") == "fix-harness"


def test_route_after_crash_triage_routes_by_label():
    assert workflow_graph._route_after_crash_triage_state({"crash_triage_label": "harness_bug"}) == "fix-harness"
    assert workflow_graph._route_after_crash_triage_state({"crash_triage_label": "upstream_bug"}) == "re-build"
    assert workflow_graph._route_after_crash_triage_state({"crash_triage_label": "inconclusive"}) == "plan"


def test_node_run_marks_finalize_timeout(tmp_path: Path, monkeypatch):
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
                final_execs_per_sec=1,
            )
        ],
    )
    monkeypatch.setenv("SHERPA_RUN_FINALIZE_TIMEOUT_SEC", "1")
    original_perf = workflow_graph.time.perf_counter
    base = original_perf()
    calls = {"n": 0}

    def _fake_perf() -> float:
        calls["n"] += 1
        return base + (calls["n"] * 2.0)

    monkeypatch.setattr(workflow_graph.time, "perf_counter", _fake_perf)
    out = workflow_graph._node_run({"generator": gen, "crash_fix_attempts": 0})
    # Keep this test resilient to internal finalize-loop call-count changes.
    if out.get("failed"):
        assert out["run_error_kind"] == "run_finalize_timeout"
        assert out["run_terminal_reason"] == "run_finalize_timeout"
    else:
        assert out["last_step"] == "run"
        assert out["run_error_kind"] in {"", "run_no_progress", "nonzero_exit_without_crash", "run_finalize_timeout"}


def test_calc_parallel_batch_budget_caps_unlimited_round_by_default(monkeypatch):
    monkeypatch.setenv("SHERPA_RUN_UNLIMITED_ROUND_BUDGET_SEC", "7200")
    rounds_left, round_budget, hard_timeout = workflow_graph._calc_parallel_batch_budget(
        pending_count=3,
        max_parallel=2,
        remaining_for_run=999999,
        configured_run_time_budget=0,
        total_budget_unlimited=True,
    )
    assert rounds_left == 2
    assert round_budget == 7200
    assert hard_timeout == 7320


def test_default_run_rss_limit_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("SHERPA_RUN_RSS_LIMIT_MB", "65536")
    monkeypatch.setenv("SHERPA_K8S_JOB_MEMORY_LIMIT", "64Gi")
    assert workflow_graph._default_run_rss_limit_mb() == 65536


def test_default_run_rss_limit_derived_from_k8s_memory_limit(monkeypatch):
    monkeypatch.delenv("SHERPA_RUN_RSS_LIMIT_MB", raising=False)
    monkeypatch.setenv("SHERPA_K8S_JOB_MEMORY_LIMIT", "64Gi")
    assert workflow_graph._default_run_rss_limit_mb() == int(64 * 1024 * 0.8)


def test_node_run_timeout_artifact_does_not_trigger_crash_packaging(tmp_path: Path):
    timeout_artifact = tmp_path / "fuzz" / "out" / "artifacts" / "timeout-deadbeef"
    timeout_artifact.parent.mkdir(parents=True, exist_ok=True)
    timeout_artifact.write_text("hang candidate", encoding="utf-8")

    gen = _FakeRunGenerator(
        tmp_path,
        run_results=[
            FuzzerRunResult(
                rc=70,
                new_artifacts=[timeout_artifact],
                crash_found=False,
                crash_evidence="timeout_artifact",
                first_artifact=str(timeout_artifact),
                log_tail="libFuzzer timeout",
                error="fuzzer produced timeout-like artifacts for demo_fuzz (count=1)",
                run_error_kind="run_timeout",
            )
        ],
    )

    out = workflow_graph._node_run({"generator": gen, "crash_fix_attempts": 0})
    assert out["last_step"] == "run"
    assert out["crash_found"] is False
    assert out["run_error_kind"] == "run_timeout"
    assert gen.analysis_calls == []
    route = workflow_graph._route_after_run_state(out)
    assert route == "plan"


def test_node_run_oom_artifact_is_resource_exhaustion_not_crash(tmp_path: Path):
    oom_artifact = tmp_path / "fuzz" / "out" / "artifacts" / "oom-deadbeef"
    oom_artifact.parent.mkdir(parents=True, exist_ok=True)
    oom_artifact.write_text("oom candidate", encoding="utf-8")

    gen = _FakeRunGenerator(
        tmp_path,
        run_results=[
            FuzzerRunResult(
                rc=71,
                new_artifacts=[oom_artifact],
                crash_found=False,
                crash_evidence="oom_artifact",
                first_artifact=str(oom_artifact),
                log_tail="ERROR: libFuzzer: out-of-memory",
                error="fuzzer produced oom-like artifacts for demo_fuzz",
                run_error_kind="run_resource_exhaustion",
            )
        ],
    )

    out = workflow_graph._node_run({"generator": gen, "crash_fix_attempts": 0})
    assert out["last_step"] == "run"
    assert out["crash_found"] is False
    assert out["run_error_kind"] == "run_resource_exhaustion"
    assert gen.analysis_calls == []
    route = workflow_graph._route_after_run_state(out)
    assert route == "plan"


def test_run_fuzz_workflow_stage_returns_recoverable_run_error(monkeypatch, tmp_path: Path):
    fake_out = {
        "repo_root": str(tmp_path),
        "last_step": "run",
        "message": "Fuzzing run failed.",
        "last_error": "fuzzer produced oom-like artifacts for demo_fuzz",
        "run_error_kind": "run_resource_exhaustion",
        "crash_found": False,
    }

    class _FakeCompiledWorkflow:
        def invoke(self, _state):
            return dict(fake_out)

    class _FakeWorkflow:
        def compile(self):
            return _FakeCompiledWorkflow()

    monkeypatch.setattr(workflow_graph, "build_fuzz_workflow", lambda: _FakeWorkflow())
    monkeypatch.setattr(workflow_graph, "_write_run_summary", lambda _out: None)

    result = workflow_graph.run_fuzz_workflow(
        workflow_graph.FuzzWorkflowInput(
            repo_url="https://github.com/example/repo.git",
            email=None,
            time_budget=0,
            run_time_budget=0,
            max_len=1000,
            docker_image=None,
            ai_key_path=tmp_path / ".env",
            resume_from_step="run",
            resume_repo_root=tmp_path,
            stop_after_step="run",
            coverage_loop_max_rounds=3,
            max_fix_rounds=3,
            same_error_max_retries=3,
        )
    )

    assert result["workflow_last_step"] == "run"
    assert result["workflow_recommended_next"] == "plan"


def test_node_run_stops_when_same_timeout_signature_repeats(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SHERPA_WORKFLOW_MAX_SAME_TIMEOUT_REPEATS", "1")
    timeout_artifact = tmp_path / "fuzz" / "out" / "artifacts" / "timeout-same"
    timeout_artifact.parent.mkdir(parents=True, exist_ok=True)
    timeout_artifact.write_text("hang candidate", encoding="utf-8")

    def _make_result() -> FuzzerRunResult:
        return FuzzerRunResult(
            rc=70,
            new_artifacts=[timeout_artifact],
            crash_found=False,
            crash_evidence="timeout_artifact",
            first_artifact=str(timeout_artifact),
            log_tail="libFuzzer timeout",
            error="fuzzer produced timeout-like artifacts for demo_fuzz (count=1)",
            run_error_kind="run_timeout",
        )

    first = workflow_graph._node_run(
        {"generator": _FakeRunGenerator(tmp_path, [_make_result()]), "crash_fix_attempts": 0}
    )
    assert first.get("failed") is not True
    sig = str(first.get("timeout_signature") or "")
    assert sig

    second = workflow_graph._node_run(
        {
            "generator": _FakeRunGenerator(tmp_path, [_make_result()]),
            "crash_fix_attempts": 0,
            "timeout_signature": sig,
            "same_timeout_repeats": int(first.get("same_timeout_repeats") or 0),
        }
    )
    assert second["failed"] is True
    assert second["run_error_kind"] == "run_timeout"
    assert second["same_timeout_repeats"] >= 1
    assert "same timeout/no-progress signature repeated" in second["last_error"]


def test_node_re_run_guesses_fuzzer_when_last_fuzzer_missing(tmp_path: Path, monkeypatch):
    workspace = tmp_path / ".repro_crash" / "workdir"
    out_dir = workspace / "fuzz" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    fuzzer_bin = out_dir / "fmt_format_string_fuzz"
    fuzzer_bin.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fuzzer_bin.chmod(0o755)
    artifact = out_dir / "artifacts" / "crash-deadbeef"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"boom")

    class _RunRes:
        returncode = 1
        stdout = "boom"
        stderr = "asan"

    monkeypatch.setattr(workflow_graph.subprocess, "run", lambda *a, **k: _RunRes())
    gen = SimpleNamespace(repo_root=tmp_path)
    out = workflow_graph._node_re_run(
        {
            "generator": gen,
            "last_fuzzer": "",
            "last_crash_artifact": str(artifact),
            "re_workspace_root": str(workspace),
        }
    )
    assert out["re_run_done"] is True
    assert out["re_run_ok"] is True
    assert out["crash_repro_ok"] is True


def test_node_re_run_recovers_context_from_re_build_report(tmp_path: Path, monkeypatch):
    workspace = tmp_path / ".repro_crash" / "workdir"
    out_dir = workspace / "fuzz" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    fuzzer_bin = out_dir / "fmt_format_string_fuzz"
    fuzzer_bin.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fuzzer_bin.chmod(0o755)
    artifact = out_dir / "artifacts" / "crash-deadbeef"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"boom")

    (tmp_path / "re_build_report.json").write_text(
        '{"fuzzer":"fmt_format_string_fuzz","artifact":"' + str(artifact) + '"}\n',
        encoding="utf-8",
    )

    class _RunRes:
        returncode = 1
        stdout = "boom"
        stderr = "asan"

    monkeypatch.setattr(workflow_graph.subprocess, "run", lambda *a, **k: _RunRes())
    gen = SimpleNamespace(repo_root=tmp_path)
    out = workflow_graph._node_re_run(
        {
            "generator": gen,
            "last_fuzzer": "",
            "last_crash_artifact": "",
            "re_workspace_root": str(workspace),
        }
    )
    assert out["re_run_done"] is True
    assert out["re_run_ok"] is True
    assert out["crash_repro_ok"] is True


def test_node_re_run_uses_generator_run_cmd_when_available(tmp_path: Path):
    workspace = tmp_path / ".repro_crash" / "workdir"
    out_dir = workspace / "fuzz" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    fuzzer_bin = out_dir / "fmt_format_string_fuzz"
    fuzzer_bin.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fuzzer_bin.chmod(0o755)
    artifact = out_dir / "artifacts" / "crash-deadbeef"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"boom")

    seen: dict[str, object] = {}

    def _run_cmd(cmd, *, cwd, env, timeout, idle_timeout):
        seen["cmd"] = [str(x) for x in cmd]
        seen["cwd"] = str(cwd)
        seen["timeout"] = int(timeout)
        seen["idle_timeout"] = int(idle_timeout)
        seen["env_has_marker"] = str(env.get("REPRO_MARKER") or "") == "1"
        return 1, "boom", "asan"

    def _compose_vcpkg_runtime_env(env, *, repo_root):
        out_env = dict(env)
        out_env["REPRO_MARKER"] = "1"
        return out_env

    gen = SimpleNamespace(
        repo_root=tmp_path,
        _run_cmd=_run_cmd,
        _compose_vcpkg_runtime_env=_compose_vcpkg_runtime_env,
    )
    out = workflow_graph._node_re_run(
        {
            "generator": gen,
            "last_fuzzer": "fmt_format_string_fuzz",
            "last_crash_artifact": str(artifact),
            "re_workspace_root": str(workspace),
        }
    )
    assert out["re_run_done"] is True
    assert out["re_run_ok"] is True
    assert out["crash_repro_ok"] is True
    assert seen["cwd"] == str(workspace)
    assert seen["idle_timeout"] == 0
    assert seen["env_has_marker"] is True
    assert "-runs=1" in (seen["cmd"] or [])


def test_node_re_run_recovers_artifact_from_run_summary(tmp_path: Path, monkeypatch):
    workspace = tmp_path / ".repro_crash" / "workdir"
    out_dir = workspace / "fuzz" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    fuzzer_bin = out_dir / "fmt_format_string_fuzz"
    fuzzer_bin.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fuzzer_bin.chmod(0o755)
    artifact = (tmp_path / "fuzz" / "out" / "artifacts" / "crash-deadbeef")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"boom")

    (tmp_path / "run_summary.json").write_text(
        '{"last_crash_artifact":"' + str(artifact) + '"}\n',
        encoding="utf-8",
    )

    class _RunRes:
        returncode = 1
        stdout = "boom"
        stderr = "asan"

    monkeypatch.setattr(workflow_graph.subprocess, "run", lambda *a, **k: _RunRes())
    gen = SimpleNamespace(repo_root=tmp_path)
    out = workflow_graph._node_re_run(
        {
            "generator": gen,
            "last_fuzzer": "fmt_format_string_fuzz",
            "last_crash_artifact": "",
            "re_workspace_root": str(workspace),
        }
    )
    assert out["re_run_done"] is True
    assert out["re_run_ok"] is True
    assert out["crash_repro_ok"] is True


def test_node_re_run_recovers_context_from_repro_context(tmp_path: Path, monkeypatch):
    workspace = tmp_path / ".repro_crash" / "workdir"
    out_dir = workspace / "fuzz" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    fuzzer_bin = out_dir / "fmt_format_string_fuzz"
    fuzzer_bin.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fuzzer_bin.chmod(0o755)
    artifact = (tmp_path / "fuzz" / "out" / "artifacts" / "crash-deadbeef")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"boom")

    (tmp_path / "repro_context.json").write_text(
        (
            "{"
            f"\"last_fuzzer\":\"fmt_format_string_fuzz\","
            f"\"last_crash_artifact\":\"{artifact}\","
            f"\"re_workspace_root\":\"{workspace}\""
            "}\n"
        ),
        encoding="utf-8",
    )

    class _RunRes:
        returncode = 1
        stdout = "boom"
        stderr = "asan"

    monkeypatch.setattr(workflow_graph.subprocess, "run", lambda *a, **k: _RunRes())
    gen = SimpleNamespace(repo_root=tmp_path)
    out = workflow_graph._node_re_run(
        {
            "generator": gen,
            "last_fuzzer": "",
            "last_crash_artifact": "",
            "re_workspace_root": "",
        }
    )
    assert out["re_run_done"] is True
    assert out["re_run_ok"] is True
    assert out["crash_repro_ok"] is True


def test_node_re_run_rebuilds_workspace_when_missing(tmp_path: Path, monkeypatch):
    repo_work = tmp_path / "repo-clone"
    repo_work.mkdir(parents=True, exist_ok=True)
    source_fuzz = tmp_path / "fuzz"
    source_fuzz.mkdir(parents=True, exist_ok=True)
    (source_fuzz / "build.py").write_text("print('ok')\n", encoding="utf-8")
    artifact = source_fuzz / "out" / "artifacts" / "crash-deadbeef"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"boom")

    class _RunRes:
        def __init__(self, rc: int, stdout: str = "", stderr: str = ""):
            self.returncode = rc
            self.stdout = stdout
            self.stderr = stderr

    def _clone_repo(spec):
        return repo_work

    def _python_runner():
        return "python3"

    def _fake_subprocess_run(cmd, cwd=None, capture_output=None, text=None, timeout=None, env=None):
        cmd_list = [str(x) for x in cmd]
        if cmd_list[:2] == ["python3", "build.py"]:
            out_dir = Path(cwd) / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            fuzzer = out_dir / "fmt_format_string_fuzz"
            fuzzer.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            fuzzer.chmod(0o755)
            return _RunRes(0, "build ok", "")
        if "-runs=1" in cmd_list:
            return _RunRes(1, "boom", "asan")
        raise AssertionError(f"unexpected cmd: {cmd_list}")

    monkeypatch.setattr(workflow_graph.subprocess, "run", _fake_subprocess_run)
    gen = SimpleNamespace(repo_root=tmp_path, _clone_repo=_clone_repo, _python_runner=_python_runner)
    out = workflow_graph._node_re_run(
        {
            "generator": gen,
            "repo_url": "https://github.com/fmtlib/fmt.git",
            "last_fuzzer": "",
            "last_crash_artifact": str(artifact),
            "re_workspace_root": str(tmp_path / ".repro_crash" / "missing-workdir"),
        }
    )
    assert out["re_run_done"] is True
    assert out["re_run_ok"] is True
    assert out["crash_repro_ok"] is True
