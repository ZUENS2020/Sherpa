from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "harness_generator" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import fuzz_unharnessed_repo as fur
from fuzz_unharnessed_repo import NonOssFuzzHarnessGenerator


def _fake_generator(repo_root: Path) -> NonOssFuzzHarnessGenerator:
    gen = NonOssFuzzHarnessGenerator.__new__(NonOssFuzzHarnessGenerator)
    gen.repo_root = repo_root
    gen.docker_image = None
    gen.last_seed_profile_by_fuzzer = {}
    gen.last_seed_bootstrap_by_fuzzer = {}
    gen.last_selected_target_by_fuzzer = {}
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
    (fuzz_dir / "system_packages.txt").write_text("zlib\n", encoding="utf-8")

    log_path = tmp_path / "vcpkg.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    vcpkg_dir = tmp_path / "vcpkg"
    vcpkg_dir.mkdir(parents=True, exist_ok=True)
    vcpkg_script = vcpkg_dir / "vcpkg"
    vcpkg_script.write_text(
        "#!/bin/sh\n"
        f"echo \"$@\" >> {log_path}\n"
        "if [ \"$1\" = \"list\" ]; then exit 1; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    vcpkg_script.chmod(0o755)

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
    assert "list zlib:" in log_text
    assert "install --triplet" in log_text
    assert "zlib" in log_text


def test_run_cmd_preflight_rewrites_dangerous_repo_build_dir(tmp_path: Path):
    gen = _fake_generator(tmp_path)
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text(
        "from pathlib import Path\n"
        "import shutil\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
        "BUILD_DIR = REPO_ROOT / \"build\"\n"
        "if BUILD_DIR.exists():\n"
        "    shutil.rmtree(BUILD_DIR)\n"
        "print('preflight-ok')\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["SHERPA_AUTO_INSTALL_SYSTEM_DEPS"] = "0"

    rc, out, err = gen._run_cmd(
        [sys.executable, "build.py"],
        cwd=fuzz_dir,
        env=env,
        timeout=10,
        idle_timeout=0,
    )

    assert rc == 0, err
    assert "preflight-ok" in out
    txt = build_py.read_text(encoding="utf-8")
    assert 'BUILD_DIR = REPO_ROOT / "fuzz" / "build-work"' in txt


def test_run_cmd_preflight_keeps_safe_build_dir_unchanged(tmp_path: Path):
    gen = _fake_generator(tmp_path)
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    build_py = fuzz_dir / "build.py"
    build_py.write_text(
        "from pathlib import Path\n"
        "import shutil\n"
        "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
        "BUILD_DIR = REPO_ROOT / \"fuzz\" / \"build-work\"\n"
        "if BUILD_DIR.exists():\n"
        "    shutil.rmtree(BUILD_DIR)\n"
        "print('safe-ok')\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["SHERPA_AUTO_INSTALL_SYSTEM_DEPS"] = "0"

    rc, out, err = gen._run_cmd(
        [sys.executable, "build.py"],
        cwd=fuzz_dir,
        env=env,
        timeout=10,
        idle_timeout=0,
    )

    assert rc == 0, err
    assert "safe-ok" in out
    txt = build_py.read_text(encoding="utf-8")
    assert txt.count('BUILD_DIR = REPO_ROOT / "fuzz" / "build-work"') == 1


def test_pass_generate_seeds_uses_declared_target_type_guidance(tmp_path: Path):
    gen = _fake_generator(tmp_path)
    gen.fuzz_dir = tmp_path / "fuzz"
    gen.fuzz_corpus_dir = gen.fuzz_dir / "corpus"
    gen.fuzz_dir.mkdir(parents=True, exist_ok=True)
    gen.fuzz_corpus_dir.mkdir(parents=True, exist_ok=True)
    (gen.fuzz_dir / "targets.json").write_text(
        '[{"name":"yaml_parser_parse","api":"yaml_parser_parse","lang":"c-cpp","target_type":"parser","seed_profile":"parser-structure"}]\n',
        encoding="utf-8",
    )
    harness = gen.fuzz_dir / "yaml_parser_parse_fuzz.cc"
    harness.write_text("int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long) { return 0; }\n", encoding="utf-8")

    captured: dict[str, str] = {}

    class _Patcher:
        def run_codex_command(self, instructions: str, additional_context: str = "", **_kwargs):
            captured["instructions"] = instructions
            captured["context"] = additional_context
            return "seed-ok"

    gen.patcher = _Patcher()

    gen._pass_generate_seeds("yaml_parser_parse_fuzz")

    assert "Target type for `yaml_parser_parse_fuzz` is `parser`" in captured["instructions"]
    assert "seed_profile is `parser-structure`" in captured["instructions"]
    assert "anchors and aliases" in captured["instructions"]
    assert "Current corpus summary:" in captured["instructions"]
    assert "seed_exploration_yaml_parser_parse_fuzz.json" in captured["instructions"]
    assert "seed_check_yaml_parser_parse_fuzz.json" in captured["instructions"]
    assert "Before writing new seeds, inspect repository files relevant to target inputs" in captured["instructions"]
    assert "fuzz/PLAN.md" in captured["instructions"]
    assert "If required families are still missing, or if the corpus is still much smaller than the target size, add more seeds before finishing" in captured["instructions"]
    assert "Aim for at least" in captured["instructions"]
    assert "total seed files" in captured["instructions"]
    assert "Do not stop after creating only one tiny seed per family" in captured["instructions"]
    assert "seed_check_yaml_parser_parse_fuzz.json" in captured["instructions"]
    assert "Before writing new seeds, inspect repository files relevant to target inputs" in captured["instructions"]
    assert "fuzz/PLAN.md" in captured["instructions"]
    assert "If required families are still missing, or if the corpus is still much smaller than the target size, add more seeds before finishing" in captured["instructions"]
    assert "Aim for at least" in captured["instructions"]
    assert "total seed files" in captured["instructions"]
    assert "Do not stop after creating only one tiny seed per family" in captured["instructions"]


def test_pass_generate_seeds_adds_argument_id_boundary_guidance(tmp_path: Path):
    gen = _fake_generator(tmp_path)
    gen.fuzz_dir = tmp_path / "fuzz"
    gen.fuzz_corpus_dir = gen.fuzz_dir / "corpus"
    gen.fuzz_dir.mkdir(parents=True, exist_ok=True)
    gen.fuzz_corpus_dir.mkdir(parents=True, exist_ok=True)
    (gen.fuzz_dir / "targets.json").write_text(
        '[{"name":"parse_arg_id","api":"parse_arg_id","lang":"c-cpp","target_type":"parser","seed_profile":"parser-numeric"}]\n',
        encoding="utf-8",
    )
    harness = gen.fuzz_dir / "parse_arg_id_fuzz.cc"
    harness.write_text(
        "int parse_arg_id(const char*);\n"
        "int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long) { return 0; }\n",
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    class _Patcher:
        def run_codex_command(self, instructions: str, additional_context: str = "", **_kwargs):
            captured["instructions"] = instructions
            captured["context"] = additional_context
            return "seed-ok"

    gen.patcher = _Patcher()

    gen._pass_generate_seeds("parse_arg_id_fuzzer")

    assert "seed_profile is `parser-numeric`" in captured["instructions"]
    assert "leading zeros" in captured["instructions"]
    assert "separator-boundary tokens" in captured["instructions"]
    assert "Coverage-oriented gap hints" in captured["instructions"]


def test_pass_generate_seeds_prefers_selected_seed_profile_over_observed(tmp_path: Path):
    gen = _fake_generator(tmp_path)
    gen.fuzz_dir = tmp_path / "fuzz"
    gen.fuzz_corpus_dir = gen.fuzz_dir / "corpus"
    gen.fuzz_dir.mkdir(parents=True, exist_ok=True)
    gen.fuzz_corpus_dir.mkdir(parents=True, exist_ok=True)
    (gen.fuzz_dir / "selected_targets.json").write_text(
        '[{"target_name":"demo_parse","api":"demo_parse","target_type":"parser","seed_profile":"parser-structure","seed_families_required":["document_markers"],"seed_families_optional":[]}]',
        encoding="utf-8",
    )
    (gen.fuzz_dir / "observed_target.json").write_text(
        '{"selected_target_api":"demo_parse","observed_target_api":"fmt::println","target_type":"parser","seed_profile":"parser-format","observed_harness":"demo_parse_fuzz.cc"}',
        encoding="utf-8",
    )
    harness = gen.fuzz_dir / "demo_parse_fuzz.cc"
    harness.write_text(
        'int LLVMFuzzerTestOneInput(const unsigned char* data, unsigned long size) { fmt::println("{}", (const char*)data); return 0; }\n',
        encoding="utf-8",
    )

    captured: dict[str, str] = {}

    class _Patcher:
        def run_codex_command(self, instructions: str, additional_context: str = "", **_kwargs):
            captured["instructions"] = instructions
            return "seed-ok"

    gen.patcher = _Patcher()
    gen._pass_generate_seeds("demo_parse_fuzz")

    assert "seed_profile is `parser-structure`" in captured["instructions"]
    meta = dict(gen.last_seed_bootstrap_by_fuzzer.get("demo_parse_fuzz") or {})
    assert meta.get("seed_profile") == "parser-structure"
    assert meta.get("seed_profile_source") == "selected_targets"


def test_run_fuzzer_stops_on_coverage_plateau(tmp_path: Path, monkeypatch):
    gen = _fake_generator(tmp_path)
    gen.time_budget = 900
    gen.max_len = 1024
    gen.rss_limit_mb = 32768
    gen.fuzz_out_dir = tmp_path / "fuzz" / "out"
    gen.fuzz_corpus_dir = tmp_path / "fuzz" / "corpus"
    gen.fuzz_out_dir.mkdir(parents=True, exist_ok=True)
    gen.fuzz_corpus_dir.mkdir(parents=True, exist_ok=True)
    bin_path = gen.fuzz_out_dir / "demo_fuzz"
    bin_path.write_text("", encoding="utf-8")
    os.chmod(bin_path, 0o755)

    timeline = iter([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    monkeypatch.setattr(fur.time, "monotonic", lambda: next(timeline))

    seen_cmd = {}

    def _fake_run_cmd(_cmd, **kwargs):
        seen_cmd["cmd"] = list(_cmd)
        cb = kwargs.get("line_callback")
        lines = [
            "#1 NEW cov: 6 ft: 10 corp: 3/24b lim: 24 exec/s: 100 rss: 10Mb\n",
            "#262144 pulse  cov: 6 ft: 10 corp: 3/24b lim: 24 exec/s: 100 rss: 10Mb\n",
            "#524288 pulse  cov: 6 ft: 10 corp: 3/24b lim: 24 exec/s: 100 rss: 10Mb\n",
            "#1048576 pulse  cov: 6 ft: 10 corp: 3/24b lim: 24 exec/s: 100 rss: 10Mb\n",
        ]
        for line in lines:
            if cb is not None:
                cb("stdout", line)
        return 143, "".join(lines), "\n[callback-stop] coverage_plateau (idle_no_growth=30s pulse_hits=3)"

    gen._run_cmd = _fake_run_cmd  # type: ignore[method-assign]
    old_idle = os.environ.get("SHERPA_RUN_PLATEAU_IDLE_GROWTH_SEC")
    old_pulses = os.environ.get("SHERPA_RUN_PLATEAU_PULSES")
    os.environ["SHERPA_RUN_PLATEAU_IDLE_GROWTH_SEC"] = "2"
    os.environ["SHERPA_RUN_PLATEAU_PULSES"] = "3"
    try:
        result = gen._run_fuzzer(bin_path)
    finally:
        if old_idle is None:
            os.environ.pop("SHERPA_RUN_PLATEAU_IDLE_GROWTH_SEC", None)
        else:
            os.environ["SHERPA_RUN_PLATEAU_IDLE_GROWTH_SEC"] = old_idle
        if old_pulses is None:
            os.environ.pop("SHERPA_RUN_PLATEAU_PULSES", None)
        else:
            os.environ["SHERPA_RUN_PLATEAU_PULSES"] = old_pulses

    assert result.rc == 0
    assert result.crash_found is False
    assert result.run_error_kind == ""
    assert result.plateau_detected is True
    assert result.terminal_reason == "coverage_plateau"
    assert "-rss_limit_mb=32768" in seen_cmd["cmd"]


def test_pass_generate_seeds_bootstraps_repo_examples_and_records_counts(tmp_path: Path, monkeypatch):
    gen = _fake_generator(tmp_path)
    gen.fuzz_dir = tmp_path / "fuzz"
    gen.fuzz_corpus_dir = gen.fuzz_dir / "corpus"
    gen.fuzz_dir.mkdir(parents=True, exist_ok=True)
    gen.fuzz_corpus_dir.mkdir(parents=True, exist_ok=True)
    (gen.fuzz_dir / "targets.json").write_text(
        '[{"name":"yaml_parser_parse","api":"yaml_parser_parse","lang":"c-cpp","target_type":"parser","seed_profile":"parser-structure"}]\n',
        encoding="utf-8",
    )
    harness = gen.fuzz_dir / "yaml_parser_parse_fuzz.cc"
    harness.write_text("int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long) { return 0; }\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "sample.yaml").write_text("---\na: 1\n", encoding="utf-8")

    class _Patcher:
        def run_codex_command(self, _instructions: str, **_kwargs):
            corpus_dir = gen.fuzz_corpus_dir / "yaml_parser_parse_fuzz"
            (corpus_dir / "ai_extra.yaml").write_text("...\n", encoding="utf-8")
            (gen.fuzz_dir / "seed_exploration_yaml_parser_parse_fuzz.json").write_text(
                '{"chosen_target_api":"yaml_parser_parse","observed_target_api":"","seed_profile":"parser-structure","required_families":["document_markers"],"missing_families":[],"repo_paths_reviewed":["tests/sample.yaml"],"sample_inputs_found":["tests/sample.yaml"],"summary":"reviewed yaml sample and existing corpus"}\n',
                encoding="utf-8",
            )
            (gen.fuzz_dir / "seed_check_yaml_parser_parse_fuzz.json").write_text(
                '{"seed_profile":"parser-structure","required_families":["document_markers"],"covered_families":["document_markers"],"missing_families":[],"family_counts":{"document_markers":2},"corpus_files":2,"target_corpus_files":8,"per_family_target":2,"planned_additions":["more valid/minimal docs"],"summary":"required families covered but corpus still thin"}\n',
                encoding="utf-8",
            )
            (gen.fuzz_dir / "seed_check_yaml_parser_parse_fuzz.json").write_text(
                '{"seed_profile":"parser-structure","required_families":["document_markers"],"covered_families":["document_markers"],"missing_families":[],"family_counts":{"document_markers":2},"corpus_files":2,"target_corpus_files":8,"per_family_target":2,"planned_additions":["more valid/minimal docs"],"summary":"required families covered but corpus still thin"}\n',
                encoding="utf-8",
            )
            return "seed-ok"

    orig_which = fur.which
    monkeypatch.setattr(fur, "which", lambda cmd: None if cmd == "radamsa" else orig_which(cmd))
    gen.patcher = _Patcher()

    gen._pass_generate_seeds("yaml_parser_parse_fuzz")

    corpus_dir = gen.fuzz_corpus_dir / "yaml_parser_parse_fuzz"
    assert (corpus_dir / "repo_01.yaml").is_file()
    assert (corpus_dir / "ai_extra.yaml").is_file()
    meta = gen.last_seed_bootstrap_by_fuzzer["yaml_parser_parse_fuzz"]
    assert meta["counts"]["repo_examples"] == 1
    assert meta["counts"]["ai"] >= 1
    assert "repo_examples" in meta["sources"]
    assert meta["repo_examples_filtered"] is True
    assert meta["repo_examples_accepted_count"] == 1
    assert meta["repo_examples_rejected_count"] >= 0
    assert meta["seed_exploration_path"] == "fuzz/seed_exploration_yaml_parser_parse_fuzz.json"
    assert meta["seed_check_path"] == "fuzz/seed_check_yaml_parser_parse_fuzz.json"
    assert meta["seed_check_path"] == "fuzz/seed_check_yaml_parser_parse_fuzz.json"


def test_collect_repo_seed_examples_filters_source_files_for_generic_targets(tmp_path: Path):
    gen = _fake_generator(tmp_path)
    corpus_dir = tmp_path / "fuzz" / "corpus" / "generic_fuzz"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "sample.dat").write_bytes(b"\x00\x01sample")
    (tests_dir / "helper.c").write_text("int main(void){return 0;}\n", encoding="utf-8")
    (tests_dir / "page.html").write_text("<html></html>\n", encoding="utf-8")

    selected, meta = gen._collect_repo_seed_examples("generic", "generic_fuzz", corpus_dir)

    assert [p.name for p in selected] == ["repo_01.dat"]
    assert meta["accepted_count"] == 1
    assert meta["rejected_count"] >= 2
    assert meta["filtered"] is True


def test_collect_repo_seed_examples_ignores_repo_source_for_parser_numeric(tmp_path: Path):
    gen = _fake_generator(tmp_path)
    corpus_dir = tmp_path / "fuzz" / "corpus" / "parse_arg_id_fuzzer"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "arg_tokens.txt").write_text("0\n1\n42\n", encoding="utf-8")
    (tests_dir / "arg_helper.c").write_text("int parse_arg_id(void);\n", encoding="utf-8")

    selected, meta = gen._collect_repo_seed_examples("parser-numeric", "parse_arg_id_fuzzer", corpus_dir)

    assert [p.name for p in selected] == ["repo_01.txt"]
    assert meta["accepted_count"] == 1
    assert meta["rejected_count"] >= 1


def test_pass_generate_seeds_radamsa_missing_is_non_fatal(tmp_path: Path, monkeypatch):
    gen = _fake_generator(tmp_path)
    gen.fuzz_dir = tmp_path / "fuzz"
    gen.fuzz_corpus_dir = gen.fuzz_dir / "corpus"
    gen.fuzz_dir.mkdir(parents=True, exist_ok=True)
    gen.fuzz_corpus_dir.mkdir(parents=True, exist_ok=True)
    (gen.fuzz_dir / "targets.json").write_text(
        '[{"name":"parse_arg_id","api":"parse_arg_id","lang":"c-cpp","target_type":"parser","seed_profile":"parser-numeric"}]\n',
        encoding="utf-8",
    )
    harness = gen.fuzz_dir / "parse_arg_id_fuzz.cc"
    harness.write_text("int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long) { return 0; }\n", encoding="utf-8")

    class _Patcher:
        def run_codex_command(self, _instructions: str, **_kwargs):
            corpus_dir = gen.fuzz_corpus_dir / "parse_arg_id_fuzzer"
            (corpus_dir / "seed_num").write_text("42", encoding="utf-8")
            return "seed-ok"

    orig_which = fur.which
    monkeypatch.setattr(fur, "which", lambda cmd: None if cmd == "radamsa" else orig_which(cmd))
    gen.patcher = _Patcher()

    gen._pass_generate_seeds("parse_arg_id_fuzzer")

    meta = gen.last_seed_bootstrap_by_fuzzer["parse_arg_id_fuzzer"]
    assert meta["counts"]["radamsa"] == 0
    assert meta["seed_profile"] == "parser-numeric"
