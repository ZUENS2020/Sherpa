from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "harness_generator" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fuzz_unharnessed_repo import (
    GIT_CLONE_RETRIES,
    NonOssFuzzHarnessGenerator,
    RepoSpec,
    _classify_seed_family,
    _host_git_proxy_env,
    _seed_families_for_target,
    _seed_quality_from_run,
)


def _make_generator(repo_root: Path) -> NonOssFuzzHarnessGenerator:
    gen = NonOssFuzzHarnessGenerator.__new__(NonOssFuzzHarnessGenerator)
    gen.repo_root = repo_root
    gen.fuzz_dir = repo_root / "fuzz"
    gen.fuzz_corpus_dir = gen.fuzz_dir / "corpus"
    return gen


def test_resolve_seed_target_metadata_prefers_selected_targets(tmp_path: Path):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "selected_targets.json").write_text(
        '[{"target_name":"yaml_parser_parse","api":"yaml_parser_parse","target_type":"parser","seed_profile":"parser-structure","seed_families_required":["document_markers"],"seed_families_optional":[]}]',
        encoding="utf-8",
    )
    gen = _make_generator(tmp_path)
    target_type, seed_profile = gen._resolve_seed_target_metadata(
        "yaml_parser_fuzz",
        "extern \"C\" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) { yaml_parser_parse(0, 0); return 0; }",
    )
    assert target_type == "parser"
    assert seed_profile == "parser-structure"


def test_collect_repo_seed_examples_accepts_yaml_samples_for_parser_token(tmp_path: Path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "sample.yaml").write_text("---\nkey: value\n", encoding="utf-8")
    (tests_dir / "anchor.yaml").write_text("&a foo\n*b\n", encoding="utf-8")
    corpus_dir = tmp_path / "fuzz" / "corpus" / "yaml_parser_fuzz"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    gen = _make_generator(tmp_path)
    selected, meta = gen._collect_repo_seed_examples(
        "parser-token",
        "yaml_parser_fuzz",
        corpus_dir,
        required_families=["document_markers", "anchors_aliases"],
    )
    assert len(selected) >= 1
    assert meta["accepted_count"] >= 1


def test_collect_repo_seed_examples_respects_seed_max_file_bytes(tmp_path: Path, monkeypatch):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "small.yaml").write_text("---\nk: v\n", encoding="utf-8")
    (tests_dir / "large.yaml").write_text("a" * 6000, encoding="utf-8")
    corpus_dir = tmp_path / "fuzz" / "corpus" / "yaml_parser_fuzz"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SHERPA_SEED_MAX_FILE_BYTES", "4096")
    gen = _make_generator(tmp_path)

    selected, meta = gen._collect_repo_seed_examples(
        "parser-token",
        "yaml_parser_fuzz",
        corpus_dir,
        required_families=["document_markers"],
    )

    assert len(selected) == 1
    assert selected[0].name.endswith(".yaml")
    assert meta["accepted_count"] == 1


def test_collect_repo_seed_examples_bootstraps_archive_samples_when_repo_has_none(tmp_path: Path):
    corpus_dir = tmp_path / "fuzz" / "corpus" / "archive_fuzz"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    gen = _make_generator(tmp_path)

    selected, meta = gen._collect_repo_seed_examples(
        "archive-container",
        "archive_unpack_fuzz",
        corpus_dir,
        required_families=["flow_structures"],
    )

    assert len(selected) >= 3
    suffixes = {p.suffix for p in selected}
    assert ".zip" in suffixes or ".tar" in suffixes
    assert any(ext in suffixes for ext in {".gz", ".bz2", ".xz"})
    assert meta["accepted_count"] == len(selected)


def test_resolve_seed_target_metadata_prefers_observed_target(tmp_path: Path):
    fuzz_dir = tmp_path / "fuzz"
    fuzz_dir.mkdir(parents=True, exist_ok=True)
    (fuzz_dir / "selected_targets.json").write_text(
        '[{"target_name":"parse_replacement_field_then_tail","api":"parse_replacement_field_then_tail","target_type":"generic","seed_profile":"generic","seed_families_required":[],"seed_families_optional":[]}]',
        encoding="utf-8",
    )
    (fuzz_dir / "observed_target.json").write_text(
        '{'
        '"selected_target_name":"parse_replacement_field_then_tail",'
        '"selected_target_api":"parse_replacement_field_then_tail",'
        '"observed_target_api":"fmt::println",'
        '"observed_harness":"println_fuzz.cc",'
        '"drifted":true,'
        '"drift_reason":"runtime wrapper",'
        '"relation":"wrapper",'
        '"runtime_viability":"high"'
        '}',
        encoding="utf-8",
    )
    gen = _make_generator(tmp_path)
    target_type, seed_profile = gen._resolve_seed_target_metadata(
        "println_fuzz",
        'extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) { fmt::println("{}", (const char*)data); return 0; }',
    )

    assert target_type == "parser"
    assert seed_profile == "parser-format"


def test_seed_quality_flags_detect_low_retention_and_missing_families():
    log = "\n".join(
        [
            "#192 INITED cov: 5 ft: 19 corp: 9/160b exec/s: 0 rss: 99Mb",
            "#131072 pulse cov: 5 ft: 19 corp: 9/160b lim: 1000 exec/s: 65536 rss: 162Mb",
            "#262144 pulse cov: 5 ft: 19 corp: 9/160b lim: 1000 exec/s: 52428 rss: 163Mb",
        ]
    )
    quality = _seed_quality_from_run(
        log=log,
        initial_corpus_files=192,
        initial_corpus_bytes=44434,
        final_stats={
            "cov": 5,
            "ft": 19,
            "corpus_files": 9,
            "corpus_size_bytes": 160,
        },
        required_families=["flow_structures", "anchors_aliases"],
        covered_families=["anchors_aliases"],
        repo_examples_count=0,
        plateau_idle_seconds=180,
    )
    flags = set(quality["quality_flags"])
    assert "low_retention" in flags
    assert "missing_required_families" in flags
    assert "repo_examples_missing" in flags


def test_host_git_proxy_env_prefers_runtime_proxy_env(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.10:6789")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,.svc")
    monkeypatch.setenv("SHERPA_DOCKER_HTTP_PROXY", "http://host.docker.internal:7897")

    env = _host_git_proxy_env()

    assert env["HTTP_PROXY"] == "http://10.0.0.10:6789"
    assert env["http_proxy"] == "http://10.0.0.10:6789"
    assert env["NO_PROXY"] == "127.0.0.1,localhost,.svc"
    assert env["no_proxy"] == "127.0.0.1,localhost,.svc"
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_host_git_clone_retries_before_failing_over(monkeypatch, tmp_path: Path):
    attempts = []
    sleeps = []
    clone_url = "https://github.com/fmtlib/fmt.git"

    def fake_run(cmd, timeout=None, env=None):
        attempts.append((tuple(cmd), timeout, dict(env or {})))
        if len(attempts) == 1:
            return 128, "", "GnuTLS, handshake failed", False
        dest = Path(cmd[-1])
        dest.mkdir(parents=True, exist_ok=True)
        return 0, "", "", False

    monkeypatch.setattr("fuzz_unharnessed_repo._run_cmd_capture", fake_run)
    monkeypatch.setattr("fuzz_unharnessed_repo._candidate_clone_urls", lambda url: [clone_url])
    monkeypatch.setattr("fuzz_unharnessed_repo._host_git_proxy_override_args", lambda: [])
    monkeypatch.setattr("fuzz_unharnessed_repo._host_git_proxy_env", lambda: {"HTTP_PROXY": "http://192.168.1.79:6789"})
    monkeypatch.setattr("fuzz_unharnessed_repo.time.sleep", lambda seconds: sleeps.append(seconds))

    gen = NonOssFuzzHarnessGenerator.__new__(NonOssFuzzHarnessGenerator)
    gen.docker_image = None
    dest = tmp_path / "repo"

    repo_root = NonOssFuzzHarnessGenerator._clone_repo(gen, RepoSpec(url=clone_url, workdir=dest))

    assert repo_root == dest
    assert dest.exists()
    clone_attempts = [item for item in attempts if len(item[0]) >= 2 and item[0][1] == "clone"]

    assert len(clone_attempts) == 2
    assert clone_attempts[0][1] == clone_attempts[1][1]
    assert clone_attempts[0][2]["HTTP_PROXY"] == "http://192.168.1.79:6789"
    assert sleeps == [0.5]
    assert GIT_CLONE_RETRIES >= 2


def test_fmt_seed_families_replace_generic_parser_format():
    required, optional = _seed_families_for_target(
        "parser-format",
        "fmt::println",
        "fmt::format_to",
        "replacement field",
    )
    assert "replacement_fields" in required
    assert "width_precision" in required
    assert "malformed_replacement_fields" in required
    assert optional == []


def test_filter_seed_corpus_rejects_noisy_fmt_binary_variants(tmp_path: Path):
    corpus_dir = tmp_path / "fuzz" / "corpus" / "println_fuzzer"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "repo_01.txt").write_text("{}\n", encoding="utf-8")
    (corpus_dir / "seed_a.txt").write_text("{:08x}\n", encoding="utf-8")
    (corpus_dir / "seed_b.txt").write_bytes(b"\x00\xff\x10\x00\xfe")
    gen = _make_generator(tmp_path)

    filtered = gen._filter_seed_corpus(
        corpus_dir,
        seed_profile="parser-format",
        required_families=["replacement_fields", "format_specifiers"],
        target_markers=["fmt::println", "fmt::format_to"],
    )

    assert filtered["seed_noise_rejected_count"] >= 1
    kept_files = {p.name for p in corpus_dir.iterdir() if p.is_file()}
    assert "seed_b.txt" not in kept_files
    covered = _classify_seed_family(corpus_dir / "seed_a.txt")
    assert "format_specifiers" in covered


def test_filter_seed_corpus_rejects_oversized_and_radamsa_big_files(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "fuzz" / "corpus" / "demo_fuzzer"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "repo_01.txt").write_text("{}\n", encoding="utf-8")
    (corpus_dir / "radamsa_01.bin").write_bytes(b"A" * 7000)
    (corpus_dir / "seed_big.bin").write_bytes(b"B" * 9000)
    monkeypatch.setenv("SHERPA_SEED_MAX_FILE_BYTES", "8192")
    monkeypatch.setenv("SHERPA_RADAMSA_MAX_FILE_BYTES", "4096")
    gen = _make_generator(tmp_path)

    filtered = gen._filter_seed_corpus(
        corpus_dir,
        seed_profile="generic",
        required_families=[],
        target_markers=["demo"],
    )

    kept_files = {p.name for p in corpus_dir.iterdir() if p.is_file()}
    assert "repo_01.txt" in kept_files
    assert "radamsa_01.bin" not in kept_files
    assert "seed_big.bin" not in kept_files
    assert int(filtered["seed_oversized_rejected_count"]) >= 2


def test_filter_seed_corpus_prunes_total_bytes_preferring_radamsa(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "fuzz" / "corpus" / "demo_fuzzer"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "repo_01.txt").write_bytes(b"R" * 7000)
    (corpus_dir / "seed_01.txt").write_bytes(b"S" * 7000)
    (corpus_dir / "radamsa_01.txt").write_bytes(b"X" * 7000)
    monkeypatch.setenv("SHERPA_SEED_MAX_FILE_BYTES", "8192")
    monkeypatch.setenv("SHERPA_RADAMSA_MAX_FILE_BYTES", "8192")
    monkeypatch.setenv("SHERPA_SEED_MAX_TOTAL_BYTES", "16384")
    gen = _make_generator(tmp_path)

    filtered = gen._filter_seed_corpus(
        corpus_dir,
        seed_profile="generic",
        required_families=[],
        target_markers=["demo"],
    )

    kept_files = {p.name for p in corpus_dir.iterdir() if p.is_file()}
    assert "repo_01.txt" in kept_files
    assert "radamsa_01.txt" not in kept_files
    assert int(filtered["seed_total_pruned_count"]) >= 1


def test_filter_seed_corpus_soft_mode_keeps_family_variants(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "fuzz" / "corpus" / "format_fuzzer"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(8):
        (corpus_dir / f"seed_{idx:02d}.txt").write_text(f"{{{idx}}}\n", encoding="utf-8")
    monkeypatch.setenv("SHERPA_SEED_FILTER_MODE", "soft")
    gen = _make_generator(tmp_path)

    filtered = gen._filter_seed_corpus(
        corpus_dir,
        seed_profile="parser-format",
        required_families=["positional_arguments"],
        target_markers=["format parser"],
    )

    kept_files = sorted(p.name for p in corpus_dir.iterdir() if p.is_file())
    assert len(kept_files) >= 3
    assert filtered["seed_filter_mode"] == "soft"
    assert float(filtered["retention_ratio_ai"]) > 0.3


def test_filter_seed_corpus_off_mode_disables_shape_and_family_rejects(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "fuzz" / "corpus" / "format_fuzzer"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(8):
        (corpus_dir / f"seed_{idx:02d}.txt").write_text(f"{{{idx}}}\n", encoding="utf-8")
    monkeypatch.setenv("SHERPA_SEED_FILTER_MODE", "off")
    gen = _make_generator(tmp_path)

    filtered = gen._filter_seed_corpus(
        corpus_dir,
        seed_profile="parser-format",
        required_families=["positional_arguments"],
        target_markers=["format parser"],
    )

    kept_files = sorted(p.name for p in corpus_dir.iterdir() if p.is_file())
    assert len(kept_files) == 8
    assert filtered["seed_filter_mode"] == "off"
    assert int(filtered["filtered_by_rule_breakdown"]["shape"]) == 0
    assert int(filtered["filtered_by_rule_breakdown"]["family"]) == 0


def test_filter_seed_corpus_parser_numeric_disables_shape_dedup(tmp_path: Path, monkeypatch):
    corpus_dir = tmp_path / "fuzz" / "corpus" / "numeric_fuzzer"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(8):
        (corpus_dir / f"seed_{idx:02d}.txt").write_text(f"id={idx}\n", encoding="utf-8")
    monkeypatch.setenv("SHERPA_SEED_FILTER_MODE", "strict")
    gen = _make_generator(tmp_path)

    filtered = gen._filter_seed_corpus(
        corpus_dir,
        seed_profile="parser-numeric",
        required_families=["delimiter_fragments"],
        target_markers=["numeric parser"],
    )

    kept_files = sorted(p.name for p in corpus_dir.iterdir() if p.is_file())
    assert len(kept_files) >= 3
    assert int(filtered["filtered_by_rule_breakdown"]["shape"]) == 0
