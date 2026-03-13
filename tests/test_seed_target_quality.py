from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "harness_generator" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fuzz_unharnessed_repo import (
    NonOssFuzzHarnessGenerator,
    _classify_seed_family,
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
