from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
SRC_DIR = ROOT / "harness_generator" / "src"
for p in (APP_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import workflow_graph  # noqa: E402


def test_aliases_include_both_fuzz_and_fuzzer_suffix() -> None:
    al = workflow_graph._execution_target_fuzzer_aliases(
        {"target_name": "png_read_image", "api": "png_read_image"}
    )
    assert "png_read_image" in al
    assert "png_read_image_fuzz" in al
    assert "png_read_image_fuzzer" in al  # the regression: was missing


def test_filter_matches_fuzzer_suffixed_binary() -> None:
    # Binary built as <api>_fuzzer must match the execution-plan target <api>;
    # otherwise the run stage records no run_details despite a working binary.
    bins = [
        Path("/x/fuzz/out/png_read_image_fuzzer"),
        Path("/x/fuzz/out/png_read_info_fuzzer"),
    ]
    targets = [{"target_name": "png_read_image", "api": "png_read_image"}]
    keep = workflow_graph._filter_fuzzer_bins_by_execution_plan(bins, targets)
    assert [p.name for p in keep] == ["png_read_image_fuzzer"]


def test_filter_matches_fuzz_suffixed_binary() -> None:
    bins = [Path("/x/fuzz/out/archive_read_fuzz")]
    targets = [{"target_name": "archive_read", "api": "archive_read"}]
    keep = workflow_graph._filter_fuzzer_bins_by_execution_plan(bins, targets)
    assert [p.name for p in keep] == ["archive_read_fuzz"]


def test_filter_matches_exact_name_binary() -> None:
    bins = [Path("/x/fuzz/out/png_read_image")]
    targets = [{"target_name": "png_read_image", "api": "png_read_image"}]
    keep = workflow_graph._filter_fuzzer_bins_by_execution_plan(bins, targets)
    assert [p.name for p in keep] == ["png_read_image"]


def test_target_already_fuzzer_suffixed_still_resolves() -> None:
    al = workflow_graph._execution_target_fuzzer_aliases({"target_name": "foo_fuzzer"})
    assert "foo" in al
    assert "foo_fuzzer" in al
