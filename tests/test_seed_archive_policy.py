from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "harness_generator" / "src" / "fuzz_unharnessed_repo.py"


def test_archive_seed_guidance_prefers_real_samples() -> None:
    text = SRC.read_text(encoding="utf-8")
    assert "real-sample-first strategy" in text
    assert "Prefer real archives over hand-crafted malformed bytes." in text
    assert "ensure at least one valid archive sample exists first" in text


def test_archive_seed_filter_has_malformed_ratio_and_magic_only_guards() -> None:
    text = SRC.read_text(encoding="utf-8")
    assert "def _seed_archive_max_malformed_ratio" in text
    assert "def _is_magic_only_archive_seed" in text
    assert "archive_magic_only_rejected_count" in text
    assert "archive_malformed_ratio" in text
    assert "archive_max_malformed_ratio" in text

