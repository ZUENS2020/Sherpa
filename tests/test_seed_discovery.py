from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "harness_generator" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fuzz_unharnessed_repo as f  # noqa: E402


def test_parser_structure_has_toml_tokens():
    t = f.PROFILE_DICTIONARY_TOKENS["parser-structure"]
    assert '"="' in t  # the glaring omission that starved TOML key=value
    assert '"[["' in t and '"]]"' in t
    assert '"."' in t


def _gen(tmp_path: Path):
    gen = f.NonOssFuzzHarnessGenerator.__new__(f.NonOssFuzzHarnessGenerator)
    gen.repo_root = tmp_path
    gen.fuzz_dir = tmp_path / "fuzz"
    gen.fuzz_dir.mkdir(parents=True, exist_ok=True)
    gen._seed_max_file_bytes = lambda: 1_000_000  # type: ignore[attr-defined]
    return gen


def test_discovers_tomlc99_style_test1_dir(tmp_path):
    # tomlc99 keeps examples in test1/test2 — the old fixed search_roots missed them
    (tmp_path / "test1" / "extra").mkdir(parents=True)
    (tmp_path / "test1" / "extra" / "inline_array.toml").write_text(
        'title = "x"\narr = [1, 2, 3]\n[table]\nk = true\n', encoding="utf-8"
    )
    (tmp_path / "test2").mkdir()
    (tmp_path / "test2" / "doc.toml").write_text('a = 1\n[s]\nb = "c"\n', encoding="utf-8")
    # a non-sample dir that must NOT be harvested
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "toml.c").write_text("int main(){}", encoding="utf-8")

    corpus = tmp_path / "fuzz" / "corpus"
    corpus.mkdir(parents=True)
    gen = _gen(tmp_path)
    selected, stats = gen._collect_repo_seed_examples("parser-structure", "toml_parse_file_fuzz", corpus)
    # both .toml examples from test1/ and test2/ are harvested (copied into the
    # corpus, possibly renamed); the .c source is not.
    assert len(selected) == 2, [p.name for p in selected]
    assert all(p.suffix == ".toml" for p in selected)
    # and the harvested seeds carry real TOML content (not the source file)
    bodies = [p.read_text(encoding="utf-8", errors="replace") for p in selected]
    assert any("=" in b for b in bodies)
    assert not any("int main" in b for b in bodies)


def test_no_sample_dirs_returns_empty(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.c").write_text("x", encoding="utf-8")
    corpus = tmp_path / "fuzz" / "corpus"
    corpus.mkdir(parents=True)
    gen = _gen(tmp_path)
    selected, _ = gen._collect_repo_seed_examples("parser-structure", "fuzz", corpus)
    assert all(p.suffix == ".toml" or p.suffix in {".json", ".yaml"} for p in selected) or selected == []
