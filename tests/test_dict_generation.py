from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "harness_generator" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fuzz_unharnessed_repo as f  # noqa: E402


def test_scaffold_filter():
    for junk in ["toml.h", "/tmp/tomlfuzz_XXXXXX", "rb", "wb", "r+b", "%s", "%d",
                 "pngread.c", "include/foo.hpp", "a\\b"]:
        assert f._is_scaffold_dict_literal(junk), junk
    for keep in ["[", "true", "false", "[[", "]]", "key", "=", "---", "null", "0x"]:
        assert not f._is_scaffold_dict_literal(keep), keep


def test_generated_dict_excludes_scaffold_and_prioritizes_grammar(tmp_path):
    # minimal generator instance (bypass __init__)
    gen = f.NonOssFuzzHarnessGenerator.__new__(f.NonOssFuzzHarnessGenerator)
    gen.repo_root = tmp_path
    gen.fuzz_dir = tmp_path / "fuzz"
    gen.fuzz_dir.mkdir(parents=True)
    # a tomlc99-style harness full of scaffold string literals
    (gen.fuzz_dir / "toml_parse_file_fuzz.c").write_text(
        '#include "toml.h"\n'
        'int LLVMFuzzerTestOneInput(const unsigned char* d, unsigned long n){\n'
        '  char tmpname[] = "/tmp/tomlfuzz_XXXXXX";\n'
        '  FILE* fp = fopen(tmpname, "rb");\n'
        '  char errbuf[256];\n'
        '  toml_parse_file(fp, errbuf, sizeof(errbuf));\n'
        '  return 0;\n}\n',
        encoding="utf-8",
    )
    bin_path = gen.fuzz_dir / "out" / "toml_parse_file"
    bin_path.parent.mkdir(parents=True)
    bin_path.write_text("", encoding="utf-8")

    dict_path = gen._generate_dictionary(bin_path, "parser-structure")
    assert dict_path and dict_path.is_file()
    body = dict_path.read_text(encoding="utf-8")

    # scaffold noise must be gone
    assert "toml.h" not in body
    assert "tomlfuzz" not in body
    assert "/tmp" not in body
    assert '"rb"' not in body
    # curated grammar tokens present (libFuzzer dict normalizes to \xNN form:
    # '[' -> \x5b, 'true' -> \x74\x72\x75\x65)
    assert "\\x5b" in body  # '['
    assert "\\x74\\x72\\x75\\x65" in body  # 'true'
    # every emitted token came from the curated profile (no harness scaffold left)
    n_tokens = sum(1 for l in body.splitlines() if l.startswith("token_"))
    assert n_tokens >= 10
