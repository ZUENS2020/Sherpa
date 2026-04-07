from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMEFUZZ_DIR = ROOT / "promefuzz-mcp"
if str(PROMEFUZZ_DIR) not in sys.path:
    sys.path.insert(0, str(PROMEFUZZ_DIR))

from promefuzz_mcp.preprocessor.ast import ASTPreprocessor, Meta


class _BadMeta:
    def __init__(self) -> None:
        self.meta = None


def test_ast_preprocessor_skips_invalid_meta_payload(tmp_path: Path) -> None:
    src_a = tmp_path / "a.c"
    src_b = tmp_path / "b.c"
    src_a.write_text("int a(void){return 0;}\n", encoding="utf-8")
    src_b.write_text("int b(void){return 0;}\n", encoding="utf-8")

    pre = ASTPreprocessor(source_paths=[tmp_path])
    pre.source_files = [src_a, src_b]
    pre.builder = type("_FakeBuilder", (), {"get_preprocessor_bin": lambda self: Path("/tmp/preprocessor")})()

    def _fake_process_file(source_file: Path, _pre_bin: Path):
        if source_file.name == "a.c":
            return Meta({"functions": {"a": {"name": "a"}}})
        return _BadMeta()

    pre._process_file = _fake_process_file  # type: ignore[method-assign]

    merged, output_file = pre.run(output_dir=tmp_path / "out")

    assert output_file.is_file()
    assert "functions" in merged.meta
    assert "a" in merged.meta.get("functions", {})
    assert str(src_b) in pre.invalid_meta_files
