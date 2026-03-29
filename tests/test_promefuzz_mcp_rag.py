from __future__ import annotations

import sys
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MCP_DIR = ROOT / "promefuzz-mcp"
KNOWLEDGE_PY = MCP_DIR / "promefuzz_mcp" / "comprehender" / "knowledge.py"
spec = importlib.util.spec_from_file_location("promefuzz_knowledge", KNOWLEDGE_PY)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules["promefuzz_knowledge"] = module
spec.loader.exec_module(module)
KnowledgeBase = module.KnowledgeBase


def test_knowledge_base_initialize_and_retrieve(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "readme.md").write_text(
        "zlib has inflate and deflate functions for compression.\n"
        "inflate stream parser accepts compressed input blocks.\n",
        encoding="utf-8",
    )
    kb_dir = tmp_path / "kb"
    kb = KnowledgeBase(document_paths=[str(docs)], output_path=str(kb_dir))
    ok, out_path = kb.initialize()
    assert ok is True
    assert out_path == kb_dir
    assert (kb_dir / "metadata.json").is_file()
    assert (kb_dir / "index.json").is_file()
    assert (kb_dir / "chunks.json").is_file()
    assert len(kb.documents) >= 1
    assert len(kb.chunks) >= 1

    rows = kb.retrieve("inflate parser", top_k=3)
    assert len(rows) >= 1
    first = rows[0]
    assert "source_path" in first
    assert "snippet" in first
    assert float(first.get("score") or 0) > 0


def test_knowledge_base_loads_cache_without_source_paths(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "api.md").write_text("fmt parser uses replacement fields and format specs", encoding="utf-8")
    kb_dir = tmp_path / "kb"
    kb1 = KnowledgeBase(document_paths=[str(docs)], output_path=str(kb_dir))
    kb1.initialize()

    kb2 = KnowledgeBase(document_paths=[], output_path=str(kb_dir))
    ok2, _ = kb2.initialize()
    assert ok2 is True
    rows = kb2.retrieve("replacement fields", top_k=2)
    assert len(rows) >= 1
