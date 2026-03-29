from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MCP_DIR = ROOT / "promefuzz-mcp"

if "loguru" not in sys.modules:
    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    class _DummyLoguru:
        logger = _DummyLogger()

    sys.modules["loguru"] = _DummyLoguru()

if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

from promefuzz_mcp.comprehender.knowledge import KnowledgeBase
from promefuzz_mcp.server_tools import register_tools


class _DummyMCP:
    def __init__(self):
        self.tools: dict[str, Any] = {}

    def tool(self):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


async def _collect(gen):
    out = []
    async for item in gen:
        out.append(item)
    return out


def test_init_knowledge_base_reports_embedding_degraded_without_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("SHERPA_PROMEFUZZ_ENABLE_RAG", "1")
    monkeypatch.setenv("SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER", "1")

    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "readme.md").write_text("zlib inflate and deflate usage examples", encoding="utf-8")
    kb_out = tmp_path / "kb"

    mcp = _DummyMCP()
    register_tools(mcp)
    rows = asyncio.run(
        _collect(
            mcp.tools["init_knowledge_base"](
                document_paths=[str(docs)],
                output_path=str(kb_out),
            )
        )
    )
    final = rows[-1]
    assert final.get("status") == "success"
    assert final.get("enabled") is True
    assert final.get("embedding_provider") == "openrouter"
    assert final.get("embedding_ok") is False
    assert final.get("rag_degraded") is True


def test_retrieve_documents_returns_rag_status_fields(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("SHERPA_PROMEFUZZ_ENABLE_RAG", "1")
    monkeypatch.setenv("SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER", "1")

    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "api.md").write_text("inflate parses compressed blocks from input stream", encoding="utf-8")
    kb_dir = tmp_path / "kb"
    kb = KnowledgeBase(document_paths=[str(docs)], output_path=str(kb_dir))
    kb.initialize()

    mcp = _DummyMCP()
    register_tools(mcp)
    rows = asyncio.run(
        _collect(
            mcp.tools["retrieve_documents"](
                query="inflate parser",
                knowledge_base_id=str(kb_dir),
                top_k=3,
            )
        )
    )
    final = rows[-1]
    assert final.get("status") == "success"
    assert isinstance(final.get("results"), list)
    assert final.get("embedding_provider") == "openrouter"
    assert "embedding_ok" in final
    assert "rag_degraded" in final


def test_comprehend_function_usage_outputs_evidence_contract(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("SHERPA_PROMEFUZZ_ENABLE_RAG", "1")
    monkeypatch.setenv("SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER", "1")

    docs = tmp_path / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "usage.md").write_text(
        "Function inflate() expands compressed bytes.\n"
        "Function deflate() compresses raw data.\n",
        encoding="utf-8",
    )
    kb_dir = tmp_path / "kb"
    kb = KnowledgeBase(document_paths=[str(docs)], output_path=str(kb_dir))
    kb.initialize()

    mcp = _DummyMCP()
    register_tools(mcp)
    rows = asyncio.run(
        _collect(
            mcp.tools["comprehend_function_usage"](
                function_name="inflate",
                knowledge_base_id=str(kb_dir),
            )
        )
    )
    final = rows[-1]
    assert final.get("status") == "completed"
    assert isinstance(final.get("claim"), str) and final.get("claim")
    assert isinstance(final.get("evidence"), list)
    assert isinstance(final.get("confidence"), float)
    assert "degraded" in final
    assert "degraded_reason" in final
    assert isinstance(final.get("limitations"), list)
