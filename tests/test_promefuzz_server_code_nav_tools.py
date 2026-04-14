from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMEFUZZ_DIR = ROOT / "promefuzz-mcp"
if str(PROMEFUZZ_DIR) not in sys.path:
    sys.path.insert(0, str(PROMEFUZZ_DIR))

from promefuzz_mcp.server_tools import register_tools


class _FakeMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self):
        def _decorator(fn):
            self.tools[fn.__name__] = fn
            return fn

        return _decorator


def _build_meta(tmp_path: Path) -> tuple[Path, Path]:
    src = tmp_path / "demo.c"
    src.write_text(
        "int add(int a,int b){return a+b;}\n"
        "int use_add(){return add(1,2);}\n",
        encoding="utf-8",
    )
    meta = {
        "functions": {
            f"{src}:1:1": {
                "name": "add",
                "declLoc": f"{src}:1:1",
                "heldbyNamespace": "",
                "heldbyClass": "",
            },
            f"{src}:2:1": {
                "name": "use_add",
                "declLoc": f"{src}:2:1",
                "heldbyNamespace": "",
                "heldbyClass": "",
            },
        },
        "classes": {},
    }
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return src, meta_path


def test_code_nav_tools_basic(tmp_path: Path) -> None:
    src, meta_path = _build_meta(tmp_path)
    fake = _FakeMCP()
    register_tools(fake)

    list_defs = fake.tools["list_definitions"]
    read_def = fake.tools["read_definition"]
    read_src = fake.tools["read_source"]
    find_refs = fake.tools["find_references"]

    listed = asyncio.run(list_defs(str(meta_path), symbol_query="add", kind="function", limit=10))
    assert listed["status"] == "success"
    assert listed["count"] >= 2
    assert any(row.get("name") == "add" for row in listed["definitions"])

    def_doc = asyncio.run(read_def(str(meta_path), symbol="add", kind="function", occurrence=0, context_lines=10))
    assert def_doc["status"] == "success"
    assert def_doc["found"] is True
    assert "add(" in str(def_doc.get("snippet") or "")

    src_doc = asyncio.run(read_src(str(src), start_line=1, end_line=1, max_chars=1000))
    assert src_doc["status"] == "success"
    assert "int add" in str(src_doc.get("snippet") or "")

    refs_doc = asyncio.run(find_refs(str(meta_path), symbol="add", source_roots=[str(tmp_path)], limit=20))
    assert refs_doc["status"] == "success"
    assert refs_doc["count"] >= 2
    assert any(bool(r.get("is_definition")) for r in refs_doc["references"])

