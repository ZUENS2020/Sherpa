from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCP_SRC = ROOT / "promefuzz-mcp"
if str(MCP_SRC) not in sys.path:
    sys.path.insert(0, str(MCP_SRC))

from promefuzz_mcp.preprocessor.api_extractor import (  # noqa: E402
    APIExtractor,
    APIFunction,
    _classify_header_public,
)


def test_classify_public_vs_internal_headers(tmp_path: Path) -> None:
    repo = tmp_path / "tree-sitter"
    repo.mkdir()

    public = repo / "lib" / "include" / "tree_sitter" / "api.h"
    internal = repo / "lib" / "src" / "array.h"
    binding = repo / "lib" / "binding_web" / "lib" / "tree-sitter.h"
    toplevel = repo / "tree_sitter.h"
    for p in (public, internal, binding, toplevel):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("/* h */\n", encoding="utf-8")

    assert _classify_header_public(public, repo) is True
    assert _classify_header_public(toplevel, repo) is True
    assert _classify_header_public(internal, repo) is False
    assert _classify_header_public(binding, repo) is False


def test_extract_tags_is_public(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    pub_hdr = repo / "include" / "lib.h"
    int_hdr = repo / "lib" / "src" / "internal.h"
    for p in (pub_hdr, int_hdr):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("/* h */\n", encoding="utf-8")

    # Minimal stand-in for preprocessor Meta: only `.meta` dict is used.
    class _Meta:
        def __init__(self, functions: dict) -> None:
            self.meta = {"functions": functions}

    meta = _Meta(
        {
            f"{pub_hdr}:1:1": {"name": "public_fn", "declLoc": f"{pub_hdr}:1:1"},
            f"{int_hdr}:2:1": {"name": "_internal_fn", "declLoc": f"{int_hdr}:2:1"},
        }
    )

    extractor = APIExtractor(
        header_paths=[repo / "include", repo],
        meta=meta,
        repo_root=repo,
    )
    collection, _ = extractor.extract()
    by_name = {f.name: f for f in collection.funcs}

    assert by_name["public_fn"].is_public is True
    assert by_name["_internal_fn"].is_public is False
    # to_dict round-trips the flag
    assert APIFunction("h", "x", "h:1:1", "h:1:1", is_public=False).to_dict()["is_public"] is False
