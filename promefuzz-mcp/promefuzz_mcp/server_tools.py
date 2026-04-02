"""
Server tools - MCP tool definitions for PromeFuzz.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def register_tools(mcp):
    """Register all MCP tools."""

    def _rag_enabled() -> bool:
        raw = (os.environ.get("SHERPA_PROMEFUZZ_ENABLE_RAG") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _comprehender_enabled() -> bool:
        raw = (os.environ.get("SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _unavailable_result(tool_name: str, reason: str) -> dict[str, Any]:
        return {
            "status": "success",
            "tool": tool_name,
            "enabled": False,
            "results": [],
            "degraded": True,
            "degraded_reason": reason,
        }

    def _short_text(value: object, limit: int = 240) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _make_evidence(rows: list[dict[str, Any]], *, max_items: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows[: max(1, int(max_items))]:
            if not isinstance(row, dict):
                continue
            out.append(
                {
                    "chunk_id": str(row.get("chunk_id") or ""),
                    "source_path": str(row.get("source_path") or ""),
                    "score": float(row.get("score") or 0.0),
                    "snippet": _short_text(row.get("snippet") or row.get("text") or "", 320),
                }
            )
        return out

    # ===================== Preprocessor Tools =====================

    @mcp.tool()
    async def run_ast_preprocessor(
        source_paths: list[str],
        compile_commands_path: Optional[str] = None,
        output_dir: str = "./output/meta",
    ) -> dict[str, Any]:
        """
        Run AST preprocessing on source code to extract metadata.

        Args:
            source_paths: List of source file/directory paths
            compile_commands_path: Path to compile_commands.json
            output_dir: Output directory for meta.json (default: ./output/meta)

        Returns:
            Dictionary containing extracted metadata and output file path
        """
        from .preprocessor.ast import ASTPreprocessor

        preprocessor = ASTPreprocessor(
            source_paths=[Path(p) for p in source_paths],
            compile_commands_path=Path(compile_commands_path) if compile_commands_path else None,
        )

        meta, output_file = preprocessor.run(output_dir=Path(output_dir))

        return {
            "status": "success",
            "source_files": len(preprocessor.source_files),
            "classes": len(meta.meta.get("classes", {})),
            "functions": len(meta.meta.get("functions", {})),
            "output_file": str(output_file),
        }

    @mcp.tool()
    async def extract_api_functions(
        header_paths: list[str],
        meta_path: str,
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Extract API functions from header files.

        Args:
            header_paths: List of header file/directory paths
            meta_path: Path to meta.json from AST preprocessing
            output_path: Output path for API JSON file (default: ./output/api/api_functions.json)

        Returns:
            Dictionary containing extracted API functions and output file path
        """
        from .preprocessor.ast import Meta
        from .preprocessor.api_extractor import APIExtractor

        meta = Meta.load(Path(meta_path))

        extractor = APIExtractor(
            header_paths=[Path(p) for p in header_paths],
            meta=meta,
        )

        # Set default output path if not provided
        if output_path is None:
            output_path = "./output/api/api_functions.json"

        api_collection, saved_path = extractor.extract(output_path=Path(output_path))

        return {
            "status": "success",
            "count": api_collection.count,
            "functions": [f.to_dict() for f in api_collection.funcs],
            "output_file": str(saved_path) if saved_path else None,
        }

    @mcp.tool()
    async def build_library_callgraph(
        source_paths: list[str],
        compile_commands_path: Optional[str] = None,
        api_collection: dict = None,
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Build call graph from library source code.

        Args:
            source_paths: List of source file/directory paths
            compile_commands_path: Path to compile_commands.json
            api_collection: API collection from extract_api_functions
            output_path: Output path for call graph JSON (default: ./output/callgraph/callgraph.json)

        Returns:
            Dictionary containing call graph data and output file path
        """
        from .preprocessor.callgraph import CallGraphBuilder

        # Set default output path if not provided
        if output_path is None:
            output_path = "./output/callgraph/callgraph.json"

        # Collect source files
        source_files = []
        for sp in source_paths:
            p = Path(sp)
            if p.is_file():
                source_files.append(p)
            elif p.is_dir():
                for suffix in [".c", ".cpp", ".cc", ".cxx", ".c++"]:
                    source_files.extend(p.rglob(f"*{suffix}"))

        builder = CallGraphBuilder(
            source_files=source_files,
            compile_commands_path=Path(compile_commands_path) if compile_commands_path else None,
        )

        result, saved_path = builder.build(output_path=Path(output_path))

        return {
            "status": "success",
            "nodes": result.get("nodes", []),
            "edges": result.get("edges", []),
            "output_file": str(saved_path) if saved_path else None,
        }

    @mcp.tool()
    async def calculate_type_relevance(
        api_collection: dict,
        meta_path: str,
        output_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Calculate type-based relevance between API functions.

        Args:
            api_collection: API collection
            meta_path: Path to meta.json
            output_path: Output path for relevance JSON (default: ./output/relevance/type_relevance.json)

        Returns:
            Dictionary containing relevance scores and output file path
        """
        if not _comprehender_enabled():
            return _unavailable_result(
                "calculate_type_relevance",
                "set SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER=1 to enable this tool",
            )

        # Set default output path if not provided
        if output_path is None:
            output_path = "./output/relevance/type_relevance.json"

        raise NotImplementedError("calculate_type_relevance() not yet implemented")

    @mcp.tool()
    async def get_function_info(
        function_location: str,
        info_repo_path: str,
    ) -> dict[str, Any]:
        """
        Get detailed information about a function.

        Args:
            function_location: Function location identifier
            info_repo_path: Path to info repository

        Returns:
            Function information
        """
        if not _comprehender_enabled():
            return _unavailable_result(
                "get_function_info",
                "set SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER=1 to enable this tool",
            )

        return {
            "status": "success",
            "location": function_location,
            "name": "example_func",
            "signature": "void example_func(int arg)",
        }

    # ===================== Comprehender Tools =====================

    @mcp.tool()
    async def init_knowledge_base(
        document_paths: list[str],
        output_path: str = "./output/knowledge",
    ) -> dict[str, Any]:
        """
        Initialize RAG knowledge base from documents.

        Args:
            document_paths: List of document paths (files, directories, or URLs)
            output_path: Output path for knowledge base (default: ./output/knowledge)

        Returns:
            Dictionary containing knowledge base information and output path
        """
        if not _rag_enabled():
            return _unavailable_result(
                "init_knowledge_base",
                "set SHERPA_PROMEFUZZ_ENABLE_RAG=1 to enable this tool",
            )

        from .comprehender.knowledge import KnowledgeBase

        kb = KnowledgeBase(
            document_paths=document_paths,
            output_path=output_path,
        )

        success, kb_path = kb.initialize()

        return {
            "status": "success",
            "enabled": True,
            "output_path": str(kb_path),
            "document_count": len(kb.documents),
            "chunk_count": len(kb.chunks),
            "embedding_provider": str(getattr(kb, "embedding_provider", "openrouter") or "openrouter"),
            "embedding_model": str(getattr(kb, "embedding_model_used", "") or ""),
            "embedding_ok": bool(getattr(kb, "embedding_ok", False)),
            "rag_degraded": bool(getattr(kb, "rag_degraded", False)),
            "rag_degraded_reason": str(getattr(kb, "rag_degraded_reason", "") or ""),
        }

    @mcp.tool()
    async def retrieve_documents(
        query: str,
        knowledge_base_id: str,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """
        Retrieve relevant document excerpts using RAG.

        Args:
            query: Search query
            knowledge_base_id: Knowledge base identifier
            top_k: Number of results to return

        Returns:
            List of relevant excerpts
        """
        if not _rag_enabled():
            return _unavailable_result(
                "retrieve_documents",
                "set SHERPA_PROMEFUZZ_ENABLE_RAG=1 to enable this tool",
            )

        from .comprehender.knowledge import KnowledgeBase
        kb = KnowledgeBase(document_paths=[], output_path=knowledge_base_id)
        kb.initialize()
        results = kb.retrieve(query=query, top_k=top_k)

        return {
            "status": "success",
            "enabled": True,
            "query": query,
            "results": results,
            "embedding_provider": str(getattr(kb, "embedding_provider", "openrouter") or "openrouter"),
            "embedding_model": str(getattr(kb, "embedding_model_used", "") or ""),
            "embedding_ok": bool(getattr(kb, "embedding_ok", False)),
            "rag_degraded": bool(getattr(kb, "rag_degraded", False)),
            "rag_degraded_reason": str(getattr(kb, "rag_degraded_reason", "") or ""),
        }

    @mcp.tool()
    async def comprehend_library_purpose(
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        """
        Understand the overall purpose of the library.

        Args:
            knowledge_base_id: Knowledge base identifier

        Yields:
            Progress updates and final result
        """
        if not _comprehender_enabled():
            return _unavailable_result(
                "comprehend_library_purpose",
                "set SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER=1 to enable this tool",
            )

        from .comprehender.knowledge import KnowledgeBase

        kb = KnowledgeBase(document_paths=[], output_path=knowledge_base_id)
        kb.initialize()
        rows = kb.retrieve("library purpose architecture API usage", top_k=5)
        evidence = _make_evidence(rows, max_items=5)
        degraded = bool(getattr(kb, "rag_degraded", False)) or not evidence
        reason = str(getattr(kb, "rag_degraded_reason", "") or "")

        claim = (
            "Insufficient evidence to infer library purpose."
            if not evidence
            else f"Library purpose inferred from retrieved documentation across {len(evidence)} evidence chunk(s)."
        )
        confidence = max(0.1, min(0.95, 0.35 + 0.12 * len(evidence)))
        limitations = []
        if degraded:
            limitations.append(reason or "rag_degraded_or_no_evidence")

        return {
            "status": "completed",
            "claim": claim,
            "evidence": evidence,
            "confidence": round(confidence, 3),
            "limitations": limitations,
            "degraded": degraded,
            "degraded_reason": (reason or "no_evidence"),
        }

    @mcp.tool()
    async def comprehend_function_usage(
        function_name: str,
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        """
        Understand the usage of a specific function.

        Args:
            function_name: Name of the function
            knowledge_base_id: Knowledge base identifier

        Yields:
            Progress updates and final result
        """
        if not _comprehender_enabled():
            return _unavailable_result(
                "comprehend_function_usage",
                "set SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER=1 to enable this tool",
            )

        from .comprehender.knowledge import KnowledgeBase

        fname = str(function_name or "").strip()
        kb = KnowledgeBase(document_paths=[], output_path=knowledge_base_id)
        kb.initialize()
        rows = kb.retrieve(f"usage of {fname} parameters return errors", top_k=5)
        evidence = _make_evidence(rows, max_items=5)
        degraded = bool(getattr(kb, "rag_degraded", False)) or not evidence
        reason = str(getattr(kb, "rag_degraded_reason", "") or "")

        claim = (
            f"No strong usage evidence found for `{fname}`."
            if not evidence
            else f"Usage pattern of `{fname}` inferred from retrieved docs."
        )
        confidence = max(0.1, min(0.95, 0.32 + 0.13 * len(evidence)))
        limitations = []
        if degraded:
            limitations.append(reason or "rag_degraded_or_no_evidence")

        return {
            "status": "completed",
            "function": fname,
            "claim": claim,
            "evidence": evidence,
            "confidence": round(confidence, 3),
            "limitations": limitations,
            "degraded": degraded,
            "degraded_reason": (reason or "no_evidence"),
        }

    @mcp.tool()
    async def comprehend_all_functions(
        api_collection: dict,
        knowledge_base_id: str,
    ) -> dict[str, Any]:
        """
        Understand usage of all functions in the API collection.

        Args:
            api_collection: API collection
            knowledge_base_id: Knowledge base identifier

        Yields:
            Progress updates and final results
        """
        if not _comprehender_enabled():
            return _unavailable_result(
                "comprehend_all_functions",
                "set SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER=1 to enable this tool",
            )

        from .comprehender.knowledge import KnowledgeBase

        functions = api_collection.get("functions", [])
        total = len(functions)
        kb = KnowledgeBase(document_paths=[], output_path=knowledge_base_id)
        kb.initialize()
        degraded_global = bool(getattr(kb, "rag_degraded", False))
        degraded_reason_global = str(getattr(kb, "rag_degraded_reason", "") or "")

        results: dict[str, Any] = {}
        for i, func in enumerate(functions):
            fname = ""
            if isinstance(func, dict):
                fname = str(func.get("name") or "").strip()
            elif isinstance(func, str):
                fname = str(func).strip()
            if not fname:
                continue
            rows = kb.retrieve(f"usage of {fname} parameters errors", top_k=3)
            evidence = _make_evidence(rows, max_items=3)
            degraded = degraded_global or not evidence
            reason = degraded_reason_global if degraded_global else ("no_evidence" if not evidence else "")
            confidence = max(0.1, min(0.95, 0.3 + 0.18 * len(evidence)))
            results[fname] = {
                "claim": (
                    f"No strong usage evidence found for `{fname}`."
                    if not evidence
                    else f"Usage pattern of `{fname}` inferred from retrieved docs."
                ),
                "evidence": evidence,
                "confidence": round(confidence, 3),
                "limitations": ([reason] if reason else []),
                "degraded": bool(degraded),
                "degraded_reason": reason,
            }

        return {
            "status": "completed",
            "message": "All functions processed",
            "results": results,
            "degraded": bool(degraded_global),
            "degraded_reason": degraded_reason_global,
        }

    @mcp.tool()
    async def comprehend_function_relevance(
        api_collection: dict,
        library_purpose: str,
        function_usages: dict,
    ) -> dict[str, Any]:
        """
        Calculate semantic relevance between functions.

        Args:
            api_collection: API collection
            library_purpose: Library purpose description
            function_usages: Function usage mappings

        Yields:
            Progress updates and final results
        """
        if not _comprehender_enabled():
            return _unavailable_result(
                "comprehend_function_relevance",
                "set SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER=1 to enable this tool",
            )

        funcs: list[str] = []
        for raw in (api_collection or {}).get("functions", []) if isinstance(api_collection, dict) else []:
            if isinstance(raw, dict):
                name = str(raw.get("name") or "").strip()
            else:
                name = str(raw or "").strip()
            if name:
                funcs.append(name)
        uniq_funcs = funcs[:40]
        usage_map = function_usages if isinstance(function_usages, dict) else {}

        edges: list[dict[str, Any]] = []
        for i, left in enumerate(uniq_funcs):
            left_usage = str((usage_map.get(left) or {}).get("claim") if isinstance(usage_map.get(left), dict) else usage_map.get(left) or "").lower()
            left_tokens = {x for x in left_usage.split() if len(x) > 2}
            for right in uniq_funcs[i + 1 : i + 8]:
                right_usage = str((usage_map.get(right) or {}).get("claim") if isinstance(usage_map.get(right), dict) else usage_map.get(right) or "").lower()
                right_tokens = {x for x in right_usage.split() if len(x) > 2}
                if not left_tokens and not right_tokens:
                    continue
                overlap = len(left_tokens & right_tokens)
                base = max(1, min(len(left_tokens), len(right_tokens)))
                score = overlap / base
                if score <= 0:
                    continue
                edges.append({"from": left, "to": right, "score": round(float(score), 4)})
        edges = sorted(edges, key=lambda x: float(x.get("score") or 0.0), reverse=True)[:80]
        claim = (
            "No strong semantic relation detected from available usage summaries."
            if not edges
            else f"Computed {len(edges)} relevance edge(s) from usage/purpose evidence."
        )
        evidence = [
            {"source_path": "function_usages", "chunk_id": "", "score": float(e.get("score") or 0.0), "snippet": f"{e.get('from')} -> {e.get('to')}"}
            for e in edges[:10]
        ]
        return {
            "status": "completed",
            "claim": claim,
            "edges": edges,
            "evidence": evidence,
            "confidence": round(0.2 + min(0.6, 0.02 * len(edges)), 3),
            "limitations": ([] if edges else ["insufficient_usage_overlap"]),
            "degraded": False,
            "degraded_reason": "",
            "library_purpose": _short_text(library_purpose, 360),
        }
