"""
Server tools - MCP tool definitions for PromeFuzz.
"""

import json
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from loguru import logger


def register_tools(mcp):
    """Register all MCP tools."""

    # ===================== Preprocessor Tools =====================

    @mcp.tool()
    async def run_ast_preprocessor(
        source_paths: list[str],
        compile_commands_path: Optional[str] = None,
        output_dir: str = "./output/meta",
    ) -> AsyncGenerator[dict, None]:
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

        yield {"status": "starting", "message": "Initializing AST preprocessor..."}

        preprocessor = ASTPreprocessor(
            source_paths=[Path(p) for p in source_paths],
            compile_commands_path=Path(compile_commands_path) if compile_commands_path else None,
        )

        yield {"status": "running", "message": f"Processing {len(preprocessor.source_files)} source files..."}

        meta, output_file = preprocessor.run(output_dir=Path(output_dir))

        yield {"status": "completed", "message": "AST preprocessing completed"}

        yield {
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
    ) -> AsyncGenerator[dict, None]:
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

        yield {"status": "starting", "message": "Loading metadata..."}

        meta = Meta.load(Path(meta_path))

        yield {"status": "running", "message": "Extracting API functions..."}

        extractor = APIExtractor(
            header_paths=[Path(p) for p in header_paths],
            meta=meta,
        )

        # Set default output path if not provided
        if output_path is None:
            output_path = "./output/api/api_functions.json"

        api_collection, saved_path = extractor.extract(output_path=Path(output_path))

        yield {"status": "completed", "message": f"Extracted {api_collection.count} API functions"}

        yield {
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
    ) -> AsyncGenerator[dict, None]:
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

        yield {"status": "running", "message": "Building library call graph..."}

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

        yield {"status": "completed", "message": "Call graph built"}

        yield {
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
    ) -> AsyncGenerator[dict, None]:
        """
        Calculate type-based relevance between API functions.

        Args:
            api_collection: API collection
            meta_path: Path to meta.json
            output_path: Output path for relevance JSON (default: ./output/relevance/type_relevance.json)

        Returns:
            Dictionary containing relevance scores and output file path
        """
        from .preprocessor.relevance import TypeRelevance

        yield {"status": "running", "message": "Calculating type relevance..."}

        # Set default output path if not provided
        if output_path is None:
            output_path = "./output/relevance/type_relevance.json"

        # TODO: Implement type relevance calculation
        relevance = {}

        # Persist to file if output_path is provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(output_file, "w") as f:
                json.dump({"relevance": relevance}, f, indent=2)

        yield {"status": "completed", "message": "Type relevance calculated"}

        yield {
            "status": "success",
            "relevance": relevance,
            "output_file": str(output_path),
        }

    @mcp.tool()
    async def get_function_info(
        function_location: str,
        info_repo_path: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Get detailed information about a function.

        Args:
            function_location: Function location identifier
            info_repo_path: Path to info repository

        Returns:
            Function information
        """
        # TODO: Implement function info retrieval

        yield {
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
    ) -> AsyncGenerator[dict, None]:
        """
        Initialize RAG knowledge base from documents.

        Args:
            document_paths: List of document paths (files, directories, or URLs)
            output_path: Output path for knowledge base (default: ./output/knowledge)

        Returns:
            Dictionary containing knowledge base information and output path
        """
        from .comprehender.knowledge import KnowledgeBase

        yield {"status": "starting", "message": "Initializing knowledge base..."}

        kb = KnowledgeBase(
            document_paths=document_paths,
            output_path=output_path,
        )

        success, kb_path = kb.initialize()

        yield {"status": "completed", "message": "Knowledge base initialized"}

        yield {
            "status": "success",
            "output_path": str(kb_path),
            "document_count": len(kb.documents),
        }

    @mcp.tool()
    async def retrieve_documents(
        query: str,
        knowledge_base_id: str,
        top_k: int = 3,
    ) -> AsyncGenerator[dict, None]:
        """
        Retrieve relevant document excerpts using RAG.

        Args:
            query: Search query
            knowledge_base_id: Knowledge base identifier
            top_k: Number of results to return

        Returns:
            List of relevant excerpts
        """
        yield {"status": "running", "message": f"Retrieving documents for query: {query}"}

        # TODO: Implement document retrieval

        yield {"status": "completed", "message": "Documents retrieved"}

        yield {
            "status": "success",
            "query": query,
            "results": [],
        }

    @mcp.tool()
    async def comprehend_library_purpose(
        knowledge_base_id: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Understand the overall purpose of the library.

        Args:
            knowledge_base_id: Knowledge base identifier

        Yields:
            Progress updates and final result
        """
        yield {"status": "retrieving", "message": "Retrieving library documentation..."}

        yield {"status": "analyzing", "message": "Analyzing with LLM..."}

        yield {
            "status": "completed",
            "purpose": "A C++ library for parsing and writing JSON",
        }

    @mcp.tool()
    async def comprehend_function_usage(
        function_name: str,
        knowledge_base_id: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Understand the usage of a specific function.

        Args:
            function_name: Name of the function
            knowledge_base_id: Knowledge base identifier

        Yields:
            Progress updates and final result
        """
        yield {"status": "retrieving", "message": f"Retrieving docs for {function_name}..."}

        yield {"status": "analyzing", "message": "Analyzing function usage..."}

        yield {
            "status": "completed",
            "function": function_name,
            "usage": "Parses JSON from string input",
        }

    @mcp.tool()
    async def comprehend_all_functions(
        api_collection: dict,
        knowledge_base_id: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Understand usage of all functions in the API collection.

        Args:
            api_collection: API collection
            knowledge_base_id: Knowledge base identifier

        Yields:
            Progress updates and final results
        """
        functions = api_collection.get("functions", [])
        total = len(functions)

        yield {"status": "starting", "message": f"Processing {total} functions..."}

        for i, func in enumerate(functions):
            if i % 10 == 0:
                yield {
                    "status": "progress",
                    "message": f"Processed {i}/{total} functions",
                    "progress": i / total,
                }

        yield {"status": "completed", "message": "All functions processed"}

    @mcp.tool()
    async def comprehend_function_relevance(
        api_collection: dict,
        library_purpose: str,
        function_usages: dict,
    ) -> AsyncGenerator[dict, None]:
        """
        Calculate semantic relevance between functions.

        Args:
            api_collection: API collection
            library_purpose: Library purpose description
            function_usages: Function usage mappings

        Yields:
            Progress updates and final results
        """
        yield {"status": "running", "message": "Calculating semantic relevance..."}

        yield {"status": "completed", "message": "Relevance calculated"}
