"""
Preprocessor module - Call graph building.
"""

from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger
import json
import subprocess
import tempfile

from ..build import BinaryBuilder
from .api_extractor import APICollection


class CallGraphBuilder:
    """Build call graph from source code."""

    def __init__(
        self,
        source_files: List[Path],
        compile_commands_path: Optional[Path] = None,
        api_collection: Optional[APICollection] = None,
    ):
        self.source_files = source_files
        self.compile_commands_path = compile_commands_path
        self.api_collection = api_collection
        self.builder = BinaryBuilder()

    def build(self, output_path: Optional[Path] = None) -> Tuple[dict, Optional[Path]]:
        """
        Build call graph.

        Args:
            output_path: Optional output path for call graph JSON file

        Returns:
            Tuple of (call graph dict, output file path or None)
        """
        cgprocessor_bin = self.builder.get_cgprocessor_bin()

        all_edges = []

        for source_file in self.source_files:
            edges = self._process_file(source_file, cgprocessor_bin)
            all_edges.extend(edges)

        result = {"edges": all_edges}
        logger.info(f"Built call graph with {len(all_edges)} edges")

        # Persist to file if output_path is provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved call graph to {output_path}")
            return result, output_path

        return result, None

    def _process_file(self, source_file: Path, cgprocessor_bin: Path) -> list:
        """Process a single source file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cg_file = Path(tmp_dir) / "cg.json"

            cmd = f"{cgprocessor_bin} {source_file} -o {cg_file}"
            if self.compile_commands_path:
                cmd += f" -p {self.compile_commands_path.resolve().parent}"

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"Failed to process {source_file}: {result.stderr}")
                return []

            if cg_file.exists():
                with open(cg_file, "r") as f:
                    data = json.load(f)
                edges = []
                for calling_info in data.values():
                    edges.append((
                        calling_info.get("callerName", ""),
                        calling_info.get("callerDeclLoc", ""),
                        calling_info.get("calleeName", ""),
                        calling_info.get("calleeDeclLoc", ""),
                    ))
                return edges

            return []
