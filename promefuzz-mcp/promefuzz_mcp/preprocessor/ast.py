"""
Preprocessor module - AST preprocessing.
"""

from pathlib import Path
from typing import Optional, Tuple
from loguru import logger
import subprocess
import json
import tempfile
import shutil

from ..build import BinaryBuilder


class Meta:
    """Metadata container."""

    def __init__(self, meta_dict: dict):
        self.meta = meta_dict

    @classmethod
    def load(cls, path: Path) -> "Meta":
        """Load meta from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data)

    def dump(self, path: Path):
        """Dump meta to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.meta, f, indent=2)


class ASTPreprocessor:
    """AST Preprocessor for C/C++ source files."""

    def __init__(
        self,
        source_paths: list[Path],
        compile_commands_path: Optional[Path] = None,
        pool_size: int = 1,
    ):
        self.source_paths = source_paths
        self.compile_commands_path = compile_commands_path
        self.pool_size = pool_size
        self.source_files: list[Path] = []

        # Get binary builder
        self.builder = BinaryBuilder()

        # Collect source files
        self._collect_source_files()

    def _collect_source_files(self):
        """Collect all source files from source paths."""
        suffixes = [".c", ".cpp", ".cc", ".cxx", ".c++"]

        for source_path in self.source_paths:
            if source_path.is_file():
                self.source_files.append(source_path)
            elif source_path.is_dir():
                for suffix in suffixes:
                    self.source_files.extend(source_path.rglob(f"*{suffix}"))

        logger.info(f"Found {len(self.source_files)} source files")

    def run(self, output_dir: Optional[Path] = None) -> Tuple["Meta", Path]:
        """
        Run AST preprocessing on all source files.

        Args:
            output_dir: Optional output directory for meta.json. Defaults to ./output/meta

        Returns:
            Tuple of (Meta object, output file path)
        """
        preprocessor_bin = self.builder.get_preprocessor_bin()

        # Default output directory
        if output_dir is None:
            output_dir = Path("./output/meta")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "meta.json"

        all_meta = {}

        for source_file in self.source_files:
            meta = self._process_file(source_file, preprocessor_bin)
            all_meta.update(meta.meta)

        # Persist to file
        final_meta = Meta(all_meta)
        final_meta.dump(output_file)
        logger.info(f"Saved meta to {output_file}")

        return final_meta, output_file

    def _process_file(self, source_file: Path, preprocessor_bin: Path) -> Meta:
        """Process a single source file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta_file = Path(tmp_dir) / "meta.json"

            cmd = f"{preprocessor_bin} {source_file} -o {meta_file}"
            if self.compile_commands_path:
                cmd += f" -p {self.compile_commands_path.resolve().parent}"

            logger.debug(f"Running: {cmd}")

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"Failed to process {source_file}: {result.stderr}")
                return Meta({})

            if meta_file.exists():
                return Meta.load(meta_file)

            return Meta({})
