"""
Preprocessor module - API extraction.
"""

from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger
import json

from .ast import Meta


class APIFunction:
    """Represents an API function."""

    def __init__(self, header: str, name: str, loc: str, decl_loc: str):
        self.header = header
        self.name = name
        self.loc = loc
        self.decl_loc = decl_loc

    def __str__(self):
        return f"{self.name} at {self.loc}"

    def to_dict(self):
        return {
            "header": self.header,
            "name": self.name,
            "loc": self.loc,
            "decl_loc": self.decl_loc,
        }


class APICollection:
    """Collection of API functions."""

    def __init__(self, functions: List[APIFunction] = None):
        self.funcs = functions or []

    @property
    def count(self) -> int:
        return len(self.funcs)

    def get_by_name(self, name: str) -> List[APIFunction]:
        return [f for f in self.funcs if f.name == name]

    def save(self, output_path: Path):
        """Save API collection to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "count": self.count,
                "functions": [func.to_dict() for func in self.funcs]
            }, f, indent=2)
        logger.info(f"Saved {self.count} API functions to {output_path}")


class APIExtractor:
    """Extract API functions from header files."""

    def __init__(
        self,
        header_paths: list[Path],
        meta: Meta,
        exclude_paths: list[Path] = None,
    ):
        self.header_paths = header_paths
        self.meta = meta
        self.exclude_paths = exclude_paths or []
        self.header_files: list[Path] = []

        self._collect_headers()

    def _collect_headers(self):
        """Collect header files from paths."""
        suffixes = [".h", ".hpp", ".hxx", ".hh"]

        for header_path in self.header_paths:
            if header_path.is_file():
                self.header_files.append(header_path)
            elif header_path.is_dir():
                for suffix in suffixes:
                    self.header_files.extend(header_path.rglob(f"*{suffix}"))

        logger.info(f"Found {len(self.header_files)} header files")

    def extract(self, output_path: Optional[Path] = None) -> Tuple[APICollection, Optional[Path]]:
        """
        Extract API functions from headers.

        Args:
            output_path: Optional output path for API JSON file

        Returns:
            Tuple of (APICollection, output file path or None)
        """
        api_functions = []

        functions = self.meta.meta.get("functions", {})

        for func_loc, func_obj in functions.items():
            decl_loc = func_obj.get("declLoc", "")
            decl_file = decl_loc.split(":")[0] if decl_loc else ""

            for header_file in self.header_files:
                if str(header_file) == decl_file:
                    api_func = APIFunction(
                        header=str(header_file),
                        name=func_obj.get("name", ""),
                        loc=func_loc,
                        decl_loc=decl_loc,
                    )
                    api_functions.append(api_func)
                    break

        api_collection = APICollection(api_functions)
        logger.info(f"Extracted {len(api_functions)} API functions")

        # Persist to file if output_path is provided
        if output_path:
            api_collection.save(output_path)
            return api_collection, output_path

        return api_collection, None
