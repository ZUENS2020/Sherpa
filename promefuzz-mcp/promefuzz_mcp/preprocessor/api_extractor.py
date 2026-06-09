"""
Preprocessor module - API extraction.
"""

from pathlib import Path
from typing import Optional, Tuple, List
from loguru import logger
import json
import os

from .ast import Meta


# Path fragments that mark a header as NON-public (internal impl, language
# bindings, tests, examples). Matched case-insensitively against the POSIX
# path of each header file. Overridable via SHERPA_VULN_PUBLIC_EXCLUDE_DIRS
# (comma-separated, replaces defaults).
_DEFAULT_INTERNAL_HEADER_MARKERS = (
    "/lib/src/",
    "/src/internal/",
    "/internal/",
    "/private/",
    "/binding_",
    "/bindings/",
    "/binding/",
    "/test/",
    "/tests/",
    "/example/",
    "/examples/",
    "/contrib/",
    "/third_party/",
    "/3rdparty/",
    "/vendor/",
)

# Path fragments that POSITIVELY mark a header as public. A header under any
# of these (and not under an internal marker) is considered public.
# Overridable via SHERPA_VULN_PUBLIC_HEADER_DIRS (comma-separated).
_DEFAULT_PUBLIC_HEADER_MARKERS = (
    "/include/",
)


def _env_markers(env_name: str, default: tuple) -> tuple:
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        return default
    parts = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    return parts or default


def _classify_header_public(header_path: Path, repo_root: Optional[Path]) -> bool:
    """Heuristically decide whether a header is part of the public API surface.

    Rules (first match wins):
      1. Under an internal/binding/test marker dir  -> non-public
      2. Under a public include/ dir                -> public
      3. Top-level header directly in repo root     -> public
      4. Otherwise                                  -> non-public (conservative)
    """
    internal_markers = _env_markers(
        "SHERPA_VULN_PUBLIC_EXCLUDE_DIRS", _DEFAULT_INTERNAL_HEADER_MARKERS
    )
    public_markers = _env_markers(
        "SHERPA_VULN_PUBLIC_HEADER_DIRS", _DEFAULT_PUBLIC_HEADER_MARKERS
    )
    # Classify by the header's location *within the repo*, not its absolute
    # filesystem path — otherwise prefixes like macOS's /private/var tmp dirs
    # would spuriously match internal markers.
    rel_posix = header_path.as_posix().lower()
    is_top_level = False
    if repo_root is not None:
        try:
            rel = header_path.resolve().relative_to(repo_root.resolve())
            rel_posix = rel.as_posix().lower()
            is_top_level = len(rel.parts) == 1
        except Exception:
            rel_posix = header_path.as_posix().lower()
    # Sentinel-pad so markers like "/include/" match a leading segment too.
    probe = "/" + rel_posix.lstrip("/")
    for marker in internal_markers:
        if marker in probe:
            return False
    for marker in public_markers:
        if marker in probe:
            return True
    # A header sitting directly in the repo root is part of the public surface.
    return is_top_level


class APIFunction:
    """Represents an API function."""

    def __init__(
        self,
        header: str,
        name: str,
        loc: str,
        decl_loc: str,
        is_public: bool = True,
    ):
        self.header = header
        self.name = name
        self.loc = loc
        self.decl_loc = decl_loc
        # Whether the declaring header lives in a public include directory
        # (vs. an internal/binding/test header). Consumers use this to build
        # a trustworthy public-API surface for fuzz-target selection.
        self.is_public = is_public

    def __str__(self):
        return f"{self.name} at {self.loc}"

    def to_dict(self):
        return {
            "header": self.header,
            "name": self.name,
            "loc": self.loc,
            "decl_loc": self.decl_loc,
            "is_public": self.is_public,
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
        repo_root: Optional[Path] = None,
    ):
        self.header_paths = header_paths
        self.meta = meta
        self.exclude_paths = exclude_paths or []
        # Repo root used to classify top-level headers as public. Defaults to
        # the first directory header path when not provided.
        self.repo_root = repo_root or next(
            (p for p in header_paths if p.is_dir()), None
        )
        self.header_files: list[Path] = []
        # Maps str(header_file) -> is_public, populated in _collect_headers.
        self.header_public: dict[str, bool] = {}

        self._collect_headers()

    def _collect_headers(self):
        """Collect header files from paths, classifying each public/internal."""
        suffixes = [".h", ".hpp", ".hxx", ".hh"]

        collected: list[Path] = []
        for header_path in self.header_paths:
            if header_path.is_file():
                collected.append(header_path)
            elif header_path.is_dir():
                for suffix in suffixes:
                    collected.extend(header_path.rglob(f"*{suffix}"))

        # Dedupe while preserving order, and classify each.
        seen: set[str] = set()
        public_count = 0
        for hf in collected:
            key = str(hf)
            if key in seen:
                continue
            seen.add(key)
            self.header_files.append(hf)
            is_public = _classify_header_public(hf, self.repo_root)
            self.header_public[key] = is_public
            if is_public:
                public_count += 1

        logger.info(
            f"Found {len(self.header_files)} header files "
            f"({public_count} public, {len(self.header_files) - public_count} internal)"
        )

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
            if not decl_file:
                decl_file = func_loc.split(":")[0] if func_loc else ""

            for header_file in self.header_files:
                if str(header_file) == decl_file:
                    api_func = APIFunction(
                        header=str(header_file),
                        name=func_obj.get("name", ""),
                        loc=func_loc,
                        decl_loc=decl_loc,
                        is_public=self.header_public.get(str(header_file), True),
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
