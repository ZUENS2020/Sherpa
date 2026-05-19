"""
Preprocessor module - AST preprocessing.
"""

from pathlib import Path
from typing import Any, Optional, Tuple
from loguru import logger
import subprocess
import json
import tempfile
import shutil

from ..build import BinaryBuilder

_SIMD_SUFFIXES = {
    "_neon", "_sse", "_sse2", "_sse3", "_sse4", "_sse41", "_sse42",
    "_ssse3", "_msa", "_avx", "_avx2", "_avx512", "_altivec", "_vsx",
}

_BUILD_BLACKLIST_PARTS = frozenset({
    "build", "build64", "_build",
    "CMakeFiles", "_deps", "_install", "out", "dist",
    "node_modules", "vendor", "third_party", "thirdparty",
    ".git", ".svn", ".hg", "__pycache__",
})

_HEADER_SUFFIXES = (".h", ".hpp", ".hh", ".hxx", ".inc")


def _is_simd_file(path: Path) -> bool:
    stem = path.stem.lower()
    return any(stem.endswith(s) for s in _SIMD_SUFFIXES)


def _is_build_artifact(path: Path) -> bool:
    for part in path.parts:
        if part in _BUILD_BLACKLIST_PARTS:
            return True
        if part.startswith(("build-", "build_", "cmake-build-")):
            return True
    return False


def _dir_has_headers(d: Path, *, max_depth: int = 2) -> bool:
    """True iff `d` contains at least one header within max_depth levels."""
    if not d.is_dir() or _is_build_artifact(d):
        return False
    try:
        for entry in d.iterdir():
            if _is_build_artifact(entry):
                continue
            if entry.is_file() and entry.suffix.lower() in _HEADER_SUFFIXES:
                return True
            if max_depth > 0 and entry.is_dir():
                if _dir_has_headers(entry, max_depth=max_depth - 1):
                    return True
    except (OSError, PermissionError):
        return False
    return False


def _scan_include_dirs(repo_root: Path) -> list[str]:
    """Return -I candidates ordered by priority.

    Tier 1 — canonical names (include/src/source/lib); OpenSSL hits here.
    Tier 2 — repo_root itself; covers cJSON-style flat layouts.
    Tier 3 — first-level subdirs containing headers; covers wolfssl/wolfssl/.
    Build-artifact dirs are excluded everywhere. Capped at 16 entries.
    """
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _push(p: Path) -> None:
        if not p.is_dir() or _is_build_artifact(p):
            return
        try:
            rp = p.resolve()
        except (OSError, RuntimeError):
            return
        if rp in seen:
            return
        seen.add(rp)
        candidates.append(p)

    for name in ("include", "src", "source", "lib"):
        _push(repo_root / name)
    _push(repo_root)
    try:
        for entry in sorted(repo_root.iterdir()):
            if entry.is_dir() and _dir_has_headers(entry, max_depth=2):
                _push(entry)
    except (OSError, PermissionError):
        pass

    return [str(p) for p in candidates[:16]]


class Meta:
    """Metadata container."""

    def __init__(self, meta_dict: Optional[dict[str, Any]]):
        self.meta: dict[str, Any] = meta_dict if isinstance(meta_dict, dict) else {}

    @classmethod
    def load(cls, path: Path) -> "Meta":
        """Load meta from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load meta json {path}: {e}")
            return cls({})
        if not isinstance(data, dict):
            logger.warning(f"Invalid meta json payload (non-dict) in {path}: {type(data).__name__}")
            return cls({})
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
        self.invalid_meta_files: list[str] = []
        self._extra_include_args: str = ""

        # Get binary builder
        self.builder = BinaryBuilder()

        # Collect source files
        self._collect_source_files()

        # Precompute -I args once when no compile_commands.json is available.
        if not self.compile_commands_path:
            repo_root = self._infer_repo_root()
            if repo_root is not None:
                self._extra_include_args = "".join(
                    f" --extra-arg=-I{inc}" for inc in _scan_include_dirs(repo_root)
                )

    def _infer_repo_root(self) -> Optional[Path]:
        for sp in self.source_paths:
            if sp.is_dir():
                return sp.parent if sp.name in ("src", "source", "lib") else sp
            if sp.is_file():
                return sp.parent.parent if sp.parent.name in ("src", "source", "lib") else sp.parent
        return None

    def _collect_source_files(self):
        """Collect all source files from source paths."""
        suffixes = [".c", ".cpp", ".cc", ".cxx", ".c++"]

        for source_path in self.source_paths:
            if source_path.is_file():
                if not _is_simd_file(source_path):
                    self.source_files.append(source_path)
            elif source_path.is_dir():
                for suffix in suffixes:
                    for f in source_path.rglob(f"*{suffix}"):
                        if _is_simd_file(f):
                            continue
                        if _is_build_artifact(f):
                            continue
                        if any(part in ("test", "tests", "demo", "demos", "examples", "example")
                               for part in f.parts):
                            continue
                        self.source_files.append(f)

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

        all_meta: dict[str, Any] = {}
        self.invalid_meta_files = []
        total = len(self.source_files)

        for idx, source_file in enumerate(self.source_files):
            meta = self._process_file(source_file, preprocessor_bin)
            meta_payload = getattr(meta, "meta", None)
            if not isinstance(meta_payload, dict):
                logger.warning(f"Skipping invalid meta payload from {source_file}: {type(meta_payload).__name__}")
                self.invalid_meta_files.append(str(source_file))
                continue
            all_meta.update(meta_payload)
            if (idx + 1) % 100 == 0 or (idx + 1) == total:
                logger.info(f"AST preprocessing progress: {idx + 1}/{total} files ({int((idx + 1) / total * 100)}%)")

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
            else:
                cmd += self._extra_include_args

            logger.debug(f"Running: {cmd}")

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout (30s) processing {source_file}")
                return Meta({})

            if result.returncode != 0:
                logger.warning(f"Failed to process {source_file}: {result.stderr[:200]}")
                return Meta({})

            if meta_file.exists():
                meta = Meta.load(meta_file)
                if not meta.meta:
                    self.invalid_meta_files.append(str(source_file))
                return meta

            return Meta({})
