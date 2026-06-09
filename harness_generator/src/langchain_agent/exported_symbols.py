"""Extract the set of exported/linkable symbols from built libraries.

This is the *authoritative* public-API oracle: a symbol a fuzz harness can
actually link against must be a defined, externally-visible symbol in one of
the built `.a`/`.so`/`.dylib` artifacts. Header-derived API lists are a useful
pre-build heuristic, but only the symbol table proves linkability.

Used at the target-selection / re-selection stage (post-build). Degrades to an
empty set on any failure so it never blocks the pipeline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from loguru import logger

_LIB_SUFFIXES = {".a", ".so", ".dylib", ".lib"}
# nm type codes that denote a defined, externally-visible symbol. Uppercase =
# global/external. 'U' is undefined (a dependency, not provided here) so it's
# excluded. We keep the common code-bearing/data codes.
_EXPORTED_TYPE_CODES = set("TDBRWVCS")  # text/data/bss/rodata/weak/common/...


def _find_libraries(roots: Iterable[Path], *, limit: int = 200) -> list[Path]:
    libs: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root or not root.exists():
            continue
        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in _LIB_SUFFIXES:
                    continue
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                libs.append(p)
                if len(libs) >= limit:
                    return libs
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"[exported_symbols] scan failed under {root}: {exc}")
    return libs


def _nm_symbols(lib: Path, *, timeout: float = 60.0) -> set[str]:
    """Return defined+exported symbol names from a single library via nm.

    Tries `nm --defined-only` first (works for static archives and most ELF);
    falls back to `nm -D` (dynamic symbols) for shared objects.
    """
    nm = shutil.which("nm")
    if not nm:
        logger.debug("[exported_symbols] nm not on PATH; skipping")
        return set()

    out_names: set[str] = set()
    for args in (["--defined-only", str(lib)], ["-D", "--defined-only", str(lib)]):
        try:
            proc = subprocess.run(
                [nm, *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except Exception as exc:
            logger.debug(f"[exported_symbols] nm {args} on {lib.name} failed: {exc}")
            continue
        if proc.returncode != 0 and not proc.stdout:
            continue
        for line in proc.stdout.splitlines():
            # Formats:
            #   "0000000000000000 T symbol_name"  (address type name)
            #   "                 T symbol_name"   (no address)
            #   "symbol_name.o:"                   (archive member header)
            parts = line.split()
            if len(parts) < 2:
                continue
            type_code = parts[-2] if len(parts) >= 3 else parts[0]
            name = parts[-1]
            # When address present: parts = [addr, type, name] -> type=parts[1]
            if len(parts) >= 3:
                type_code = parts[1]
            else:
                # parts = [type, name]
                type_code = parts[0]
            if not type_code or len(type_code) != 1:
                continue
            if type_code.upper() not in _EXPORTED_TYPE_CODES:
                continue
            # Only externally-visible (uppercase) symbols are linkable from
            # another translation unit.
            if not type_code.isupper():
                continue
            if name:
                out_names.add(name)
        if out_names:
            break
    return out_names


def extract_exported_symbols(
    repo_root: Path,
    *,
    extra_roots: Iterable[Path] | None = None,
    output_path: Path | None = None,
) -> set[str]:
    """Collect exported symbols across built libraries under the repo.

    Args:
        repo_root: repository root; scans it and repo_root/build.
        extra_roots: additional directories to scan for libraries.
        output_path: when given, persist the sorted symbol list as JSON.

    Returns:
        Set of exported symbol names (empty on failure / no libs / no nm).
    """
    roots: list[Path] = [repo_root, repo_root / "build"]
    if extra_roots:
        roots.extend(Path(r) for r in extra_roots)

    libs = _find_libraries(roots)
    if not libs:
        logger.debug("[exported_symbols] no built libraries found; empty oracle")
        if output_path is not None:
            _write_json(output_path, [], degraded_reason="no_libraries_found")
        return set()

    symbols: set[str] = set()
    for lib in libs:
        symbols |= _nm_symbols(lib)

    logger.info(
        f"[exported_symbols] {len(symbols)} exported symbols from {len(libs)} libraries"
    )
    if output_path is not None:
        _write_json(output_path, sorted(symbols), libraries=[str(p) for p in libs])
    return symbols


def load_exported_symbols(path: Path) -> set[str]:
    """Load a previously written exported_symbols.json. Empty set on failure."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return set()
    syms = doc.get("symbols") if isinstance(doc, dict) else doc
    if not isinstance(syms, list):
        return set()
    return {str(s) for s in syms if str(s).strip()}


def _write_json(
    output_path: Path,
    symbols: list[str],
    *,
    libraries: list[str] | None = None,
    degraded_reason: str = "",
) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "count": len(symbols),
                    "symbols": symbols,
                    "libraries": libraries or [],
                    "degraded_reason": degraded_reason,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"[exported_symbols] failed to write {output_path}: {exc}")
