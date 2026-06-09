"""Public-API surface oracle for vulnerability-directed target selection.

A fuzz harness can only link against *public* / *exported* symbols. vuln-hunt
candidates that name internal helpers, language bindings (``*_wasm``), or
static functions are not reachable from a harness and otherwise only fail at
build time. This module is the single source of truth for deciding whether a
symbol is part of the public surface, and for mapping an internal sink to the
nearest public entry point that reaches it.

Sources (unioned, best-effort, degrade to empty):
  1. companion ``work/api_functions.json`` entries with ``is_public == true``
     (header-derived, available pre-build)
  2. build-stage ``fuzz/exported_symbols.json`` (``nm``-derived, authoritative)

All lookups degrade safely: when no oracle data exists, classification falls
back to name-pattern heuristics so nothing is wrongly rejected.

This module is a leaf: it depends only on the standard library and
``exported_symbols``. It must not import ``workflow_graph`` (no cycles).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _is_internal_api_symbol(api: str) -> bool:
    low = str(api or "").strip().lower()
    if not low:
        return False
    patterns = (
        "::detail::",
        "::detail",
        "::internal::",
        "::internal",
        "_internal",
        "/internal/",
        ".internal.",
        "__",
    )
    return any(p in low for p in patterns)


# --- Public-API surface oracle ------------------------------------------------
# Symbol-name patterns that mark a function as a non-public binding/shim even
# when it isn't listed in (or precedes) the extracted public-API set.
_NON_PUBLIC_API_NAME_PATTERNS = (
    "_wasm",
    "_emscripten",
    "emscripten_",
    "_napi",
    "napi_",
    "_jni",
    "jni_",
    "_pyx",
    "_cffi",
)

# Cache: repo_root str -> frozenset of public/exported symbol names.
_PUBLIC_API_SYMBOL_CACHE: dict[str, frozenset[str]] = {}


def _vuln_public_api_enforce() -> bool:
    raw = (os.environ.get("SHERPA_VULN_PUBLIC_API_ENFORCE") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _companion_work_dir() -> Path | None:
    job_id = str(os.environ.get("SHERPA_JOB_ID") or "").strip()
    if not job_id:
        return None
    base_output = Path(os.environ.get("SHERPA_OUTPUT_DIR", "/shared/output")).expanduser()
    work = base_output / "_k8s_jobs" / job_id / "promefuzz" / "work"
    return work if work.is_dir() else None


def _load_public_api_symbols(repo_root: Path, *, refresh: bool = False) -> frozenset[str]:
    """Build the public/linkable symbol set for a repo.

    Sources, unioned:
      1. companion `work/api_functions.json` entries with `is_public == true`
      2. build-stage `fuzz/exported_symbols.json` (authoritative, post-build)

    Cached per repo_root. Returns empty set when no source is available, in
    which case callers must degrade to name-pattern heuristics (no false
    rejections).
    """
    key = str(repo_root)
    if not refresh and key in _PUBLIC_API_SYMBOL_CACHE:
        return _PUBLIC_API_SYMBOL_CACHE[key]

    names: set[str] = set()

    # 1. header-derived public surface (pre-build heuristic)
    work = _companion_work_dir()
    api_path = (work / "api_functions.json") if work else None
    if api_path and api_path.is_file():
        try:
            doc = json.loads(api_path.read_text(encoding="utf-8", errors="replace"))
            funcs = doc.get("functions") if isinstance(doc, dict) else doc
            for f in funcs or []:
                if not isinstance(f, dict):
                    continue
                # Treat missing is_public as public for backward compatibility
                # with old artifacts that predate the flag.
                if f.get("is_public", True):
                    nm = str(f.get("name") or "").strip()
                    if nm:
                        names.add(nm)
        except Exception:
            pass

    # 2. exported symbol table (authoritative)
    try:
        from .exported_symbols import load_exported_symbols

        exported_path = repo_root / "fuzz" / "exported_symbols.json"
        if exported_path.is_file():
            names |= load_exported_symbols(exported_path)
    except Exception:
        pass

    result = frozenset(names)
    _PUBLIC_API_SYMBOL_CACHE[key] = result
    return result


def _load_api_loc_index(repo_root: Path) -> dict[str, tuple[str, int]]:
    """Map symbol name -> (source_path, line) from companion api_functions.json.

    Prefers `decl_loc` (declaration in header) then `loc`. Used to backfill a
    candidate's missing source location. Empty when unavailable.
    """
    work = _companion_work_dir()
    api_path = (work / "api_functions.json") if work else None
    if not api_path or not api_path.is_file():
        return {}
    index: dict[str, tuple[str, int]] = {}
    try:
        doc = json.loads(api_path.read_text(encoding="utf-8", errors="replace"))
        funcs = doc.get("functions") if isinstance(doc, dict) else doc
        for f in funcs or []:
            if not isinstance(f, dict):
                continue
            nm = str(f.get("name") or "").strip()
            if not nm or nm in index:
                continue
            raw = str(f.get("decl_loc") or f.get("loc") or "").strip()
            if not raw:
                continue
            # Format: "path:line:col" (col optional). Split from the right so
            # Windows-style paths with a drive colon survive.
            parts = raw.rsplit(":", 2)
            path_part = parts[0] if parts else raw
            line_no = 0
            if len(parts) >= 2:
                try:
                    line_no = int(parts[1])
                except Exception:
                    line_no = 0
            if path_part:
                index[nm] = (path_part, line_no)
    except Exception:
        return {}
    return index


def _api_surface_class(api: str, public_set: frozenset[str]) -> str:
    """Classify an API symbol as 'public' | 'internal' | 'unknown'.

    - 'internal' when name matches a binding/shim pattern, the legacy internal
      heuristic, or (when a non-empty public_set exists) is absent from it.
    - 'public' when present in a non-empty public_set.
    - 'unknown' when no oracle is available and no pattern matches (degraded).
    """
    name = str(api or "").strip()
    if not name:
        return "unknown"
    low = name.lower()
    if any(p in low for p in _NON_PUBLIC_API_NAME_PATTERNS):
        return "internal"
    if _is_internal_api_symbol(name):
        return "internal"
    if public_set:
        return "public" if name in public_set else "internal"
    return "unknown"


def _is_non_public_api(api: str, public_set: frozenset[str]) -> bool:
    """True when the symbol is known to be non-public/non-linkable.

    Degrades safely: with no oracle and no matching pattern, returns the legacy
    `_is_internal_api_symbol` result so behaviour is unchanged when enforcement
    has no data to act on.
    """
    if not _vuln_public_api_enforce():
        return _is_internal_api_symbol(api)
    return _api_surface_class(api, public_set) == "internal"


# Cache: repo_root str -> reverse adjacency {callee_name -> set(caller_name)}.
_CALLGRAPH_REVERSE_CACHE: dict[str, dict[str, set[str]]] = {}


def _load_callgraph_reverse(repo_root: Path) -> dict[str, set[str]]:
    """Build a callee -> {callers} index from companion coverage_hints.json.

    Empty dict when no callgraph data is available.
    """
    key = str(repo_root)
    if key in _CALLGRAPH_REVERSE_CACHE:
        return _CALLGRAPH_REVERSE_CACHE[key]
    rev: dict[str, set[str]] = {}
    job_id = str(os.environ.get("SHERPA_JOB_ID") or "").strip()
    if job_id:
        base_output = Path(os.environ.get("SHERPA_OUTPUT_DIR", "/shared/output")).expanduser()
        hints_path = base_output / "_k8s_jobs" / job_id / "promefuzz" / "coverage_hints.json"
        if hints_path.is_file():
            try:
                doc = json.loads(hints_path.read_text(encoding="utf-8", errors="replace"))
                for edge in doc.get("callgraph_summary") or []:
                    if not isinstance(edge, dict):
                        continue
                    caller = str(edge.get("caller") or "").strip()
                    callee = str(edge.get("callee") or "").strip()
                    if caller and callee:
                        rev.setdefault(callee, set()).add(caller)
            except Exception:
                rev = {}
    _CALLGRAPH_REVERSE_CACHE[key] = rev
    return rev


def _nearest_public_entry(
    sink_api: str,
    public_set: frozenset[str],
    reverse_cg: dict[str, set[str]],
    *,
    max_depth: int = 6,
) -> str:
    """Reverse-BFS from an internal sink to the nearest public caller.

    Returns the public entry symbol name, or "" when none is reachable / no
    callgraph. The public_set must be non-empty for a positive result.
    """
    if not sink_api or not public_set or not reverse_cg:
        return ""
    from collections import deque

    seen: set[str] = {sink_api}
    queue: deque[tuple[str, int]] = deque((c, 1) for c in reverse_cg.get(sink_api, set()))
    while queue:
        node, depth = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        if node in public_set and not _is_non_public_api(node, public_set):
            return node
        if depth >= max_depth:
            continue
        for caller in reverse_cg.get(node, set()):
            if caller not in seen:
                queue.append((caller, depth + 1))
    return ""
