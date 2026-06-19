"""Lightweight extraction of documented API preconditions ("contracts").

Some crashes a fuzzer finds are not vulnerabilities: the API documentation
explicitly puts a precondition on the *caller* (e.g. "input must be
NUL-terminated", "must not be NULL", "call X_init first"), the implementation
trusts it and does not defensively check, and the crash only happens because the
harness fed *out-of-contract* input. Those are harness/API-misuse bugs, not
library vulnerabilities.

This module extracts documented preconditions from source doc-comments so crash
triage can tell genuine, in-contract crashes apart from out-of-contract ones.
It is intentionally heuristic and dependency-free (no LLM): it surfaces the
documented contract as evidence; the triage stage decides.
"""

from __future__ import annotations

import re
from pathlib import Path

# Precondition categories keyed to doc-comment phrasing. Kept conservative — a
# match means "the docs document this precondition", which triage weighs.
_PRECONDITION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("nul_terminated", re.compile(
        r"\b(nul|null|zero)[-\s]?terminated\b|terminating\s+(nul|null|'?\\0'?|byte)|must\s+be\s+terminated",
        re.I)),
    ("nonnull", re.compile(
        r"must\s+not\s+be\s+null|must\s+be\s+non[-\s]?null|\bnon[-\s]?null\b"
        r"|must\s+point\s+to|must\s+be\s+a\s+valid\s+pointer",
        re.I)),
    ("length_bound", re.compile(
        r"\b(len|length|size|count|n)\b[^.\n]{0,48}\b(must|should|at\s+least|no\s+more\s+than|not\s+exceed|cannot\s+exceed|<=|>=|greater\s+than|less\s+than)\b"
        r"|buffer\s+(of|with)\s+(at\s+least|size)|at\s+least\s+\d+\s+byte"
        # imperative caller obligations on output/buffer size, e.g. parson.h:85
        # "Make sure it can't serialize to a string longer than PARSON_NUM_BUF_SIZE"
        r"|\b(longer|shorter|larger|bigger|smaller)\s+than\b"
        r"|make\s+sure[^.\n]{0,80}\b(exceed|longer|larger|bigger|smaller|fit|within|less|more|than)\b"
        r"|\b(must|should|do\s+not|don't|never)[^.\n]{0,48}\b(exceed|overflow|fit\s+in|fit\s+within)\b",
        re.I)),
    ("must_initialize", re.compile(
        r"must\s+(be\s+)?initiali[sz]ed|must\s+call\s+[A-Za-z_][A-Za-z0-9_]*\s*(\(\))?\s*(first|before)"
        r"|initiali[sz]e[^.\n]{0,40}\bbefore\b|call\s+[A-Za-z_][A-Za-z0-9_]*_init",
        re.I)),
    ("ownership", re.compile(
        r"caller\s+(must|is\s+responsible)[^.\n]{0,40}\bfree\b|caller\s+owns|must\s+be\s+freed\s+by\s+the\s+caller"
        r"|free[^.\n]{0,20}with\s+[A-Za-z_][A-Za-z0-9_]*",
        re.I)),
    ("valid_input", re.compile(
        r"must\s+be\s+(valid|well[-\s]?formed)|valid\s+utf-?8|assumes?\s+(valid|well[-\s]?formed)",
        re.I)),
]

# A C/C++ stack frame line, e.g. "#3 0x55.. in json_value_free_ex json.c:1039:13"
_FRAME_RE = re.compile(r"#\d+\s+0x[0-9a-fA-F]+\s+in\s+([A-Za-z_][A-Za-z0-9_]*)")
_C_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}


def extract_documented_preconditions(source: str, func: str) -> dict:
    """Return documented preconditions for ``func`` found in ``source``.

    Result: ``{"function", "preconditions": [tags], "doc": <snippet>}``.
    ``preconditions`` is empty when no definition/declaration is found or the
    doc-comment documents nothing relevant.
    """
    out = {"function": func, "preconditions": [], "doc": ""}
    if not source or not func:
        return out
    doc = _doc_comment_above(source, func)
    if not doc:
        return out
    tags = [name for name, pat in _PRECONDITION_PATTERNS if pat.search(doc)]
    out["preconditions"] = tags
    out["doc"] = doc.strip()[:600]
    return out


def _doc_comment_above(source: str, func: str) -> str:
    """Find a definition/declaration of ``func`` and return the contiguous
    comment block immediately above it (``/** */``, ``/* */`` or ``//`` runs)."""
    # Match a function definition/declaration: `<ret> func(` at line start-ish.
    pat = re.compile(r"^[^\n]*\b" + re.escape(func) + r"\s*\(", re.M)
    best = ""
    for m in pat.finditer(source):
        line_start = source.rfind("\n", 0, m.start()) + 1
        above = source[:line_start].rstrip()
        doc = _trailing_comment_block(above)
        # Prefer the definition with the richest doc-comment.
        if len(doc) > len(best):
            best = doc
    return best


def _trailing_comment_block(text: str) -> str:
    """Extract the comment block at the very end of ``text`` (i.e. immediately
    preceding the function), if any."""
    stripped = text.rstrip()
    # Block comment /* ... */ ending at the tail.
    if stripped.endswith("*/"):
        start = stripped.rfind("/*")
        if start != -1:
            return stripped[start:]
    # Run of consecutive // lines at the tail.
    lines = stripped.splitlines()
    run: list[str] = []
    for ln in reversed(lines):
        if ln.lstrip().startswith("//"):
            run.append(ln)
        else:
            break
    if run:
        return "\n".join(reversed(run))
    return ""


def crash_frame_functions(crash_text: str, limit: int = 8) -> list[str]:
    """Distinct function names from sanitizer stack frames, top-first, skipping
    sanitizer/libfuzzer internals and the harness entrypoint."""
    skip = {"main", "LLVMFuzzerTestOneInput"}
    seen: list[str] = []
    for m in _FRAME_RE.finditer(crash_text or ""):
        fn = m.group(1)
        if fn in skip or fn.startswith(("__asan", "__ubsan", "__sanitizer", "__interceptor", "fuzzer")):
            continue
        if fn not in seen:
            seen.append(fn)
        if len(seen) >= limit:
            break
    return seen


# Identifiers that look like function calls but are language/libc noise — never
# library API whose contract we care about.
_CALL_SKIP = {
    "if", "for", "while", "switch", "return", "sizeof", "do", "else",
    "malloc", "calloc", "realloc", "free", "memcpy", "memmove", "memset",
    "strlen", "strcmp", "strncmp", "strcpy", "strncpy", "strdup", "memcmp",
    "printf", "fprintf", "sprintf", "snprintf", "fwrite", "fread", "fopen",
    "fclose", "exit", "abort", "assert", "va_start", "va_end",
    "LLVMFuzzerTestOneInput", "LLVMFuzzerInitialize", "main",
}
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def harness_called_functions(root, fuzzer_name: str, *, limit: int = 12) -> list[str]:
    """Best-effort: locate the harness source for ``fuzzer_name`` and return the
    library API functions it calls, top-first. These may include *stored-config*
    setters (e.g. ``json_set_float_serialization_format``) whose documented
    precondition governs a crash that detonates later, deep in an unrelated call
    that the API setter never appears on. Such functions are invisible to
    crash-stack-only extraction, so we surface them explicitly. Never raises."""
    try:
        if not fuzzer_name:
            return []
        root = Path(root)
        stem = fuzzer_name.strip()
        # last_fuzzer is usually "<name>_fuzz"; the source is typically "<name>.c".
        cands = {stem, stem[:-5] if stem.endswith("_fuzz") else stem}
        src = ""
        for p in root.rglob("*.c"):
            if p.stem in cands and p.is_file():
                try:
                    src = p.read_text(encoding="utf-8", errors="replace")
                    break
                except Exception:
                    continue
        if not src:
            return []
        out: list[str] = []
        for m in _CALL_RE.finditer(src):
            fn = m.group(1)
            if fn in _CALL_SKIP or fn in out:
                continue
            out.append(fn)
            if len(out) >= limit:
                break
        return out
    except Exception:
        return []


def build_contract_triage_context(
    repo_root,
    crash_text: str,
    *,
    extra_funcs: list[str] | None = None,
    fuzzer_name: str | None = None,
    max_files: int = 400,
    max_funcs: int = 12,
) -> str:
    """Build a markdown context block documenting preconditions of the functions
    on the crash stack, for the crash-triage prompt. Returns "" when nothing
    relevant is found. Best-effort and bounded; never raises."""
    try:
        root = Path(repo_root)
        funcs = crash_frame_functions(crash_text)
        # Stored-config / deferred-trigger crashes: the precondition lives on an
        # API the harness called earlier (a setter), which is NOT on the crash
        # backtrace. Add the harness's called API functions so those contracts
        # are still checked.
        for f in harness_called_functions(root, fuzzer_name or ""):
            if f and f not in funcs:
                funcs.append(f)
        for f in (extra_funcs or []):
            if f and f not in funcs:
                funcs.append(f)
        funcs = funcs[:max_funcs]
        if not funcs:
            return ""

        # Gather a bounded set of source files (skip build/test/vendor dirs).
        sources: list[str] = []
        skip_dirs = {".git", "build", "fuzz", "out", "vcpkg", "vcpkg_installed",
                     "node_modules", "third_party", "tests", "test", "examples"}
        count = 0
        for p in root.rglob("*"):
            if count >= max_files:
                break
            if p.suffix.lower() not in _C_SUFFIXES or not p.is_file():
                continue
            if any(part in skip_dirs for part in p.relative_to(root).parts[:-1]):
                continue
            try:
                sources.append(p.read_text(encoding="utf-8", errors="replace"))
                count += 1
            except Exception:
                continue
        if not sources:
            return ""

        findings: list[dict] = []
        for fn in funcs:
            for src in sources:
                if fn not in src:
                    continue
                c = extract_documented_preconditions(src, fn)
                if c["preconditions"]:
                    findings.append(c)
                    break  # first file documenting this function wins

        if not findings:
            return ""

        lines = [
            "=== api_contract (documented preconditions of crashing functions) ===",
            "If the crash is only reachable because the harness fed input that "
            "VIOLATES one of these documented preconditions, the crash is an "
            "out-of-contract / harness misuse bug — classify it as `harness_bug`, "
            "NOT `upstream_bug` (it is not a library vulnerability).",
        ]
        for c in findings:
            lines.append(f"- {c['function']}: preconditions={c['preconditions']}")
            if c["doc"]:
                snippet = " ".join(c["doc"].split())[:240]
                lines.append(f"    doc: {snippet}")
        return "\n".join(lines)
    except Exception:
        return ""
