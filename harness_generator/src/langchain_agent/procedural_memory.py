"""Cross-stage / cross-job procedural memory.

A Reflexion-style lesson store that lets the pipeline stop repeating the same
mistakes across jobs (e.g. declaring vcpkg ports for a self-contained Makefile
library, or selecting non-public/binding targets). It generalizes the proven
per-job ``constraint_memory -> fix_hint`` loop to a cross-job, all-stage store.

Design: docs/CROSS_STAGE_MEMORY_PLAN.md

Scope of this module (Phase 1-2): a file-backed store (JSON) on the shared,
per-environment output volume, plus failure classification, hygiene (confidence
/ occurrence / decay / scope), templated lessons for the two known error
classes, and rendering for prompt injection. The backend is abstracted so a
Postgres ``BaseStore`` can replace the file later without touching callers.

Everything is gated by ``SHERPA_PROCEDURAL_MEMORY`` (default off) and degrades
to a no-op on any error so it can never break a job.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

# Enumerated, coarse error categories. Keeping this closed keeps retrieval
# precise and prompts small. Extend deliberately.
ERROR_CLASSES = (
    "vcpkg_overdeclare",
    "non_public_api_selection",
    "missing_owning_source",
    "coverage_trivial_context",
    "harness_bad_free",
)


# --------------------------------------------------------------------------- #
# config (SHERPA_* convention; all optional, safe defaults)
# --------------------------------------------------------------------------- #
def _truthy(raw: str | None, default: bool = False) -> bool:
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def memory_enabled() -> bool:
    """Master switch. Default OFF until validated end-to-end (Phase 3)."""
    return _truthy(os.environ.get("SHERPA_PROCEDURAL_MEMORY"), False)


def memory_readonly() -> bool:
    """When set, retrieval/injection is allowed but writes are suppressed
    (safe canary). Independent of the master switch's write side."""
    return _truthy(os.environ.get("SHERPA_PROCEDURAL_MEMORY_READONLY"), False)


def _int_env(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(int(str(os.environ.get(name) or default).strip()), hi))
    except Exception:
        return default


def top_k() -> int:
    return _int_env("SHERPA_PROCEDURAL_MEMORY_TOPK", 3, 1, 10)


def min_occurrence() -> int:
    return _int_env("SHERPA_PROCEDURAL_MEMORY_MIN_OCCURRENCE", 2, 1, 10)


def decay_days() -> int:
    return _int_env("SHERPA_PROCEDURAL_MEMORY_DECAY_DAYS", 90, 1, 3650)


def store_path() -> Path:
    """Cross-job store location. Defaults under the shared, per-environment
    output volume so lessons persist across jobs but stay env-isolated."""
    explicit = os.environ.get("SHERPA_PROCEDURAL_MEMORY_PATH")
    if explicit:
        return Path(explicit).expanduser()
    base = Path(os.environ.get("SHERPA_OUTPUT_DIR", "/shared/output")).expanduser()
    return base / "_memory" / "procedural_memory.json"


# --------------------------------------------------------------------------- #
# storage (file backend; atomic write; never raises)
# --------------------------------------------------------------------------- #
def _empty_store() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "updated_at": 0, "lessons": {}}


def load_store(path: Path | None = None) -> dict[str, Any]:
    p = path or store_path()
    try:
        if not p.is_file():
            return _empty_store()
        raw = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(raw, dict):
            return _empty_store()
        lessons = raw.get("lessons")
        if not isinstance(lessons, dict):
            lessons = {}
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": int(raw.get("updated_at") or 0),
            "lessons": lessons,
        }
    except Exception:
        return _empty_store()


def _atomic_write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".pm-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(doc, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# lesson identity + hygiene
# --------------------------------------------------------------------------- #
def lesson_id(stage: str, error_class: str, scope: str) -> str:
    return f"{stage}/{error_class}/{scope}".strip("/")


def _now() -> int:
    return int(time.time())


def _is_active(entry: dict[str, Any]) -> bool:
    """A lesson is injectable once it has recurred enough and has not decayed."""
    if int(entry.get("occurrence_count") or 0) < min_occurrence():
        return False
    last = int(entry.get("last_seen") or 0)
    if last and (_now() - last) > decay_days() * 86400:
        return False
    return float(entry.get("confidence") or 0.0) > 0.0


# --------------------------------------------------------------------------- #
# write path (reflection)
# --------------------------------------------------------------------------- #
def record_lesson(
    *,
    stage: str,
    error_class: str,
    scope: str,
    signature: str,
    lesson: str,
    evidence: Iterable[str] | None = None,
    job_id: str = "",
    confidence: float = 0.7,
    path: Path | None = None,
) -> dict[str, Any] | None:
    """Record (or reinforce) a lesson. No-op when memory is disabled or in
    read-only mode. Returns the stored entry, or None when skipped."""
    if not memory_enabled() or memory_readonly():
        return None
    if not stage or error_class not in ERROR_CLASSES or not lesson:
        return None
    p = path or store_path()
    doc = load_store(p)
    lessons = dict(doc.get("lessons") or {})
    key = lesson_id(stage, error_class, scope or "global")
    prev = dict(lessons.get(key) or {})
    now = _now()
    occ = int(prev.get("occurrence_count") or 0) + 1
    # confidence grows with corroboration, capped
    conf = min(0.99, max(float(prev.get("confidence") or 0.0), float(confidence)) + 0.05 * (occ - 1))
    src_jobs = list(prev.get("source_jobs") or [])
    if job_id and job_id not in src_jobs:
        src_jobs.append(job_id)
    entry = {
        "lesson_id": key,
        "stage": stage,
        "error_class": error_class,
        "scope": scope or "global",
        "signature": str(signature or "")[:512],
        "lesson": str(lesson)[:1024],
        "evidence": [str(x)[:300] for x in list(evidence or []) if str(x).strip()][:8],
        "confidence": round(conf, 3),
        "occurrence_count": occ,
        "first_seen": int(prev.get("first_seen") or now),
        "last_seen": now,
        "decay_after_days": decay_days(),
        "source_jobs": src_jobs[-20:],
        "schema_version": SCHEMA_VERSION,
    }
    lessons[key] = entry
    _atomic_write(p, {"schema_version": SCHEMA_VERSION, "updated_at": now, "lessons": lessons})
    return entry


def note_success(stage: str, scope: str, *, path: Path | None = None) -> None:
    """Contradiction handling: a success in a (stage, scope) decays the
    confidence of lessons there, so stale advice fades (Mem0-style UPDATE)."""
    if not memory_enabled() or memory_readonly():
        return
    p = path or store_path()
    doc = load_store(p)
    lessons = dict(doc.get("lessons") or {})
    changed = False
    for key, entry in lessons.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("stage") == stage and entry.get("scope") == (scope or "global"):
            entry["confidence"] = round(max(0.0, float(entry.get("confidence") or 0.0) - 0.2), 3)
            changed = True
    if changed:
        _atomic_write(p, {"schema_version": SCHEMA_VERSION, "updated_at": _now(), "lessons": lessons})


# --------------------------------------------------------------------------- #
# read path (retrieval + rendering)
# --------------------------------------------------------------------------- #
def retrieve(
    *,
    stage: str,
    library_class: str = "",
    path: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Top-K active lessons for a stage, preferring matching library scope,
    ranked by confidence * recency. Empty unless memory is enabled."""
    if not memory_enabled():
        return []
    doc = load_store(path or store_path())
    lessons = [e for e in (doc.get("lessons") or {}).values() if isinstance(e, dict)]
    scope_match = f"library_class:{library_class}" if library_class else ""

    def relevant(e: dict[str, Any]) -> bool:
        if e.get("stage") != stage or not _is_active(e):
            return False
        sc = str(e.get("scope") or "global")
        return sc == "global" or sc == scope_match or not library_class

    def rank(e: dict[str, Any]) -> float:
        age_days = max(0.0, (_now() - int(e.get("last_seen") or 0)) / 86400.0)
        recency = max(0.1, 1.0 - age_days / float(e.get("decay_after_days") or decay_days()))
        scope_boost = 1.25 if str(e.get("scope") or "") == scope_match else 1.0
        return float(e.get("confidence") or 0.0) * recency * scope_boost

    hits = sorted((e for e in lessons if relevant(e)), key=rank, reverse=True)
    return hits[: (limit or top_k())]


def render_lessons_block(lessons: list[dict[str, Any]]) -> str:
    """Render a compact, prompt-injectable 'known pitfalls' block."""
    if not lessons:
        return ""
    lines = ["## Known pitfalls (learned from prior runs — avoid repeating)"]
    for e in lessons:
        conf = float(e.get("confidence") or 0.0)
        lines.append(f"- [{e.get('error_class')}] {e.get('lesson')} (confidence {conf:.2f})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# failure classification (templated lessons for known classes)
# --------------------------------------------------------------------------- #
def classify_stage_failure(
    *,
    stage: str,
    error_code: str = "",
    error_kind: str = "",
    diagnostics: str = "",
    system_packages_nonempty: bool = False,
    api_surface_exception_used: bool = False,
    library_class: str = "",
) -> dict[str, Any] | None:
    """Map a stage failure to a templated lesson when it matches a known class.

    Returns a dict ready for ``record_lesson(**result)`` (minus job_id), or None
    when the failure does not match a known, deterministic class (those are left
    for the optional LLM-reflection path in a later phase).
    """
    text = " ".join(str(x or "") for x in (error_code, error_kind, diagnostics)).lower()
    scope = f"library_class:{library_class}" if library_class else "global"

    # vcpkg over-declaration for a self-contained build
    if system_packages_nonempty and (
        "vcpkg unavailable" in text
        or "missing vcpkg toolchain" in text
        or ("vcpkg" in text and ("synthesize failed" in text or "build" in text))
    ):
        return {
            "stage": stage,
            "error_class": "vcpkg_overdeclare",
            "scope": scope,
            "signature": "stage failed; vcpkg unavailable/missing toolchain; system_packages.txt non-empty",
            "lesson": (
                "Do NOT write fuzz/system_packages.txt for self-contained make/single-file "
                "libraries. Declare vcpkg ports only when the build fails on a concrete "
                "missing EXTERNAL library (cmake/pkg-config 'not found'); never declare ports "
                "for code the repo vendors/builds itself."
            ),
            "evidence": [f"error_code={error_code}", f"error_kind={error_kind}"],
            "confidence": 0.85,
        }

    # selecting a non-public / binding target that cannot link
    if api_surface_exception_used or "non_public_api_usage" in text:
        return {
            "stage": stage,
            "error_class": "non_public_api_selection",
            "scope": scope,
            "signature": "selected target is internal/binding; non_public_api_usage",
            "lesson": (
                "Prefer public, linkable entrypoints. Do not select internal/static or "
                "language-binding symbols (e.g. *_wasm/_napi/_jni); if a vulnerable internal "
                "sink is the goal, drive it through its nearest public caller and record the "
                "sink in attack_hint.key_code_path."
            ),
            "evidence": [f"error_code={error_code}"],
            "confidence": 0.8,
        }

    return None
