from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PROGRESS_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "build",
    "build-work",
    "dist",
    "out",
    "third_party",
    "vendor",
}

_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
_DOC_SUFFIXES = {".md", ".rst", ".txt"}

_SYMBOL_RE = re.compile(
    r"^\s*(?:[A-Za-z_][\w:<>\s\*&,\[\]]+\s+)?([A-Za-z_]\w*)\s*\([^;{}]{0,240}\)\s*;"
)
_FALLBACK_CALL_RE = re.compile(r"\b([A-Za-z_]\w{2,})\s*\(")

_C_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "typedef",
    "define",
    "else",
    "do",
    "case",
    "break",
    "continue",
    "goto",
    "struct",
    "union",
    "enum",
}

_PROMEFUZZ_BUILD_ATTEMPTED = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int | None = None) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    if value < min_value:
        value = min_value
    if max_value is not None and value > max_value:
        value = max_value
    return value


def _dump_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def _extract_repo_root_from_doc(doc: dict[str, Any]) -> str:
    for key in ("repo_root", "workflow_repo_root", "resume_repo_root"):
        value = str(doc.get(key) or "").strip()
        if value:
            return value

    result = doc.get("result")
    if isinstance(result, dict):
        for key in ("repo_root", "workflow_repo_root", "resume_repo_root"):
            value = str(result.get(key) or "").strip()
            if value:
                return value
        stage_results = result.get("stage_results")
        if isinstance(stage_results, list):
            for stage_doc in reversed(stage_results):
                if not isinstance(stage_doc, dict):
                    continue
                value = str(stage_doc.get("repo_root") or "").strip()
                if value:
                    return value
                nested = stage_doc.get("result")
                if isinstance(nested, dict):
                    value = str(nested.get("repo_root") or "").strip()
                    if value:
                        return value
    return ""


def _resolve_repo_root(job_id: str, output_root: Path) -> str:
    hint = str(os.environ.get("SHERPA_PROMEFUZZ_REPO_ROOT_HINT") or "").strip()
    if hint and Path(hint).is_dir():
        return hint

    job_root = output_root / "_k8s_jobs" / job_id
    if not job_root.is_dir():
        return ""
    stage_files = sorted(job_root.glob("stage-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stage_file in stage_files:
        try:
            doc = _load_json(stage_file)
        except Exception:
            continue
        repo_root = _extract_repo_root_from_doc(doc)
        if repo_root and Path(repo_root).is_dir():
            return repo_root
    return ""


@dataclass
class RepoInventory:
    source_files: int
    header_files: int
    doc_files: int
    total_files: int
    total_size_bytes: int


def _collect_repo_inventory(repo_root: Path) -> RepoInventory:
    source_files = 0
    header_files = 0
    doc_files = 0
    total_files = 0
    total_size_bytes = 0
    for base, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in _PROGRESS_SKIP_DIRS]
        for name in files:
            total_files += 1
            p = Path(base) / name
            try:
                total_size_bytes += p.stat().st_size
            except Exception:
                pass
            ext = p.suffix.lower()
            if ext in _SOURCE_SUFFIXES:
                source_files += 1
            if ext in {".h", ".hh", ".hpp", ".hxx"}:
                header_files += 1
            if ext in _DOC_SUFFIXES:
                doc_files += 1
    return RepoInventory(
        source_files=source_files,
        header_files=header_files,
        doc_files=doc_files,
        total_files=total_files,
        total_size_bytes=total_size_bytes,
    )


def _seed_profile_for_symbol(symbol: str) -> str:
    low = symbol.lower()
    if any(k in low for k in ("parse", "token", "format", "scan", "yaml", "json", "xml")):
        return "parser-format"
    if any(k in low for k in ("read", "load", "line", "lex")):
        return "parser-token"
    if any(k in low for k in ("inflate", "deflate", "zip", "gzip", "archive", "tar", "lz", "zstd")):
        return "archive-container"
    if any(k in low for k in ("decode", "decoder", "decompress", "unpack", "deserialize")):
        return "decoder-binary"
    return "unknown"


def _score_symbol(symbol: str, source: str) -> int:
    low = symbol.lower()
    score = 1
    if any(k in low for k in ("parse", "decode", "read", "inflate", "deflate", "decompress", "token")):
        score += 4
    if source.endswith((".h", ".hh", ".hpp", ".hxx")):
        score += 2
    if source.startswith("include/"):
        score += 1
    if low.startswith("_"):
        score -= 3
    return score


def _extract_symbol_candidates(repo_root: Path, max_files: int = 200) -> list[dict[str, Any]]:
    include_dirs: list[Path] = []
    if (repo_root / "include").is_dir():
        include_dirs.append(repo_root / "include")
    include_dirs.append(repo_root)

    header_files: list[Path] = []
    for base in include_dirs:
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".h", ".hh", ".hpp", ".hxx"}:
                continue
            if any(part in _PROGRESS_SKIP_DIRS for part in p.parts):
                continue
            header_files.append(p)
            if len(header_files) >= max_files:
                break
        if len(header_files) >= max_files:
            break

    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for path in header_files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        rel = str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)
        for line in lines[:5000]:
            m = _SYMBOL_RE.match(line)
            if m:
                symbol = str(m.group(1) or "").strip()
            else:
                mc = _FALLBACK_CALL_RE.search(line)
                symbol = str(mc.group(1) or "").strip() if mc else ""
            if not symbol or symbol in _C_KEYWORDS:
                continue
            if symbol[0].isdigit():
                continue
            key = (symbol, rel)
            if key in dedup:
                continue
            score = _score_symbol(symbol, rel)
            dedup[key] = {
                "name": symbol,
                "source": rel,
                "score": score,
                "seed_profile": _seed_profile_for_symbol(symbol),
            }
    rows = sorted(dedup.values(), key=lambda x: (-int(x.get("score", 0)), str(x.get("name", ""))))
    return rows


def _compile_commands_path(repo_root: Path) -> Path | None:
    for rel in (
        "compile_commands.json",
        "build/compile_commands.json",
        "build-work/compile_commands.json",
        "fuzz/build-work/compile_commands.json",
    ):
        p = repo_root / rel
        if p.is_file():
            return p
    return None


def _promefuzz_available() -> bool:
    root = Path(str(os.environ.get("SHERPA_PROMEFUZZ_MCP_ROOT") or "/app/promefuzz-mcp")).expanduser()
    return root.is_dir()


def _run_promefuzz_pipeline(repo_root: Path, companion_root: Path) -> dict[str, Any]:
    global _PROMEFUZZ_BUILD_ATTEMPTED
    result: dict[str, Any] = {
        "enabled": False,
        "ok": False,
        "backend": "fallback-heuristic",
        "error": "",
    }
    if not _promefuzz_available():
        result["error"] = "promefuzz_mcp_root_missing"
        return result

    root = Path(str(os.environ.get("SHERPA_PROMEFUZZ_MCP_ROOT") or "/app/promefuzz-mcp")).expanduser()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    result["enabled"] = True
    source_limit = _env_int("SHERPA_PROMEFUZZ_MAX_SOURCE_FILES", 1200, min_value=50, max_value=20000)
    allow_build = _env_int("SHERPA_PROMEFUZZ_BUILD_BINARIES", 1, min_value=0, max_value=1) == 1

    try:
        from promefuzz_mcp.build import BinaryBuilder
        from promefuzz_mcp.preprocessor.api_extractor import APIExtractor
        from promefuzz_mcp.preprocessor.ast import ASTPreprocessor
    except Exception as e:
        result["error"] = f"promefuzz_import_failed:{e}"
        return result

    source_paths = [repo_root / "src"] if (repo_root / "src").is_dir() else [repo_root]
    compile_commands = _compile_commands_path(repo_root)
    preprocessor = ASTPreprocessor(source_paths=source_paths, compile_commands_path=compile_commands)
    source_file_count = len(preprocessor.source_files)
    result["source_file_count"] = source_file_count
    if source_file_count == 0:
        result["error"] = "promefuzz_no_source_files"
        return result
    if source_file_count > source_limit:
        result["error"] = f"promefuzz_source_limit_exceeded:{source_file_count}>{source_limit}"
        return result

    builder = BinaryBuilder()
    binaries_ok = bool(builder.check_binaries())
    if not binaries_ok and allow_build and not _PROMEFUZZ_BUILD_ATTEMPTED:
        _PROMEFUZZ_BUILD_ATTEMPTED = True
        binaries_ok = bool(builder.build(force=False))
    elif not binaries_ok and _PROMEFUZZ_BUILD_ATTEMPTED:
        result["error"] = "promefuzz_binaries_unavailable_after_build_attempt"
        return result
    result["binaries_ok"] = binaries_ok
    if not binaries_ok:
        result["error"] = "promefuzz_binaries_unavailable"
        return result

    work_root = companion_root / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    meta, meta_path = preprocessor.run(output_dir=work_root / "meta")

    headers: list[Path] = []
    if (repo_root / "include").is_dir():
        headers.append(repo_root / "include")
    headers.append(repo_root)
    extractor = APIExtractor(header_paths=headers, meta=meta)
    api_collection, api_path = extractor.extract(output_path=work_root / "api_functions.json")

    top_functions = [str(x.name) for x in api_collection.funcs[:40]]
    result.update(
        {
            "ok": True,
            "backend": "promefuzz-mcp",
            "meta_path": str(meta_path),
            "api_path": str(api_path) if api_path else "",
            "api_count": int(api_collection.count),
            "api_functions": top_functions,
        }
    )
    return result


def _run_rag_pipeline(repo_root: Path, companion_root: Path, api_functions: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "enabled": False,
        "ok": False,
        "error": "",
        "embedding_provider": "openrouter",
        "embedding_model": "",
        "embedding_ok": False,
        "rag_degraded": False,
        "rag_degraded_reason": "",
        "semantic_query_count": 0,
        "semantic_hit_count": 0,
        "semantic_hit_rate": 0.0,
        "cache_hit_rate": 0.0,
        "knowledge_base_path": "",
        "document_count": 0,
        "chunk_count": 0,
        "queries": [],
        "samples": [],
    }
    enable_rag = _env_int("SHERPA_PROMEFUZZ_ENABLE_RAG", 1, min_value=0, max_value=1) == 1
    if not enable_rag:
        out["error"] = "rag_disabled"
        return out

    root = Path(str(os.environ.get("SHERPA_PROMEFUZZ_MCP_ROOT") or "/app/promefuzz-mcp")).expanduser()
    if not root.is_dir():
        out["error"] = "promefuzz_mcp_root_missing"
        return out
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    out["enabled"] = True

    try:
        from promefuzz_mcp.comprehender.knowledge import KnowledgeBase
    except Exception as e:
        out["error"] = f"rag_import_failed:{e}"
        return out

    document_paths: list[str] = []
    for rel in ("README.md", "README", "CHANGELOG.md", "docs", "doc", "include", "src"):
        p = repo_root / rel
        if p.exists():
            document_paths.append(str(p))
    if not document_paths:
        out["error"] = "rag_no_document_paths"
        return out

    kb_dir = companion_root / "work" / "knowledge"
    kb = KnowledgeBase(document_paths=document_paths, output_path=str(kb_dir))
    try:
        kb.initialize()
    except Exception as e:
        out["error"] = f"rag_initialize_failed:{e}"
        return out

    queries: list[str] = []
    for fn in api_functions[:8]:
        fn_txt = str(fn or "").strip()
        if fn_txt:
            queries.append(f"usage of {fn_txt}")
    if not queries:
        queries = [f"{repo_root.name} parser", f"{repo_root.name} decoder", f"{repo_root.name} format"]

    sample_rows: list[dict[str, Any]] = []
    semantic_hit_count = 0
    for query in queries[:8]:
        rows = kb.retrieve(query=query, top_k=3)
        if not rows:
            continue
        semantic_hit_count += 1
        sample_rows.append({"query": query, "results": rows})

    semantic_query_count = len(queries[:8])
    semantic_hit_rate = (float(semantic_hit_count) / float(semantic_query_count)) if semantic_query_count > 0 else 0.0
    cache_hit_rate = 1.0 if bool(getattr(kb, "cache_loaded", False)) else 0.0

    out.update(
        {
            "ok": True,
            "knowledge_base_path": str(kb_dir),
            "document_count": len(kb.documents),
            "chunk_count": len(kb.chunks),
            "embedding_provider": str(getattr(kb, "embedding_provider", "openrouter") or "openrouter"),
            "embedding_model": str(getattr(kb, "embedding_model_used", "") or ""),
            "embedding_ok": bool(getattr(kb, "embedding_ok", False)),
            "rag_degraded": bool(getattr(kb, "rag_degraded", False)),
            "rag_degraded_reason": str(getattr(kb, "rag_degraded_reason", "") or ""),
            "semantic_query_count": semantic_query_count,
            "semantic_hit_count": semantic_hit_count,
            "semantic_hit_rate": round(semantic_hit_rate, 6),
            "cache_hit_rate": round(cache_hit_rate, 6),
            "queries": queries[:8],
            "samples": sample_rows[:8],
        }
    )
    return out


def _build_coverage_hints(
    repo_root: Path,
    inventory: RepoInventory,
    promefuzz_doc: dict[str, Any],
    fallback_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    recommended: list[dict[str, Any]] = []

    api_functions = promefuzz_doc.get("api_functions")
    if isinstance(api_functions, list):
        for name in api_functions[:50]:
            symbol = str(name or "").strip()
            if not symbol:
                continue
            recommended.append(
                {
                    "name": symbol,
                    "seed_profile": _seed_profile_for_symbol(symbol),
                    "source": "promefuzz-mcp",
                    "score": _score_symbol(symbol, "include/"),
                }
            )
    if fallback_candidates:
        recommended.extend({**row, "source": "heuristic"} for row in fallback_candidates[:120])

    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in recommended:
        name = str(row.get("name") or "").strip()
        seed_profile = str(row.get("seed_profile") or "unknown")
        if not name:
            continue
        key = (name, seed_profile)
        if key in dedup and int(dedup[key].get("score", 0)) >= int(row.get("score", 0)):
            continue
        dedup[key] = row

    ranked = sorted(dedup.values(), key=lambda x: (-int(x.get("score", 0)), str(x.get("name", ""))))[:80]
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "repo_root": str(repo_root),
        "recommended_targets": ranked,
        "signals": {
            "inventory_source_files": inventory.source_files,
            "inventory_header_files": inventory.header_files,
            "promefuzz_ok": bool(promefuzz_doc.get("ok")),
            "promefuzz_api_count": int(promefuzz_doc.get("api_count") or 0),
            "fallback_candidate_count": len(fallback_candidates),
        },
    }


def _write_status(status_path: Path, payload: dict[str, Any]) -> None:
    body = {
        "schema_version": 1,
        "updated_at": _now_iso(),
        **payload,
    }
    _dump_json_atomic(status_path, body)


def _run_once(job_id: str, output_root: Path, companion_root: Path) -> dict[str, Any]:
    repo_root_raw = _resolve_repo_root(job_id, output_root)
    if not repo_root_raw:
        raise RuntimeError("repo_root_not_ready")
    repo_root = Path(repo_root_raw)
    if not repo_root.is_dir():
        raise RuntimeError(f"repo_root_missing:{repo_root}")

    inventory = _collect_repo_inventory(repo_root)
    fallback_candidates = _extract_symbol_candidates(repo_root)
    promefuzz_doc = _run_promefuzz_pipeline(repo_root, companion_root)
    rag_doc = _run_rag_pipeline(repo_root, companion_root, list(promefuzz_doc.get("api_functions") or []))

    preprocess_doc = {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "job_id": job_id,
        "repo_root": str(repo_root),
        "analysis_backend": str(promefuzz_doc.get("backend") or "fallback-heuristic"),
        "inventory": {
            "source_files": inventory.source_files,
            "header_files": inventory.header_files,
            "doc_files": inventory.doc_files,
            "total_files": inventory.total_files,
            "total_size_bytes": inventory.total_size_bytes,
        },
        "promefuzz": promefuzz_doc,
        "rag": rag_doc,
        "fallback": {
            "candidate_count": len(fallback_candidates),
            "top_candidates": fallback_candidates[:40],
        },
    }
    hints_doc = _build_coverage_hints(repo_root, inventory, promefuzz_doc, fallback_candidates)

    preprocess_path = companion_root / "preprocess.json"
    coverage_hints_path = companion_root / "coverage_hints.json"
    _dump_json_atomic(preprocess_path, preprocess_doc)
    _dump_json_atomic(coverage_hints_path, hints_doc)
    return {
        "repo_root": str(repo_root),
        "analysis_backend": str(preprocess_doc.get("analysis_backend") or ""),
        "candidate_count": len(hints_doc.get("recommended_targets") or []),
        "preprocess_path": str(preprocess_path),
        "coverage_hints_path": str(coverage_hints_path),
        "promefuzz_ok": bool(promefuzz_doc.get("ok")),
        "rag_ok": bool(rag_doc.get("ok")),
        "rag_knowledge_base_path": str(rag_doc.get("knowledge_base_path") or ""),
        "rag_document_count": int(rag_doc.get("document_count") or 0),
        "rag_chunk_count": int(rag_doc.get("chunk_count") or 0),
        "embedding_provider": str(rag_doc.get("embedding_provider") or "openrouter"),
        "embedding_model": str(rag_doc.get("embedding_model") or ""),
        "embedding_ok": bool(rag_doc.get("embedding_ok")),
        "rag_degraded": bool(rag_doc.get("rag_degraded")),
        "rag_degraded_reason": str(rag_doc.get("rag_degraded_reason") or ""),
        "semantic_query_count": int(rag_doc.get("semantic_query_count") or 0),
        "semantic_hit_count": int(rag_doc.get("semantic_hit_count") or 0),
        "semantic_hit_rate": float(rag_doc.get("semantic_hit_rate") or 0.0),
        "cache_hit_rate": float(rag_doc.get("cache_hit_rate") or 0.0),
    }


def main() -> int:
    job_id = str(os.environ.get("SHERPA_JOB_ID") or "").strip()
    if not job_id:
        print("[promefuzz-companion] SHERPA_JOB_ID is required", flush=True)
        return 2

    output_root = Path(str(os.environ.get("SHERPA_OUTPUT_DIR") or "/shared/output")).expanduser()
    companion_root = (output_root / "_k8s_jobs" / job_id / "promefuzz").resolve()
    companion_root.mkdir(parents=True, exist_ok=True)

    poll_sec = _env_int("SHERPA_PROMEFUZZ_POLL_SEC", 30, min_value=5, max_value=600)
    refresh_sec = _env_int("SHERPA_PROMEFUZZ_REFRESH_SEC", 0, min_value=0, max_value=86400)
    run_once_only = _env_int("SHERPA_PROMEFUZZ_RUN_ONCE", 0, min_value=0, max_value=1) == 1
    mcp_url = str(os.environ.get("SHERPA_PROMEFUZZ_MCP_URL") or "").strip()
    mcp_port = _env_int("SHERPA_PROMEFUZZ_MCP_PORT", 18080, min_value=1, max_value=65535)
    mcp_path = str(os.environ.get("SHERPA_PROMEFUZZ_MCP_PATH") or "/mcp").strip() or "/mcp"
    if not mcp_path.startswith("/"):
        mcp_path = f"/{mcp_path}"

    status_path = companion_root / "status.json"
    _write_status(
        status_path,
        {
            "state": "starting",
            "job_id": job_id,
            "companion_root": str(companion_root),
            "poll_sec": poll_sec,
            "refresh_sec": refresh_sec,
            "mcp_url": mcp_url,
            "mcp_port": mcp_port,
            "mcp_path": mcp_path,
            "mcp_ready": bool(mcp_url),
        },
    )
    print(
        f"[promefuzz-companion] started job_id={job_id} poll_sec={poll_sec} refresh_sec={refresh_sec}",
        flush=True,
    )

    last_run_at = 0.0
    last_repo_root = ""
    while True:
        now = time.time()
        try:
            repo_root_now = _resolve_repo_root(job_id, output_root)
            have_outputs = (companion_root / "preprocess.json").is_file() and (companion_root / "coverage_hints.json").is_file()
            should_run = (
                bool(repo_root_now)
                and (
                    repo_root_now != last_repo_root
                    or not have_outputs
                    or (refresh_sec > 0 and (now - last_run_at) >= refresh_sec)
                )
            )
            if not repo_root_now:
                _write_status(
                    status_path,
                    {
                        "state": "waiting_repo_root",
                        "job_id": job_id,
                        "companion_root": str(companion_root),
                        "mcp_url": mcp_url,
                        "mcp_ready": bool(mcp_url),
                    },
                )
            elif should_run:
                _write_status(
                    status_path,
                    {
                        "state": "running",
                        "job_id": job_id,
                        "repo_root": repo_root_now,
                        "mcp_url": mcp_url,
                        "mcp_ready": bool(mcp_url),
                    },
                )
                run_doc = _run_once(job_id, output_root, companion_root)
                last_run_at = now
                last_repo_root = str(run_doc.get("repo_root") or repo_root_now)
                _write_status(
                    status_path,
                    {
                        "state": "ready",
                        "job_id": job_id,
                        "repo_root": last_repo_root,
                        "analysis_backend": run_doc.get("analysis_backend"),
                        "candidate_count": run_doc.get("candidate_count"),
                        "promefuzz_ok": run_doc.get("promefuzz_ok"),
                        "rag_ok": run_doc.get("rag_ok"),
                        "rag_knowledge_base_path": run_doc.get("rag_knowledge_base_path"),
                        "rag_document_count": run_doc.get("rag_document_count"),
                        "rag_chunk_count": run_doc.get("rag_chunk_count"),
                        "embedding_provider": run_doc.get("embedding_provider"),
                        "embedding_model": run_doc.get("embedding_model"),
                        "embedding_ok": run_doc.get("embedding_ok"),
                        "rag_degraded": run_doc.get("rag_degraded"),
                        "rag_degraded_reason": run_doc.get("rag_degraded_reason"),
                        "semantic_query_count": run_doc.get("semantic_query_count"),
                        "semantic_hit_count": run_doc.get("semantic_hit_count"),
                        "semantic_hit_rate": run_doc.get("semantic_hit_rate"),
                        "cache_hit_rate": run_doc.get("cache_hit_rate"),
                        "preprocess_path": run_doc.get("preprocess_path"),
                        "coverage_hints_path": run_doc.get("coverage_hints_path"),
                        "mcp_url": mcp_url,
                        "mcp_ready": bool(mcp_url),
                    },
                )
                print(
                    "[promefuzz-companion] refreshed "
                    f"repo_root={last_repo_root} backend={run_doc.get('analysis_backend')} "
                    f"candidates={run_doc.get('candidate_count')}",
                    flush=True,
                )
                if run_once_only:
                    return 0
            else:
                _write_status(
                    status_path,
                    {
                        "state": "idle",
                        "job_id": job_id,
                        "repo_root": repo_root_now,
                        "seconds_since_last_run": round(max(0.0, now - last_run_at), 2),
                        "mcp_url": mcp_url,
                        "mcp_ready": bool(mcp_url),
                    },
                )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            _write_status(
                status_path,
                {
                    "state": "degraded",
                    "job_id": job_id,
                    "error": err,
                    "traceback": traceback.format_exc()[-4000:],
                    "mcp_url": mcp_url,
                    "mcp_ready": bool(mcp_url),
                },
            )
            print(f"[promefuzz-companion] degraded: {err}", flush=True)
            if run_once_only:
                return 1

        time.sleep(poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
