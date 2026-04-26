from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import Any


_MANIFEST_VERSION = 2
_FRONTIER_VERSION = 1
_DEFAULT_MAX_INPUTS = 32
_DEFAULT_TIMEOUT_SEC = 8
_DEFAULT_PARTIAL_THRESHOLD = 0.5
_EXCLUDED_PARTS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "node_modules",
    "vendor",
    "third_party",
    "third-party",
    "external",
    "out",
    "build",
}


def _frontier_partial_threshold() -> float:
    raw = (os.environ.get("SHERPA_FRONTIER_PARTIAL_THRESHOLD") or "").strip()
    try:
        return max(0.0, min(float(raw), 1.0))
    except Exception:
        return _DEFAULT_PARTIAL_THRESHOLD


def _frontier_weights() -> dict[str, float]:
    raw = (os.environ.get("SHERPA_FRONTIER_WEIGHTS_JSON") or "").strip()
    weights = {"alpha": 1.0, "beta": 0.5, "delta": 0.001, "epsilon": 0.0001}
    if not raw:
        return weights
    try:
        doc = json.loads(raw)
    except Exception:
        return weights
    if not isinstance(doc, dict):
        return weights
    for key in list(weights.keys()):
        try:
            weights[key] = float(doc.get(key, weights[key]))
        except Exception:
            pass
    return weights


def _tokenize_identifier(text: str) -> set[str]:
    parts = [part.strip().lower() for part in re.split(r"[^A-Za-z0-9]+", text or "") if part.strip()]
    return {part for part in parts if len(part) >= 3}


@dataclass
class CoverageReplayResult:
    manifest_path: str
    frontier_path: str
    frontier_summary: dict[str, Any]
    binary_hash: str
    stage_success: bool
    stage_error: str
    manifest_fresh_for_current_binary: bool
    replay_queue_drained: bool
    pending_inputs: int
    failed_inputs: int
    processed_inputs: int
    total_inputs: int
    runtime_sec: float


def _max_replay_inputs() -> int:
    raw = (os.environ.get("SHERPA_PER_INPUT_REPLAY_MAX_INPUTS") or "").strip()
    try:
        return max(1, min(int(raw), 512))
    except Exception:
        return _DEFAULT_MAX_INPUTS


def _replay_timeout_sec() -> int:
    raw = (os.environ.get("SHERPA_PER_INPUT_REPLAY_TIMEOUT_SEC") or "").strip()
    try:
        return max(1, min(int(raw), 120))
    except Exception:
        return _DEFAULT_TIMEOUT_SEC


def _hash_file(path: Path, algo: str = "sha1") -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _repo_source_file(path: str, repo_root: Path) -> bool:
    try:
        resolved = Path(path).resolve()
        root = repo_root.resolve()
        rel = resolved.relative_to(root)
    except Exception:
        return False
    parts = {part.lower() for part in rel.parts}
    if parts & _EXCLUDED_PARTS:
        return False
    if rel.parts and rel.parts[0].lower() == "fuzz":
        return False
    return True


def _load_json_doc(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return dict(doc) if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _frontier_focus(repo_root: Path, target_api: str = "") -> dict[str, Any]:
    focus: dict[str, Any] = {
        "target_api": str(target_api or "").strip(),
        "target_tokens": set(_tokenize_identifier(target_api)),
        "focus_function_names": set(),
        "focus_tokens": set(_tokenize_identifier(target_api)),
    }
    analysis_path = repo_root / "fuzz" / "analysis_context.json"
    doc = _load_json_doc(analysis_path)
    analysis_evidence = dict(doc.get("analysis_evidence") or {})
    candidates = list(analysis_evidence.get("vuln_candidate_inventory") or [])
    callgraph_summary = list(analysis_evidence.get("callgraph_summary") or [])

    target_api_lower = str(target_api or "").strip().lower()
    for item in candidates[:80]:
        if not isinstance(item, dict):
            continue
        item_target_api = str(item.get("target_api") or item.get("api") or "").strip()
        if target_api_lower and item_target_api.lower() != target_api_lower:
            continue
        focus["focus_function_names"].add(item_target_api)
        attack_hint = dict(item.get("attack_hint") or {})
        for step in list(attack_hint.get("key_code_path") or []):
            step_text = str(step or "").strip()
            if not step_text:
                continue
            focus["focus_function_names"].add(step_text)
            focus["focus_tokens"].update(_tokenize_identifier(step_text))

    for item in callgraph_summary[:80]:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or item.get("edge") or "").strip()
        if target_api_lower and target_api_lower not in summary.lower():
            continue
        focus["focus_tokens"].update(_tokenize_identifier(summary))

    focus["focus_function_names"] = {
        str(name).strip()
        for name in set(focus["focus_function_names"])
        if str(name).strip()
    }
    focus["focus_tokens"] = {str(token).strip().lower() for token in set(focus["focus_tokens"]) if str(token).strip()}
    return focus
    try:
        doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return dict(doc) if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _discover_live_inputs(corpus_dir: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    live: dict[str, dict[str, Any]] = {}
    if not corpus_dir.is_dir():
        return live
    for path in sorted(p for p in corpus_dir.rglob("*") if p.is_file()):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = _safe_relpath(path, repo_root)
        live[rel] = {
            "input_relpath": rel,
            "input_sha1": _hash_file(path, "sha1"),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
        }
    return live


def _llvm_tools() -> tuple[str, str]:
    profdata = which("llvm-profdata") or which("llvm-profdata-18") or ""
    cov = which("llvm-cov") or which("llvm-cov-18") or ""
    return profdata, cov


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _region_count(region: Any) -> int:
    if isinstance(region, dict):
        try:
            return int(region.get("count") or 0)
        except Exception:
            return 0
    if isinstance(region, (list, tuple)) and len(region) >= 5:
        try:
            return int(region[4] or 0)
        except Exception:
            return 0
    return 0


def _region_line(region: Any) -> int:
    if isinstance(region, dict):
        try:
            return int(region.get("line_start") or region.get("line") or 0)
        except Exception:
            return 0
    if isinstance(region, (list, tuple)) and region:
        try:
            return int(region[0] or 0)
        except Exception:
            return 0
    return 0


def _summarize_export_doc(export_doc: dict[str, Any], repo_root: Path, *, focus: dict[str, Any] | None = None) -> dict[str, Any]:
    covered_functions: list[str] = []
    uncovered_functions: list[str] = []
    covered_region_count = 0
    total_region_count = 0
    repo_files_seen: set[str] = set()
    frontier_functions: list[dict[str, Any]] = []
    partial_threshold = _frontier_partial_threshold()
    focus = dict(focus or {})
    target_api = str(focus.get("target_api") or "").strip()
    target_api_lower = target_api.lower()
    focus_names = {
        str(name).strip()
        for name in set(focus.get("focus_function_names") or set())
        if str(name).strip()
    }
    focus_tokens = {
        str(token).strip().lower()
        for token in set(focus.get("focus_tokens") or set())
        if str(token).strip()
    }

    for unit in list(export_doc.get("data") or []):
        file_map: dict[str, bool] = {}
        for file_doc in list(unit.get("files") or []):
            file_name = str(file_doc.get("filename") or "").strip()
            if not file_name or not _repo_source_file(file_name, repo_root):
                continue
            repo_files_seen.add(file_name)
            file_map[file_name] = True
            summary = dict(file_doc.get("summary") or {})
            regions = dict(summary.get("regions") or {})
            try:
                covered_region_count += int(regions.get("covered") or 0)
            except Exception:
                pass
            try:
                total_region_count += int(regions.get("count") or 0)
            except Exception:
                pass
        for fn in list(unit.get("functions") or []):
            filenames = [str(x).strip() for x in list(fn.get("filenames") or []) if str(x).strip()]
            allowed_files = [name for name in filenames if file_map.get(name)]
            if filenames and not allowed_files:
                continue
            if not filenames and not repo_files_seen:
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            try:
                count = int(fn.get("count") or 0)
            except Exception:
                count = 0
            if count > 0:
                covered_functions.append(name)
            else:
                uncovered_functions.append(name)
            region_entries = list(fn.get("regions") or [])
            total_fn_regions = len(region_entries)
            uncovered_fn_regions = len([region for region in region_entries if _region_count(region) <= 0])
            covered_fn_regions = max(0, total_fn_regions - uncovered_fn_regions)
            ratio = float(covered_fn_regions) / float(total_fn_regions) if total_fn_regions > 0 else 1.0
            if count > 0 and total_fn_regions > 0 and ratio < partial_threshold and uncovered_fn_regions > 0:
                joined_text = " ".join([name, *(allowed_files or filenames)]).lower()
                target_signal = 0
                if target_api_lower and target_api_lower in joined_text:
                    target_signal += 2
                token_hits = sum(1 for token in focus_tokens if token and token in joined_text)
                target_signal += token_hits
                if name in focus_names:
                    distance_to_target = 0
                elif target_signal > 0:
                    distance_to_target = 1
                else:
                    distance_to_target = 2
                frontier_functions.append(
                    {
                        "name": name,
                        "file": allowed_files[0] if allowed_files else (filenames[0] if filenames else ""),
                        "line": _region_line(region_entries[0]) if region_entries else 0,
                        "covered_region_count": covered_fn_regions,
                        "total_region_count": total_fn_regions,
                        "uncovered_regions_nearby": uncovered_fn_regions,
                        "region_coverage_ratio": round(ratio, 4),
                        "distance_to_target": int(distance_to_target),
                        "target_signal": int(target_signal),
                    }
                )

    covered_functions = sorted(set(covered_functions))
    uncovered_functions = sorted(set(uncovered_functions))
    frontier_functions.sort(
        key=lambda item: (
            int(item.get("distance_to_target") or 99),
            -int(item.get("target_signal") or 0),
            -int(item.get("uncovered_regions_nearby") or 0),
            float(item.get("region_coverage_ratio") or 1.0),
            str(item.get("name") or ""),
        )
    )
    nearby_uncovered_regions = sum(int(item.get("uncovered_regions_nearby") or 0) for item in frontier_functions)
    closest_target_distance = min((int(item.get("distance_to_target") or 99) for item in frontier_functions), default=99)
    target_relevance_count = sum(1 for item in frontier_functions if int(item.get("target_signal") or 0) > 0)
    return {
        "covered_function_count": len(covered_functions),
        "covered_region_count": int(covered_region_count),
        "total_region_count": int(total_region_count),
        "covered_functions_sample": covered_functions[:12],
        "uncovered_functions_sample": uncovered_functions[:12],
        "repo_file_count": len(repo_files_seen),
        "frontier_functions": frontier_functions[:8],
        "unique_frontier_functions": len(frontier_functions),
        "nearby_uncovered_regions": int(nearby_uncovered_regions),
        "closest_target_distance": int(closest_target_distance if closest_target_distance != 99 else 0),
        "target_relevance_count": int(target_relevance_count),
    }


def _export_input_coverage(
    *,
    repo_root: Path,
    replay_binary: Path,
    input_path: Path,
    work_dir: Path,
    llvm_profdata: str,
    llvm_cov: str,
    timeout_sec: int,
    focus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_pattern = work_dir / "profile-%p.profraw"
    for stale in work_dir.glob("profile-*.profraw"):
        try:
            stale.unlink()
        except OSError:
            pass

    env = os.environ.copy()
    env["LLVM_PROFILE_FILE"] = str(profile_pattern)
    env.setdefault("ASAN_OPTIONS", "exitcode=76:detect_leaks=0")
    env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1")
    env.setdefault("LLVM_SYMBOLIZER_PATH", which("llvm-symbolizer") or "")

    started = time.perf_counter()
    run_proc = _run(
        [str(replay_binary), str(input_path)],
        cwd=repo_root,
        env=env,
        timeout=timeout_sec,
    )
    elapsed_us = int(max(0.0, (time.perf_counter() - started) * 1_000_000.0))

    profraw_files = sorted(work_dir.glob("profile-*.profraw"))
    if not profraw_files:
        return {
            "replay_status": "failed",
            "replay_error": "profraw_not_generated",
            "exec_time_us": elapsed_us,
            "exit_code": int(run_proc.returncode or 0),
        }

    profdata_path = work_dir / "input.profdata"
    merge_proc = _run(
        [llvm_profdata, "merge", "-sparse", *(str(p) for p in profraw_files), "-o", str(profdata_path)],
        cwd=repo_root,
        timeout=timeout_sec,
    )
    if int(merge_proc.returncode or 0) != 0 or not profdata_path.is_file():
        return {
            "replay_status": "failed",
            "replay_error": "llvm_profdata_merge_failed",
            "exec_time_us": elapsed_us,
            "exit_code": int(run_proc.returncode or 0),
        }

    export_proc = _run(
        [llvm_cov, "export", str(replay_binary), f"-instr-profile={profdata_path}", "-format=text"],
        cwd=repo_root,
        timeout=max(timeout_sec, 20),
    )
    if int(export_proc.returncode or 0) != 0:
        return {
            "replay_status": "failed",
            "replay_error": "llvm_cov_export_failed",
            "exec_time_us": elapsed_us,
            "exit_code": int(run_proc.returncode or 0),
        }

    try:
        export_doc = json.loads(export_proc.stdout or "{}")
    except Exception:
        return {
            "replay_status": "failed",
            "replay_error": "llvm_cov_export_invalid_json",
            "exec_time_us": elapsed_us,
            "exit_code": int(run_proc.returncode or 0),
        }

    summary = _summarize_export_doc(export_doc, repo_root, focus=focus)
    return {
        "replay_status": "ok",
        "replay_error": "",
        "exec_time_us": elapsed_us,
        "exit_code": int(run_proc.returncode or 0),
        **summary,
    }


def _build_frontier_summary(
    *,
    manifest_inputs: list[dict[str, Any]],
    repo_root: Path,
    replay_binary: Path,
    binary_hash: str,
    pending_inputs: int,
    failed_inputs: int,
    focus: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ok_inputs = [
        dict(item)
        for item in manifest_inputs
        if str(item.get("replay_status") or "") == "ok"
    ]
    weights = _frontier_weights()

    def _frontier_score(item: dict[str, Any]) -> float:
        return (
            float(weights["alpha"]) * float(item.get("unique_frontier_functions") or 0)
            + float(weights["beta"]) * float(item.get("nearby_uncovered_regions") or 0)
            + 0.75 * float(item.get("target_relevance_count") or 0)
            - 0.5 * float(item.get("closest_target_distance") or 0)
            - float(weights["delta"]) * float(item.get("exec_time_us") or 0)
            - float(weights["epsilon"]) * float(item.get("size_bytes") or 0)
        )

    for item in ok_inputs:
        item["frontier_score"] = round(_frontier_score(item), 6)

    ok_inputs.sort(
        key=lambda item: (
            float(item.get("frontier_score") or 0.0),
            int(item.get("covered_function_count") or 0),
            int(item.get("covered_region_count") or 0),
        ),
        reverse=True,
    )
    top_inputs = []
    touched_functions: list[str] = []
    function_to_inputs: dict[str, list[str]] = {}
    for item in ok_inputs[:5]:
        fn_sample = [str(x).strip() for x in list(item.get("covered_functions_sample") or []) if str(x).strip()]
        touched_functions.extend(fn_sample[:5])
        input_relpath = str(item.get("input_relpath") or "")
        for fn_name in fn_sample[:8]:
            refs = function_to_inputs.setdefault(fn_name, [])
            if input_relpath and input_relpath not in refs:
                refs.append(input_relpath)
        top_inputs.append(
            {
                "input_relpath": input_relpath,
                "size_bytes": int(item.get("size_bytes") or 0),
                "covered_function_count": int(item.get("covered_function_count") or 0),
                "covered_region_count": int(item.get("covered_region_count") or 0),
                "exec_time_us": int(item.get("exec_time_us") or 0),
                "unique_frontier_functions": int(item.get("unique_frontier_functions") or 0),
                "nearby_uncovered_regions": int(item.get("nearby_uncovered_regions") or 0),
                "target_relevance_count": int(item.get("target_relevance_count") or 0),
                "closest_target_distance": int(item.get("closest_target_distance") or 0),
                "frontier_score": float(item.get("frontier_score") or 0.0),
                "covered_functions_sample": fn_sample[:8],
                "frontier_functions": list(item.get("frontier_functions") or [])[:5],
                "rationale": (
                    f"partial frontier via {int(item.get('unique_frontier_functions') or 0)} functions"
                    f" with {int(item.get('nearby_uncovered_regions') or 0)} uncovered regions nearby"
                    f"; target_relevance={int(item.get('target_relevance_count') or 0)}"
                ),
                "repo_file_count": int(item.get("repo_file_count") or 0),
            }
        )

    top_frontier_functions = [
        {
            "name": name,
            "input_count": len(input_relpaths),
            "input_relpaths": input_relpaths[:4],
            "best_distance_to_target": min(
                (
                    int(fn.get("distance_to_target") or 99)
                    for item in ok_inputs
                    for fn in list(item.get("frontier_functions") or [])
                    if isinstance(fn, dict) and str(fn.get("name") or "") == name
                ),
                default=99,
            ),
        }
        for name, input_relpaths in sorted(
            function_to_inputs.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )[:12]
    ]

    summary = {
        "version": _FRONTIER_VERSION,
        "binary_hash": binary_hash,
        "replay_binary": str(replay_binary),
        "generated_at": int(time.time()),
        "top_inputs": top_inputs,
        "top_input_count": len(top_inputs),
        "top_frontier_functions": top_frontier_functions,
        "top_frontier_function_count": len(top_frontier_functions),
        "failed_input_count": int(failed_inputs),
        "pending_input_count": int(pending_inputs),
        "covered_function_union_sample": sorted({name for name in touched_functions if name})[:20],
    }
    frontier_doc = {
        **summary,
        "function_to_inputs": function_to_inputs,
        "inputs": [
            {
                "input_relpath": str(item.get("input_relpath") or ""),
                "replay_status": str(item.get("replay_status") or ""),
                "covered_function_count": int(item.get("covered_function_count") or 0),
                "covered_region_count": int(item.get("covered_region_count") or 0),
                "exec_time_us": int(item.get("exec_time_us") or 0),
                "covered_functions_sample": list(item.get("covered_functions_sample") or []),
                "repo_file_count": int(item.get("repo_file_count") or 0),
            }
            for item in ok_inputs[:32]
        ],
        "repo_root": str(repo_root),
    }
    return summary, frontier_doc


def collect_per_input_frontier(
    *,
    repo_root: Path,
    fuzzer_name: str,
    replay_binary: Path,
    target_api: str = "",
) -> CoverageReplayResult:
    started = time.perf_counter()
    coverage_dir = repo_root / "fuzz" / "coverage" / "per_input" / fuzzer_name
    work_dir = coverage_dir / "tmp"
    manifest_path = coverage_dir / "manifest.json"
    frontier_path = coverage_dir / "frontier.json"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    empty_summary = {
        "version": _FRONTIER_VERSION,
        "binary_hash": "",
        "replay_binary": str(replay_binary),
        "generated_at": int(time.time()),
        "top_inputs": [],
        "top_input_count": 0,
        "top_frontier_functions": [],
        "top_frontier_function_count": 0,
        "failed_input_count": 0,
        "pending_input_count": 0,
        "covered_function_union_sample": [],
    }

    if not replay_binary.is_file():
        return CoverageReplayResult(
            manifest_path=str(manifest_path),
            frontier_path=str(frontier_path),
            frontier_summary=empty_summary,
            binary_hash="",
            stage_success=False,
            stage_error="replay_binary_missing",
            manifest_fresh_for_current_binary=False,
            replay_queue_drained=False,
            pending_inputs=0,
            failed_inputs=0,
            processed_inputs=0,
            total_inputs=0,
            runtime_sec=time.perf_counter() - started,
        )

    llvm_profdata, llvm_cov = _llvm_tools()
    if not llvm_profdata or not llvm_cov:
        return CoverageReplayResult(
            manifest_path=str(manifest_path),
            frontier_path=str(frontier_path),
            frontier_summary=empty_summary,
            binary_hash="",
            stage_success=False,
            stage_error="llvm_cov_tools_missing",
            manifest_fresh_for_current_binary=False,
            replay_queue_drained=False,
            pending_inputs=0,
            failed_inputs=0,
            processed_inputs=0,
            total_inputs=0,
            runtime_sec=time.perf_counter() - started,
        )

    binary_hash = _hash_file(replay_binary, "sha256")
    focus = _frontier_focus(repo_root, target_api=target_api)
    previous_manifest = _load_json_doc(manifest_path)
    previous_inputs = {
        str(item.get("input_relpath") or ""): dict(item)
        for item in list(previous_manifest.get("inputs") or [])
        if str(item.get("input_relpath") or "").strip()
    }
    live_inputs = _discover_live_inputs(repo_root / "fuzz" / "corpus" / fuzzer_name, repo_root)
    full_replay_required = str(previous_manifest.get("binary_hash") or "") != binary_hash

    candidate_queue: list[str] = []
    if full_replay_required:
        candidate_queue.extend(sorted(live_inputs.keys()))
    else:
        for rel, meta in live_inputs.items():
            prev = previous_inputs.get(rel) or {}
            if (
                not prev
                or str(prev.get("input_sha1") or "") != str(meta.get("input_sha1") or "")
                or int(prev.get("size_bytes") or 0) != int(meta.get("size_bytes") or 0)
                or int(prev.get("mtime_ns") or 0) != int(meta.get("mtime_ns") or 0)
                or str(prev.get("replay_status") or "") == "pending"
            ):
                candidate_queue.append(rel)
    candidate_queue = sorted(set(candidate_queue))

    max_inputs = _max_replay_inputs()
    timeout_sec = _replay_timeout_sec()
    process_queue = candidate_queue[:max_inputs]
    pending_queue = candidate_queue[max_inputs:]

    next_inputs: dict[str, dict[str, Any]] = {}
    for rel, meta in live_inputs.items():
        base = dict(previous_inputs.get(rel) or {})
        base.update(meta)
        base.setdefault("replay_status", "pending" if full_replay_required else str(base.get("replay_status") or "pending"))
        next_inputs[rel] = base

    for rel in pending_queue:
        item = dict(next_inputs.get(rel) or live_inputs.get(rel) or {})
        item["replay_status"] = "pending"
        item["replay_error"] = ""
        next_inputs[rel] = item

    stage_success = True
    stage_error = ""
    failed_inputs = 0
    processed_inputs = 0
    for rel in process_queue:
        input_path = repo_root / rel
        if not input_path.is_file():
            next_inputs.pop(rel, None)
            continue
        input_work_dir = work_dir / hashlib.sha1(rel.encode("utf-8", "replace")).hexdigest()[:12]
        input_work_dir.mkdir(parents=True, exist_ok=True)
        try:
            replay_doc = _export_input_coverage(
                repo_root=repo_root,
                replay_binary=replay_binary,
                input_path=input_path,
                work_dir=input_work_dir,
                llvm_profdata=llvm_profdata,
                llvm_cov=llvm_cov,
                timeout_sec=timeout_sec,
                focus=focus,
            )
        except subprocess.TimeoutExpired:
            replay_doc = {
                "replay_status": "failed",
                "replay_error": "replay_timeout",
                "exec_time_us": int(timeout_sec * 1_000_000),
                "exit_code": 124,
            }
        except Exception as exc:
            stage_success = False
            stage_error = f"replay_stage_error:{type(exc).__name__}"
            break

        entry = dict(next_inputs.get(rel) or live_inputs.get(rel) or {})
        entry.update(replay_doc)
        entry["replayed_at"] = int(time.time())
        next_inputs[rel] = entry
        processed_inputs += 1
        if str(replay_doc.get("replay_status") or "") != "ok":
            failed_inputs += 1

    manifest_inputs = [dict(next_inputs[key]) for key in sorted(next_inputs.keys())]
    pending_inputs = len(
        [
            item
            for item in manifest_inputs
            if str(item.get("replay_status") or "") == "pending"
        ]
    )
    failed_inputs = len(
        [
            item
            for item in manifest_inputs
            if str(item.get("replay_status") or "") == "failed"
        ]
    )

    frontier_summary, frontier_doc = _build_frontier_summary(
        manifest_inputs=manifest_inputs,
        repo_root=repo_root,
        replay_binary=replay_binary,
        binary_hash=binary_hash,
        pending_inputs=pending_inputs,
        failed_inputs=failed_inputs,
        focus=focus,
    )
    manifest_doc = {
        "version": _MANIFEST_VERSION,
        "generated_at": int(time.time()),
        "binary_hash": binary_hash,
        "replay_binary": str(replay_binary),
        "fuzzer_name": fuzzer_name,
        "stage_success": stage_success,
        "stage_error": stage_error,
        "inputs": manifest_inputs,
    }
    manifest_path.write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frontier_path.write_text(json.dumps(frontier_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return CoverageReplayResult(
        manifest_path=str(manifest_path),
        frontier_path=str(frontier_path),
        frontier_summary=frontier_summary,
        binary_hash=binary_hash,
        stage_success=stage_success,
        stage_error=stage_error,
        manifest_fresh_for_current_binary=bool(binary_hash),
        replay_queue_drained=(pending_inputs == 0),
        pending_inputs=pending_inputs,
        failed_inputs=failed_inputs,
        processed_inputs=processed_inputs,
        total_inputs=len(live_inputs),
        runtime_sec=time.perf_counter() - started,
    )
