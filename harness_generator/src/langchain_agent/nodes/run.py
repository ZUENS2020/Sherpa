"""Carved from workflow_graph.py - '_node_run' LangGraph node."""

from __future__ import annotations
from loguru import logger
import hashlib
import importlib
import json
import os
import re
import subprocess
import tempfile
import textwrap
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict, cast
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from persistent_config import load_config
import workflow_common as _wf_common
import workflow_observability as _wf_obs
import workflow_coverage_decision as _wf_coverage_decision
import workflow_normalization as _wf_norm
import workflow_target_scoring as _wf_target_scoring
import workflow_target_selection as _wf_target_selection
import workflow_summary as _wf_summary
import procedural_memory as _proc_mem
from coverage_replay import collect_per_input_frontier
from seed_families import seed_families_for_target as _seed_families_for_target
from workflow_context_store import (
    context_dir_for_repo_root,
    merge_result_into_contexts,
    read_context_docs,
    strip_meta,
    write_context_docs,
)
from fuzz_unharnessed_repo import (
    FuzzerRunResult,
    HarnessGeneratorError,
    NonOssFuzzHarnessGenerator,
    RepoSpec,
    extract_crash_stack_signature,
    snapshot_repo_text,
    write_patch_from_snapshot,
)

from wf_state import (
    FuzzWorkflowRuntimeState,
    _enter_step,
    _fmt_dt,
    _sha256_text,
    _wf_log,
)
from workflow_helpers import (
    _auto_stop_policy,
    _build_harness_feedback,
    _build_seed_feedback,
    _calc_parallel_batch_budget,
    _emit_fuzz_metrics,
    _execution_target_fuzzer_aliases,
    _execution_target_fuzzer_name,
    _execution_target_identity,
    _extract_repair_top_trace,
    _filter_fuzzer_bins_by_execution_plan,
    _max_same_timeout_repeats,
    _order_fuzzer_bins_by_execution_plan,
    _preferred_execution_target,
    _quality_flags_from_seed_quality,
    _record_decision_trace,
    _remaining_time_budget_sec,
    _run_cpu_budget,
    _run_finalize_timeout_sec,
    _run_idle_timeout_sec,
    _run_ignore_non_fatal_enabled,
    _run_inner_workers_min,
    _run_inner_workers_target,
    _run_outer_parallelism_max,
    _run_parallel_early_stop_enabled,
    _run_parallel_engine,
    _run_stop_on_first_crash,
    _seed_quality_from_run_details_for_target,
    _solve_parallelism,
    _sync_execution_plan_doc_from_selected_targets,
    _target_type_from_run_details,
    _time_budget_exceeded_state,
    _verify_stage_no_ai,
    _workflow_target_state_from_execution_plan,
    _write_repro_context,
)


def _node_run(state: FuzzWorkflowRuntimeState) -> FuzzWorkflowRuntimeState:
    gen = state.get("generator")
    if gen is None:
        raise RuntimeError("workflow not initialized: missing generator")
    state, stop_now = _enter_step(state, "run")
    if stop_now:
        return state
    t0 = time.perf_counter()
    _wf_log(cast(dict[str, Any], state), "-> run")
    try:
        # If we've already seen crashes in a previous round, archive old artifacts so
        # new crashes are detectable.
        fix_attempts = int(state.get("crash_fix_attempts") or 0)
        if fix_attempts:
            try:
                art_dir = gen.fuzz_out_dir / "artifacts"
                if art_dir.is_dir():
                    archive = art_dir / f"old-{fix_attempts}"
                    archive.mkdir(exist_ok=True)
                    for p in art_dir.glob("*"):
                        if p.is_file():
                            p.rename(archive / p.name)
            except Exception:
                pass

        bins = gen._discover_fuzz_binaries()
        _, execution_plan_doc = _sync_execution_plan_doc_from_selected_targets(gen.repo_root)
        execution_targets = [
            item for item in list(execution_plan_doc.get("execution_targets") or [])
            if isinstance(item, dict)
        ]
        boundary_target_state = _workflow_target_state_from_execution_plan(
            gen.repo_root,
            execution_plan_doc,
        )
        if boundary_target_state:
            state = cast(FuzzWorkflowRuntimeState, {**state, **boundary_target_state})
        if execution_targets:
            bins = _filter_fuzzer_bins_by_execution_plan(list(bins), execution_targets)
        if not bins:
            if execution_targets:
                raise HarnessGeneratorError("No fuzzer binaries found under fuzz/out/ matching execution_plan.json")
            raise HarnessGeneratorError("No fuzzer binaries found under fuzz/out/")

        crash_found = False
        last_artifact = ""
        last_fuzzer = ""
        run_rc = 0
        crash_evidence = "none"
        run_error_kind = ""
        run_terminal_reason = ""
        run_idle_seconds = 0
        run_last_error = ""
        run_details: list[dict[str, Any]] = []
        run_batch_plan: list[dict[str, Any]] = []
        run_children_exit_count = 0
        run_cancel_requested_count = 0
        run_cancel_effective_count = 0
        total_time_budget = _wf_common.parse_budget_value(state.get("time_budget"), default=900)
        run_time_budget_raw = state.get("run_time_budget")
        if run_time_budget_raw is None:
            configured_run_time_budget = total_time_budget
        else:
            configured_run_time_budget = _wf_common.parse_budget_value(run_time_budget_raw, default=total_time_budget)
        if configured_run_time_budget < 0:
            raise HarnessGeneratorError("run_time_budget must be >= 0")
        total_budget_unlimited = total_time_budget <= 0
        prev_crash_sig = str(state.get("crash_signature") or "").strip()
        prev_crash_repeats = int(state.get("same_crash_repeats") or 0)
        prev_timeout_sig = str(state.get("timeout_signature") or "").strip()
        prev_timeout_repeats = int(state.get("same_timeout_repeats") or 0)
        max_same_crash_repeats_raw = os.environ.get("SHERPA_WORKFLOW_MAX_SAME_CRASH_REPEATS", "1")
        try:
            max_same_crash_repeats = max(0, min(int(max_same_crash_repeats_raw), 10))
        except Exception:
            max_same_crash_repeats = 1
        max_same_timeout_repeats = _max_same_timeout_repeats()
        auto_stop_policy = _auto_stop_policy()
        max_parallel_raw = os.environ.get("SHERPA_PARALLEL_FUZZERS", "3")
        try:
            requested_outer_parallelism = max(1, min(int(max_parallel_raw), 64))
        except Exception:
            requested_outer_parallelism = 3
        stop_on_first_crash = _run_stop_on_first_crash()
        parallel_early_stop = _run_parallel_early_stop_enabled()
        if stop_on_first_crash and len(bins) > 1 and not parallel_early_stop:
            # Compatibility mode: force serial when parallel early stop is disabled.
            requested_outer_parallelism = 1
        cpu_budget = _run_cpu_budget()
        outer_parallelism_max = _run_outer_parallelism_max(requested_outer_parallelism)
        inner_workers_min = _run_inner_workers_min()
        requested_inner_workers = _run_inner_workers_target()
        requested_engine = _run_parallel_engine()
        ignore_non_fatal = _run_ignore_non_fatal_enabled()
        solved_parallel = _solve_parallelism(
            cpu_budget=cpu_budget,
            n_targets=len(bins),
            requested_outer=requested_outer_parallelism,
            outer_parallelism_max=outer_parallelism_max,
            inner_workers_min=inner_workers_min,
            requested_inner=requested_inner_workers,
            engine=requested_engine,
            sanitizer=str(getattr(gen, "sanitizer", "") or ""),
        )
        max_parallel = int(solved_parallel.get("outer_parallelism") or 1)
        inner_workers = int(solved_parallel.get("inner_workers") or 1)
        parallel_engine = str(solved_parallel.get("parallel_engine") or "single")
        reload_enabled = bool(solved_parallel.get("reload_enabled"))
        parallel_warning = str(solved_parallel.get("warning") or "").strip()
        idle_timeout_sec = _run_idle_timeout_sec()
        finalize_timeout_sec = _run_finalize_timeout_sec()

        current_run_parallel_cfg = {
            bin_path.name: {
                "parallel_engine": parallel_engine,
                "parallel_role": "reserved",
                "outer_slot": idx % max(1, max_parallel),
                "inner_workers": inner_workers,
                "reload_enabled": reload_enabled,
                "ignore_non_fatal": ignore_non_fatal,
            }
            for idx, bin_path in enumerate(bins)
        }
        prev_run_parallel_cfg = getattr(gen, "current_run_parallel_config_by_fuzzer", None)
        setattr(gen, "current_run_parallel_config_by_fuzzer", current_run_parallel_cfg)

        _wf_log(
            cast(dict[str, Any], state),
            (
                f"run: fuzzers={len(bins)} parallel_outer={max_parallel} inner={inner_workers} "
                f"engine={parallel_engine} cpu_budget={cpu_budget} "
                f"stop_on_first_crash={int(stop_on_first_crash)} "
                f"parallel_early_stop={int(parallel_early_stop)}"
            ),
        )
        if parallel_warning:
            _wf_log(cast(dict[str, Any], state), f"run: {parallel_warning}")

        def _calc_crash_signature(fuzzer_name: str, artifact_path: str) -> str:
            parts: list[str] = [f"fuzzer={fuzzer_name}", f"artifact={artifact_path}"]
            crash_info = gen.repo_root / "crash_info.md"
            crash_analysis = gen.repo_root / "crash_analysis.md"
            combined_log = ""
            for p in (crash_info, crash_analysis):
                if not p.is_file():
                    continue
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                tail = "\n".join(txt.splitlines()[-400:])
                parts.append(f"== {p.name} ==\n{tail}")
                combined_log += txt + "\n"
            # Also compute stack-based signature for better dedup
            stack_sig = extract_crash_stack_signature(combined_log)
            nonlocal _last_stack_sig
            _last_stack_sig = stack_sig
            crash_type = str(stack_sig.get("crash_type") or "unknown").strip().lower() or "unknown"
            top_frames = str(stack_sig.get("top_frames") or "").strip()
            stack_top = top_frames.split("|", 1)[0].strip() if top_frames else "unknown_top"
            key_frame_hash = str(stack_sig.get("stack_signature") or "").strip() or _sha256_text(
                f"{crash_type}:{top_frames}"
            )[:16]
            normalized = f"{crash_type}|{stack_top}|{key_frame_hash}"
            if crash_type != "unknown" or stack_top != "unknown_top":
                return normalized
            return _sha256_text("\n\n".join(parts))

        _last_stack_sig: dict[str, str] = {}

        def _calc_timeout_signature(kind: str, details: list[dict[str, Any]]) -> str:
            parts: list[str] = [f"kind={kind}"]
            for d in details[:5]:
                parts.append(
                    "|".join(
                        [
                            str(d.get("fuzzer") or ""),
                            str(d.get("run_error_kind") or ""),
                            str(d.get("effective_rc") or d.get("rc") or ""),
                            str(d.get("error") or "")[:400],
                            str(d.get("first_artifact") or ""),
                        ]
                    )
                )
            return _sha256_text("\n".join(parts))

        last_seed_profile = str(state.get("coverage_seed_profile") or "")
        seed_count_total: dict[str, int] = {"repo_examples": 0, "ai": 0, "radamsa": 0, "deterministic": 0, "total": 0}
        seed_count_raw_total: dict[str, int] = {"repo_examples": 0, "ai": 0, "radamsa": 0, "deterministic": 0, "total": 0}
        seed_count_filtered_total: dict[str, int] = {"repo_examples": 0, "ai": 0, "radamsa": 0, "deterministic": 0, "total": 0}
        seed_generation_failed_fuzzers: list[str] = []
        seed_generation_error_by_fuzzer: dict[str, str] = {}
        seed_generation_skipped_reason = ""

        def _accumulate_seed_counts(dst: dict[str, int], src: Any) -> None:
            if not isinstance(src, dict):
                return
            for key in ("repo_examples", "ai", "radamsa", "deterministic", "total"):
                try:
                    dst[key] = int(dst.get(key, 0)) + int(src.get(key) or 0)
                except Exception:
                    continue

        seed_sources: set[str] = set()
        repo_examples_filtered = False
        repo_examples_rejected_count = 0
        repo_examples_accepted_count = 0
        seed_noise_rejected_count = 0
        missing_execution_targets: list[str] = []
        seed_family_coverage_state: dict[str, Any] = {}
        bins = _order_fuzzer_bins_by_execution_plan(list(bins), execution_targets)
        preferred_target = _preferred_execution_target(execution_targets, cast(dict[str, Any], state))
        preferred_identity = _execution_target_identity(preferred_target) if preferred_target else {}
        execution_target_by_fuzzer: dict[str, dict[str, Any]] = {}
        for item in execution_targets:
            for alias in _execution_target_fuzzer_aliases(item):
                execution_target_by_fuzzer.setdefault(alias, dict(item))
        bins_by_name = {p.name: p for p in bins}
        bins_by_stem = {p.stem: p for p in bins}
        seed_fuzzers: list[Path] = []
        if execution_targets:
            for item in execution_targets:
                candidate = None
                for alias in _execution_target_fuzzer_aliases(item):
                    candidate = bins_by_name.get(alias) or bins_by_stem.get(Path(alias).stem)
                    if candidate is not None:
                        break
                if candidate is not None:
                    if candidate not in seed_fuzzers:
                        seed_fuzzers.append(candidate)
                else:
                    missing_name = str(item.get("target_name") or _execution_target_fuzzer_name(item))
                    if missing_name and missing_name not in missing_execution_targets:
                        missing_execution_targets.append(missing_name)
        if not seed_fuzzers:
            seed_fuzzers = list(bins)
        if _verify_stage_no_ai():
            seed_generation_skipped_reason = "verify_stage_no_ai"
            _wf_log(cast(dict[str, Any], state), "run: AI seed generation skipped by SHERPA_VERIFY_STAGE_NO_AI=1")
            bootstrap_fn = getattr(gen, "_bootstrap_deterministic_seed_corpus", None)
            if callable(bootstrap_fn):
                for bin_path in seed_fuzzers:
                    fuzzer_name = bin_path.name
                    try:
                        meta = bootstrap_fn(fuzzer_name)
                        if not isinstance(meta, dict):
                            continue
                        profile_map = getattr(gen, "last_seed_profile_by_fuzzer", {}) or {}
                        if not last_seed_profile:
                            last_seed_profile = str(profile_map.get(fuzzer_name) or meta.get("seed_profile") or "")
                        _accumulate_seed_counts(seed_count_total, meta.get("counts") or {})
                        _accumulate_seed_counts(seed_count_raw_total, meta.get("seed_counts_raw") or {})
                        _accumulate_seed_counts(seed_count_filtered_total, meta.get("seed_counts_filtered") or {})
                        sources = meta.get("sources") or []
                        if isinstance(sources, list):
                            for src in sources:
                                src_text = str(src or "").strip()
                                if src_text:
                                    seed_sources.add(src_text)
                        repo_examples_filtered = bool(meta.get("repo_examples_filtered") or repo_examples_filtered)
                        repo_examples_rejected_count += int(meta.get("repo_examples_rejected_count") or 0)
                        repo_examples_accepted_count += int(meta.get("repo_examples_accepted_count") or 0)
                        seed_noise_rejected_count += int(meta.get("seed_noise_rejected_count") or 0)
                        if not seed_family_coverage_state and isinstance(meta.get("seed_family_coverage"), dict):
                            seed_family_coverage_state = dict(meta.get("seed_family_coverage") or {})
                    except Exception as e:
                        seed_generation_failed_fuzzers.append(fuzzer_name)
                        seed_generation_error_by_fuzzer[fuzzer_name] = str(e)[:400]
                        logger.info(f"[warn] deterministic seed bootstrap skipped ({fuzzer_name}): {e}")
        else:
            _wf_log(cast(dict[str, Any], state), "run: generating AI seeds before fuzzing")
            # Seed generation uses OpenCode and shared repo context; keep it serial.
            prev_seed_timeout = getattr(gen, "seed_generation_timeout_sec", None)
            try:
                for idx, bin_path in enumerate(seed_fuzzers):
                    remaining_for_seed = _remaining_time_budget_sec(state, min_timeout=0)
                    if remaining_for_seed <= 0:
                        return _time_budget_exceeded_state(state, step_name="run")
                    fuzzers_left = len(seed_fuzzers) - idx
                    per_fuzzer_budget = max(1, remaining_for_seed // max(1, fuzzers_left))
                    setattr(gen, "seed_generation_timeout_sec", per_fuzzer_budget)
                    fuzzer_name = bin_path.name
                    try:
                        gen._pass_generate_seeds(fuzzer_name)
                        profile_map = getattr(gen, "last_seed_profile_by_fuzzer", {}) or {}
                        if not last_seed_profile:
                            last_seed_profile = str(profile_map.get(fuzzer_name) or "")
                        bootstrap_map = getattr(gen, "last_seed_bootstrap_by_fuzzer", {}) or {}
                        meta = bootstrap_map.get(fuzzer_name) or {}
                        if isinstance(meta, dict):
                            _accumulate_seed_counts(seed_count_total, meta.get("counts") or {})
                            _accumulate_seed_counts(seed_count_raw_total, meta.get("seed_counts_raw") or {})
                            _accumulate_seed_counts(seed_count_filtered_total, meta.get("seed_counts_filtered") or {})
                            sources = meta.get("sources") or []
                            if isinstance(sources, list):
                                for src in sources:
                                    src_text = str(src or "").strip()
                                    if src_text:
                                        seed_sources.add(src_text)
                            repo_examples_filtered = bool(meta.get("repo_examples_filtered") or repo_examples_filtered)
                            repo_examples_rejected_count += int(meta.get("repo_examples_rejected_count") or 0)
                            repo_examples_accepted_count += int(meta.get("repo_examples_accepted_count") or 0)
                            seed_noise_rejected_count += int(meta.get("seed_noise_rejected_count") or 0)
                            if not seed_family_coverage_state and isinstance(meta.get("seed_family_coverage"), dict):
                                seed_family_coverage_state = dict(meta.get("seed_family_coverage") or {})
                    except Exception as e:
                        # Seed generation is best-effort; do not block fuzzing.
                        seed_generation_failed_fuzzers.append(fuzzer_name)
                        seed_generation_error_by_fuzzer[fuzzer_name] = str(e)[:400]
                        logger.info(f"[warn] seed generation skipped ({fuzzer_name}): {e}")
            finally:
                setattr(gen, "seed_generation_timeout_sec", prev_seed_timeout)

        run_results: dict[str, FuzzerRunResult] = {}
        run_exec_errors: dict[str, str] = {}
        finalized_fuzzers: set[str] = set()
        first_crash_fuzzer = ""
        early_stop_reason = ""
        early_stopped_fuzzers: list[str] = []

        def _run_one(bin_path: Path) -> tuple[str, FuzzerRunResult]:
            return bin_path.name, gen._run_fuzzer(bin_path)

        def _capture_timeout_from_error(err_text: str) -> tuple[str, int]:
            lowered = (err_text or "").lower()
            if "idle-timeout" in lowered:
                return "run_idle_timeout", idle_timeout_sec
            if "timed out after" in lowered or "[timeout]" in lowered:
                return "run_timeout", 0
            return "", 0

        # Execute fuzzers in parallel batches and cap each batch to remaining total budget.
        pending_bins = list(bins)
        prev_run_budget = getattr(gen, "current_run_time_budget_sec", None)
        prev_run_hard_timeout = getattr(gen, "current_run_hard_timeout_sec", None)
        try:
            while pending_bins:
                remaining_for_run = _remaining_time_budget_sec(state, min_timeout=0)
                if remaining_for_run <= 0:
                    if not run_last_error:
                        run_last_error = "time budget exceeded during run phase"
                    if not run_error_kind:
                        run_error_kind = "workflow_time_budget_exceeded"
                    for skipped in pending_bins:
                        run_exec_errors[skipped.name] = "skipped: workflow total time budget exhausted before execution"
                        finalized_fuzzers.add(skipped.name)
                    pending_bins = []
                    break

                rounds_left, round_budget, hard_timeout = _calc_parallel_batch_budget(
                    pending_count=len(pending_bins),
                    max_parallel=max_parallel,
                    remaining_for_run=remaining_for_run,
                    configured_run_time_budget=configured_run_time_budget,
                    total_budget_unlimited=total_budget_unlimited,
                )
                setattr(gen, "current_run_time_budget_sec", round_budget)
                setattr(gen, "current_run_hard_timeout_sec", hard_timeout)

                batch = pending_bins[:max_parallel]
                pending_bins = pending_bins[max_parallel:]
                run_batch_plan.append(
                    {
                        "round": len(run_batch_plan) + 1,
                        "batch_size": len(batch),
                        "pending_before": len(batch) + len(pending_bins),
                        "rounds_left": rounds_left,
                        "remaining_total_budget_sec": remaining_for_run,
                        "round_budget_sec": round_budget,
                        "hard_timeout_sec": hard_timeout,
                    }
                )
                _wf_log(
                    cast(dict[str, Any], state),
                    (
                        "run batch: "
                        f"size={len(batch)} round_budget={round_budget}s hard_timeout={hard_timeout}s "
                        f"remaining_total={remaining_for_run}s"
                    ),
                )

                if len(batch) <= 1:
                    for bin_path in batch:
                        try:
                            name, run = _run_one(bin_path)
                            run_results[name] = run
                            finalized_fuzzers.add(name)
                            run_children_exit_count += 1
                            if stop_on_first_crash and run.crash_found:
                                pending_bins = []
                                break
                        except Exception as e:
                            run_exec_errors[bin_path.name] = str(e)
                            finalized_fuzzers.add(bin_path.name)
                            run_children_exit_count += 1
                            detected_kind, detected_idle = _capture_timeout_from_error(str(e))
                            if detected_kind and not run_error_kind:
                                run_error_kind = detected_kind
                                run_terminal_reason = detected_kind
                                if detected_idle > 0:
                                    run_idle_seconds = detected_idle
                else:
                    with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                        futures = {pool.submit(_run_one, bin_path): bin_path for bin_path in batch}
                        batch_should_stop = False
                        processed_futures: set[Any] = set()
                        for fut in as_completed(futures):
                            bin_path = futures[fut]
                            processed_futures.add(fut)
                            try:
                                name, run = fut.result()
                                run_results[name] = run
                                finalized_fuzzers.add(name)
                                run_children_exit_count += 1
                                if (
                                    stop_on_first_crash
                                    and parallel_early_stop
                                    and run.crash_found
                                ):
                                    first_crash_fuzzer = str(name)
                                    early_stop_reason = "first_crash_parallel_early_stop"
                                    terminator = getattr(gen, "terminate_active_run_processes", None)
                                    if callable(terminator):
                                        try:
                                            terminator(reason=f"first_crash:{name}")
                                        except Exception:
                                            pass
                                    for pending_fut in futures:
                                        if pending_fut is not fut:
                                            run_cancel_requested_count += 1
                                            try:
                                                if pending_fut.cancel():
                                                    run_cancel_effective_count += 1
                                            except Exception:
                                                pass
                                    for other_bin in batch:
                                        if other_bin.name != name and other_bin.name not in early_stopped_fuzzers:
                                            early_stopped_fuzzers.append(other_bin.name)
                                    # Collect already-finished futures before leaving this batch
                                    # so early-stop does not drop completed results.
                                    for remaining_fut, remaining_bin in futures.items():
                                        if remaining_fut in processed_futures or remaining_fut is fut:
                                            continue
                                        if not remaining_fut.done():
                                            continue
                                        processed_futures.add(remaining_fut)
                                        try:
                                            rname, rrun = remaining_fut.result(timeout=0)
                                            run_results[rname] = rrun
                                            finalized_fuzzers.add(rname)
                                            run_children_exit_count += 1
                                        except Exception as e:
                                            run_exec_errors[remaining_bin.name] = str(e)
                                            finalized_fuzzers.add(remaining_bin.name)
                                            run_children_exit_count += 1
                                    batch_should_stop = True
                                    break
                            except Exception as e:
                                run_exec_errors[bin_path.name] = str(e)
                                finalized_fuzzers.add(bin_path.name)
                                run_children_exit_count += 1
                                detected_kind, detected_idle = _capture_timeout_from_error(str(e))
                                if detected_kind and not run_error_kind:
                                    run_error_kind = detected_kind
                                    run_terminal_reason = detected_kind
                                    if detected_idle > 0:
                                        run_idle_seconds = detected_idle
                        if batch_should_stop:
                            pending_bins = []
                if stop_on_first_crash and any(run.crash_found for run in run_results.values()):
                    if not first_crash_fuzzer:
                        for crash_name, crash_run in run_results.items():
                            if crash_run.crash_found:
                                first_crash_fuzzer = crash_name
                                break
                    if not early_stop_reason and first_crash_fuzzer:
                        early_stop_reason = "first_crash_stop"
                    for skipped in pending_bins:
                        if skipped.name not in early_stopped_fuzzers:
                            early_stopped_fuzzers.append(skipped.name)
                    pending_bins = []
                    break
        finally:
            setattr(gen, "current_run_time_budget_sec", prev_run_budget)
            setattr(gen, "current_run_hard_timeout_sec", prev_run_hard_timeout)
            setattr(gen, "current_run_parallel_config_by_fuzzer", prev_run_parallel_cfg)

        _wf_log(cast(dict[str, Any], state), "run children exited, collecting results...")
        finalize_started = time.perf_counter()
        finalize_deadline = (
            finalize_started + float(finalize_timeout_sec) if finalize_timeout_sec > 0 else None
        )

        def _finalize_timed_out(stage: str) -> bool:
            nonlocal run_error_kind, run_terminal_reason, run_last_error
            if finalize_deadline is None:
                return False
            if time.perf_counter() <= finalize_deadline:
                return False
            run_error_kind = "run_finalize_timeout"
            run_terminal_reason = "run_finalize_timeout"
            run_last_error = f"run finalize timed out while {stage} (>{finalize_timeout_sec}s)"
            return True

        first_nonzero_rc = 0
        crash_candidates: list[tuple[str, Path, FuzzerRunResult]] = []

        def _make_run_detail_fallback(
            *,
            fuzzer_name: str,
            rc: int,
            run_error_kind_value: str,
            exception_kind: str,
            error: str,
        ) -> dict[str, Any]:
            return {
                "fuzzer": fuzzer_name,
                "rc": rc,
                "effective_rc": rc,
                "crash_found": False,
                "crash_evidence": "none",
                "run_error_kind": run_error_kind_value,
                "exception_kind": exception_kind,
                "error": error,
                "new_artifacts": [],
                "first_artifact": "",
                "final_cov": 0,
                "final_ft": 0,
                "final_iteration": 0,
                "final_execs_per_sec": 0,
                "final_rss_mb": 0,
                "final_corpus_files": 0,
                "final_corpus_size_bytes": 0,
                "corpus_files": 0,
                "corpus_size_bytes": 0,
                "seed_quality": {},
                "parallel_engine": str((current_run_parallel_cfg.get(fuzzer_name) or {}).get("parallel_engine") or "single"),
                "parallel_role": str((current_run_parallel_cfg.get(fuzzer_name) or {}).get("parallel_role") or "reserved"),
                "outer_slot": int((current_run_parallel_cfg.get(fuzzer_name) or {}).get("outer_slot") or 0),
                "inner_workers": int((current_run_parallel_cfg.get(fuzzer_name) or {}).get("inner_workers") or 1),
                "reload_enabled": bool((current_run_parallel_cfg.get(fuzzer_name) or {}).get("reload_enabled")),
            }

        for bin_path in bins:
            if _finalize_timed_out("collecting run details"):
                break
            fuzzer_name = bin_path.name
            if fuzzer_name not in finalized_fuzzers:
                continue
            exec_err = run_exec_errors.get(fuzzer_name, "")
            if exec_err:
                detail_kind = "run_exception"
                detail_rc = 1
                if not run_last_error:
                    run_last_error = f"fuzzer run crashed for {fuzzer_name}: {exec_err}"
                if not run_error_kind:
                    run_error_kind = "run_exception"
                detected_kind, detected_idle = _capture_timeout_from_error(exec_err)
                if detected_kind and not run_terminal_reason:
                    run_terminal_reason = detected_kind
                    if detected_idle > 0:
                        run_idle_seconds = detected_idle
                    detail_kind = detected_kind
                    detail_rc = 124 if detected_kind in {"run_timeout", "run_idle_timeout"} else 1
                if first_nonzero_rc == 0:
                    first_nonzero_rc = detail_rc
                run_details.append(
                    _make_run_detail_fallback(
                        fuzzer_name=fuzzer_name,
                        rc=detail_rc,
                        run_error_kind_value=detail_kind,
                        exception_kind=detail_kind,
                        error=exec_err,
                    )
                )
                continue

            run = run_results.get(fuzzer_name)
            if run is None:
                # Defensive fallback: if a future completed without result/exception.
                if not run_last_error:
                    run_last_error = f"missing run result for {fuzzer_name}"
                if not run_error_kind:
                    run_error_kind = "run_exception"
                if first_nonzero_rc == 0:
                    first_nonzero_rc = 1
                run_details.append(
                    _make_run_detail_fallback(
                        fuzzer_name=fuzzer_name,
                        rc=1,
                        run_error_kind_value="run_exception",
                        exception_kind="run_exception",
                        error="missing run result",
                    )
                )
                continue

            rc = int(run.rc)
            if first_nonzero_rc == 0 and rc != 0:
                first_nonzero_rc = rc
            if not run_error_kind and run.run_error_kind:
                run_error_kind = run.run_error_kind
            if run.run_error_kind in {"run_idle_timeout", "run_timeout"} and not run_terminal_reason:
                run_terminal_reason = run.run_error_kind
                if run.run_error_kind == "run_idle_timeout":
                    run_idle_seconds = idle_timeout_sec
            if not run_terminal_reason and str(run.terminal_reason or "").strip():
                run_terminal_reason = str(run.terminal_reason).strip()
                if run.terminal_reason == "coverage_plateau":
                    run_idle_seconds = int(run.plateau_idle_seconds or 0)

            run_details.append(
                {
                    "fuzzer": fuzzer_name,
                    "target_name": str(
                        _execution_target_identity(execution_target_by_fuzzer.get(fuzzer_name) or {}).get("target_name")
                        or ""
                    ),
                    "target_api": str(
                        _execution_target_identity(execution_target_by_fuzzer.get(fuzzer_name) or {}).get("target_api")
                        or ""
                    ),
                    "target_type": str(
                        _execution_target_identity(execution_target_by_fuzzer.get(fuzzer_name) or {}).get("target_type")
                        or ""
                    ),
                    "rc": rc,
                    "effective_rc": rc,
                    "crash_found": bool(run.crash_found),
                    "crash_evidence": run.crash_evidence,
                    "run_error_kind": run.run_error_kind,
                    "exception_kind": "",
                    "error": run.error or "",
                    "log_tail": run.log_tail or "",
                    "new_artifacts": [str(p) for p in (run.new_artifacts or [])],
                    "first_artifact": run.first_artifact or "",
                    "final_cov": int(run.final_cov),
                    "final_ft": int(run.final_ft),
                    "final_iteration": int(run.final_iteration),
                    "final_execs_per_sec": int(run.final_execs_per_sec),
                    "final_rss_mb": int(run.final_rss_mb),
                    "final_corpus_files": int(run.final_corpus_files),
                    "final_corpus_size_bytes": int(run.final_corpus_size_bytes),
                    "corpus_files": int(run.corpus_files),
                    "corpus_size_bytes": int(run.corpus_size_bytes),
                    "terminal_reason": str(run.terminal_reason or ""),
                    "plateau_detected": bool(run.plateau_detected),
                    "plateau_idle_seconds": int(run.plateau_idle_seconds or 0),
                    "plateau_hit_count": int(run.plateau_hit_count or 0),
                    "plateau_last_hit_at": float(run.plateau_last_hit_at or 0.0),
                    "progress_sample_file": str(run.progress_sample_file or ""),
                    "seed_quality": dict(run.seed_quality or {}),
                    "parallel_engine": str(run.parallel_engine or "single"),
                    "parallel_role": str(run.parallel_role or "reserved"),
                    "outer_slot": int(run.outer_slot or 0),
                    "inner_workers": int(run.inner_workers or 1),
                    "reload_enabled": bool(run.reload_enabled),
                }
            )
            if run.error and not run_last_error:
                run_last_error = run.error
            if run.crash_found and run.first_artifact:
                crash_candidates.append((fuzzer_name, Path(run.first_artifact), run))

        # Detect "no real progress" runs so the workflow can repair instead of silently
        # ending in a false-success/false-running state.
        if not crash_candidates and not run_last_error:
            no_progress_fuzzers: list[str] = []
            seed_rejected_fuzzers: list[str] = []
            stalled_after_cov_fuzzers: list[str] = []
            for detail in run_details:
                if bool(detail.get("crash_found")):
                    continue
                if int(detail.get("rc") or 0) != 0:
                    continue
                final_execs = int(detail.get("final_execs_per_sec") or 0)
                final_cov = int(detail.get("final_cov") or 0)
                final_ft = int(detail.get("final_ft") or 0)
                final_corpus_files = int(detail.get("final_corpus_files") or 0)
                final_corpus_size_bytes = int(detail.get("final_corpus_size_bytes") or 0)
                log_or_err = f"{detail.get('error') or ''}\n{detail.get('log_tail') or ''}".lower()
                warned_no_progress = (
                    "no interesting inputs were found so far" in log_or_err
                    or "inited exec/s: 0" in log_or_err
                    or "exec/s: 0" in log_or_err
                )
                if warned_no_progress and final_execs > 0 and final_cov <= 0 and final_ft <= 0 and (
                    final_corpus_files <= 1 or final_corpus_size_bytes <= 1
                ):
                    seed_rejected_fuzzers.append(str(detail.get("fuzzer") or "unknown"))
                if final_execs <= 0 and warned_no_progress:
                    no_progress_fuzzers.append(str(detail.get("fuzzer") or "unknown"))
                if warned_no_progress and final_execs > 0 and (final_cov > 0 or final_ft > 0):
                    stalled_after_cov_fuzzers.append(str(detail.get("fuzzer") or "unknown"))
            if seed_rejected_fuzzers:
                run_error_kind = "run_seed_rejected"
                joined = ", ".join(seed_rejected_fuzzers[:5])
                run_last_error = (
                    "fuzzer inputs were likely rejected by target parser "
                    f"(no interesting inputs, zero cov/ft, tiny corpus): {joined}"
                )
            if no_progress_fuzzers:
                if not run_error_kind:
                    run_error_kind = "run_no_progress"
                    joined = ", ".join(no_progress_fuzzers[:5])
                    run_last_error = (
                        "fuzzer run made no measurable progress "
                        f"(exec/s=0 with no-interesting-input warnings): {joined}"
                    )
            if stalled_after_cov_fuzzers:
                if not run_error_kind:
                    run_error_kind = "run_stalled_after_coverage"
                    joined = ", ".join(stalled_after_cov_fuzzers[:5])
                    run_last_error = (
                        "fuzzer stalled after finding initial coverage "
                        f"(exec/s=0 with cov>0 but no further progress): {joined}"
                    )

        # Auto-repair corrupted dict files on dict_parse_error so next build
        # regenerates them from scratch.
        if run_error_kind == "dict_parse_error":
            dict_dir = gen.fuzz_dir / "dict"
            if dict_dir.is_dir():
                for df in dict_dir.iterdir():
                    if df.suffix == ".dict":
                        try:
                            df.unlink()
                        except OSError:
                            pass

        if crash_candidates:
            if _finalize_timed_out("packaging crash artifacts"):
                crash_found = False
                run_rc = 1
                crash_evidence = "none"
            else:
                last_fuzzer, first, crash_run = crash_candidates[0]
                gen._analyze_and_package(last_fuzzer, first)
                crash_found = True
                last_artifact = str(first)
                run_rc = int(crash_run.rc)
                crash_evidence = crash_run.crash_evidence
        else:
            run_rc = first_nonzero_rc
            crash_evidence = "none"

        if crash_found:
            msg = "Fuzzing completed (crash found and packaged)."
        elif run_last_error:
            msg = "Fuzzing run failed."
        else:
            msg = "Fuzzing completed."

        seed_bootstrap_all = getattr(gen, "last_seed_bootstrap_by_fuzzer", {}) or {}

        def _first_seed_meta_list(*path: str) -> tuple[bool, list[str]]:
            for meta in seed_bootstrap_all.values():
                if not isinstance(meta, dict):
                    continue
                cur: Any = meta
                ok = True
                for key in path:
                    if not isinstance(cur, dict):
                        ok = False
                        break
                    cur = cur.get(key)
                if not ok:
                    continue
                if isinstance(cur, (list, tuple, set)):
                    return True, [str(v).strip() for v in cur if str(v).strip()]
            return False, []

        families_suggested_found, families_suggested_values = _first_seed_meta_list("seed_families_suggested")
        families_covered_found, families_covered_values = _first_seed_meta_list("seed_family_coverage", "covered")
        families_missing_found, families_missing_values = _first_seed_meta_list("seed_family_coverage", "missing")

        crash_signature = ""
        same_crash_repeats = 0
        if crash_found and last_fuzzer and last_artifact:
            crash_signature = _calc_crash_signature(last_fuzzer, last_artifact)
            same_crash_repeats = (prev_crash_repeats + 1) if (prev_crash_sig and crash_signature == prev_crash_sig) else 0

        timeout_signature = ""
        same_timeout_repeats = 0
        timeout_like_kinds = {"run_timeout", "run_idle_timeout", "run_finalize_timeout", "run_no_progress"}
        if run_error_kind in timeout_like_kinds:
            timeout_signature = _calc_timeout_signature(run_error_kind, run_details)
            same_timeout_repeats = (
                (prev_timeout_repeats + 1)
                if (prev_timeout_sig and timeout_signature == prev_timeout_sig)
                else 0
            )

        coverage_target_name = str(preferred_identity.get("target_name") or "").strip()
        coverage_target_api = str(preferred_identity.get("target_api") or "").strip()
        coverage_target_type = str(preferred_identity.get("target_type") or "").strip()
        if not coverage_target_type:
            coverage_target_type = _target_type_from_run_details(
                run_details,
                target_name=coverage_target_name,
                target_api=coverage_target_api,
                fuzzer_name=str(preferred_identity.get("expected_fuzzer_name") or ""),
            )
        coverage_seed_profile = str(preferred_identity.get("seed_profile") or last_seed_profile or "").strip()
        coverage_fuzzer_name = str(preferred_identity.get("expected_fuzzer_name") or coverage_target_name).strip()
        current_seed_quality = _seed_quality_from_run_details_for_target(
            run_details,
            dict(state.get("coverage_seed_quality") or {}),
            target_name=coverage_target_name,
            target_api=coverage_target_api,
            fuzzer_name=coverage_fuzzer_name,
        )
        current_quality_flags = _quality_flags_from_seed_quality(current_seed_quality)

        out = {
            **state,
            "last_step": "run",
            "last_error": run_last_error,
            "crash_found": crash_found,
            "run_rc": run_rc,
            "crash_evidence": crash_evidence,
            "run_error_kind": run_error_kind,
            "run_terminal_reason": run_terminal_reason,
            "run_idle_seconds": int(run_idle_seconds or 0),
            "run_children_exit_count": int(run_children_exit_count),
            "run_cancel_requested_count": int(run_cancel_requested_count),
            "run_cancel_effective_count": int(run_cancel_effective_count),
            "run_details": run_details,
            "run_batch_plan": run_batch_plan,
            "run_parallel_engine": parallel_engine,
            "run_parallel_outer": int(max_parallel),
            "run_parallel_inner": int(inner_workers),
            "run_parallel_cpu_budget": int(cpu_budget),
            "first_crash_fuzzer": first_crash_fuzzer,
            "early_stop_reason": early_stop_reason,
            "early_stopped_fuzzers": list(early_stopped_fuzzers),
            "last_crash_artifact": last_artifact,
            "last_fuzzer": last_fuzzer,
            "coverage_target_name": (
                coverage_target_name
                or str(state.get("synthesize_selected_target_name") or "").strip()
                or str(state.get("coverage_target_name") or "").strip()
            ),
            "coverage_target_api": (
                coverage_target_api
                or str(state.get("synthesize_observed_target_api") or "").strip()
                or str(state.get("synthesize_selected_target_api") or "").strip()
                or str(state.get("coverage_target_api") or "").strip()
                or str(state.get("selected_target_api") or "").strip()
            ),
            "coverage_target_type": coverage_target_type or str(state.get("coverage_target_type") or ""),
            "coverage_seed_profile": coverage_seed_profile,
            "coverage_seed_quality": current_seed_quality,
            "coverage_seed_families_suggested": list(
                families_suggested_values
                if families_suggested_found
                else list(state.get("coverage_seed_families_suggested") or [])
            ),
            "coverage_seed_families_covered": list(
                families_covered_values
                if families_covered_found
                else list(state.get("coverage_seed_families_covered") or [])
            ),
            "coverage_seed_families_missing": list(
                families_missing_values
                if families_missing_found
                else list(state.get("coverage_seed_families_missing") or [])
            ),
            "coverage_quality_flags": list(
                current_quality_flags
                if current_seed_quality
                else list(state.get("coverage_quality_flags") or [])
            ),
            "coverage_target_depth_score": int(state.get("coverage_target_depth_score") or 0),
            "coverage_target_depth_class": str(state.get("coverage_target_depth_class") or ""),
            "coverage_selection_bias_reason": str(state.get("coverage_selection_bias_reason") or ""),
            "coverage_corpus_sources": sorted(seed_sources),
            "coverage_seed_counts": seed_count_total,
            "coverage_seed_counts_raw": seed_count_raw_total,
            "coverage_seed_counts_filtered": seed_count_filtered_total,
            "coverage_seed_noise_rejected_count": seed_noise_rejected_count,
            "coverage_seed_generation_failed_fuzzers": list(seed_generation_failed_fuzzers),
            "coverage_seed_generation_error_by_fuzzer": dict(seed_generation_error_by_fuzzer),
            "coverage_seed_generation_failed_count": int(len(seed_generation_failed_fuzzers)),
            "coverage_seed_generation_skipped_reason": seed_generation_skipped_reason,
            "coverage_seed_generation_degraded": bool(
                seed_generation_failed_fuzzers
                or (not seed_generation_skipped_reason and int(seed_count_filtered_total.get("total") or 0) <= 1)
                or bool(current_seed_quality.get("cold_start_failure") or False)
            ),
            "coverage_missing_execution_targets": missing_execution_targets,
            "coverage_seed_family_coverage": seed_family_coverage_state,
            "coverage_repo_examples_filtered": repo_examples_filtered,
            "coverage_repo_examples_rejected_count": repo_examples_rejected_count,
            "coverage_repo_examples_accepted_count": repo_examples_accepted_count,
            "crash_signature": crash_signature,
            "crash_stack_signature": _last_stack_sig.get("stack_signature", ""),
            "crash_stack_type": _last_stack_sig.get("crash_type", ""),
            "crash_stack_top_frames": _last_stack_sig.get("top_frames", ""),
            "same_crash_repeats": same_crash_repeats,
            "crash_signature_dedup_hit": bool(same_crash_repeats > 0),
            "timeout_signature": timeout_signature,
            "same_timeout_repeats": same_timeout_repeats,
            "message": msg,
            "auto_stop_policy": auto_stop_policy,
            "auto_stop_blocked_reason": str(state.get("auto_stop_blocked_reason") or ""),
            "continuous_loop_count": int(state.get("continuous_loop_count") or 0),
        }
        if run_error_kind == "workflow_time_budget_exceeded":
            out["failed"] = True
            out["last_error"] = out.get("last_error") or "time budget exceeded during run phase"
            out["message"] = "workflow stopped (time budget exceeded)"
        if run_error_kind in {"run_idle_timeout", "run_timeout", "run_finalize_timeout"}:
            if run_error_kind == "run_idle_timeout":
                out["message"] = "run stalled (idle timeout), routing to plan-repair"
                if not out.get("last_error"):
                    out["last_error"] = f"run stalled: no output for >= {idle_timeout_sec}s"
            elif run_error_kind == "run_finalize_timeout":
                out["message"] = "run finalize timed out, routing to plan-repair"
            else:
                out["message"] = "run timed out, routing to plan-repair"
        if crash_found and same_crash_repeats >= max_same_crash_repeats:
            out["failed"] = True
            out["last_error"] = (
                "same crash signature repeated after crash-fix attempts "
                f"(repeats={same_crash_repeats + 1}, threshold={max_same_crash_repeats + 1})"
            )
            out["message"] = "workflow stopped (same crash repeated)"
        if run_error_kind in timeout_like_kinds and same_timeout_repeats >= max_same_timeout_repeats:
            if auto_stop_policy == "legacy_mixed":
                out["failed"] = True
                out["last_error"] = (
                    "same timeout/no-progress signature repeated "
                    f"(repeats={same_timeout_repeats + 1}, threshold={max_same_timeout_repeats + 1})"
                )
                out["message"] = "workflow stopped (same timeout/no-progress repeated)"
            else:
                out["auto_stop_blocked_reason"] = "same_timeout_repeats"
                out["continuous_loop_count"] = int(out.get("continuous_loop_count") or 0) + 1
                out["message"] = "same timeout/no-progress repeated; continue under hard_fail_only"
        if crash_found and last_fuzzer and last_artifact:
            _write_repro_context(
                gen.repo_root,
                repo_url=str(out.get("repo_url") or ""),
                last_fuzzer=last_fuzzer,
                last_crash_artifact=last_artifact,
                crash_signature=crash_signature,
                re_workspace_root=str(out.get("re_workspace_root") or ""),
            )
        quality_flags = list(out.get("coverage_quality_flags") or [])
        if bool(state.get("synthesize_target_drifted")):
            quality_flags.append("target_runtime_mismatch")
        if list(out.get("coverage_seed_families_missing") or []):
            quality_flags.append("seed_family_undercovered")
        if list(out.get("coverage_missing_execution_targets") or []):
            quality_flags.append("missing_execution_targets")
        raw_total = int((out.get("coverage_seed_counts_raw") or {}).get("total") or 0)
        noise_rejected = int(out.get("coverage_seed_noise_rejected_count") or 0)
        if noise_rejected > 0 and (raw_total <= 0 or float(noise_rejected) / float(max(raw_total, 1)) >= 0.25):
            quality_flags.append("seed_noise_high")
        observed_api = str(out.get("coverage_target_api") or "").lower()
        if observed_api in {"println", "fmt::println", "print", "fmt::print", "format", "fmt::format", "format_to", "fmt::format_to", "vformat", "fmt::vformat"}:
            quality_flags.append("generic_wrapper_fallback")
        out["coverage_quality_flags"] = sorted({flag for flag in quality_flags if flag})
        out["coverage_seed_feedback"] = _build_seed_feedback(cast(dict[str, Any], out))
        out["coverage_harness_feedback"] = _build_harness_feedback(cast(dict[str, Any], out))
        try:
            seed_feedback_path = gen.repo_root / "fuzz" / "seed_feedback.json"
            seed_feedback_path.parent.mkdir(parents=True, exist_ok=True)
            by_fuzzer: dict[str, Any] = {}
            for detail in run_details:
                fuzzer_name = str(detail.get("fuzzer") or "").strip()
                if not fuzzer_name:
                    continue
                seed_quality = dict(detail.get("seed_quality") or {})
                if not seed_quality:
                    continue
                by_fuzzer[fuzzer_name] = {
                    "seed_profile": str(seed_quality.get("seed_profile") or out.get("coverage_seed_profile") or ""),
                    "initial_inited_cov": int(seed_quality.get("initial_inited_cov") or 0),
                    "final_cov": int(seed_quality.get("final_cov") or 0),
                    "cov_delta": int(seed_quality.get("cov_delta") or 0),
                    "initial_inited_ft": int(seed_quality.get("initial_inited_ft") or 0),
                    "final_ft": int(seed_quality.get("final_ft") or 0),
                    "ft_delta": int(seed_quality.get("ft_delta") or 0),
                    "early_new_units_30s": int(seed_quality.get("early_new_units_30s") or 0),
                    "early_new_units_60s": int(seed_quality.get("early_new_units_60s") or 0),
                    "initial_corpus_files": int(seed_quality.get("initial_corpus_files") or 0),
                    "final_corpus_files": int(seed_quality.get("final_corpus_files") or 0),
                    "quality_flags": list(seed_quality.get("quality_flags") or []),
                    "missing_suggested_families": list(out.get("coverage_seed_families_missing") or []),
                    "attack_hint_total_count": int(seed_quality.get("attack_hint_total_count") or 0),
                    "attack_hint_covered_count": int(seed_quality.get("attack_hint_covered_count") or 0),
                    "attack_hint_missing_values": list(seed_quality.get("attack_hint_missing_values") or []),
                    "attack_hint_coverage_ratio": float(seed_quality.get("attack_hint_coverage_ratio") or 1.0),
                    "merge_retained_ratio_files": float(seed_quality.get("merge_retained_ratio_files") or 1.0),
                    "merge_retained_ratio_bytes": float(seed_quality.get("merge_retained_ratio_bytes") or 1.0),
                    "cold_start_failure": bool(seed_quality.get("cold_start_failure") or False),
                    "updated_at": int(time.time()),
                }
            failed_seed_gen = set(out.get("coverage_seed_generation_failed_fuzzers") or [])
            if failed_seed_gen:
                for fuzzer_name in sorted(failed_seed_gen):
                    item = dict(by_fuzzer.get(fuzzer_name) or {})
                    item["seed_generation_failed"] = True
                    item["seed_generation_error"] = str(
                        (out.get("coverage_seed_generation_error_by_fuzzer") or {}).get(fuzzer_name) or ""
                    )
                    item["updated_at"] = int(time.time())
                    by_fuzzer[fuzzer_name] = item
            seed_feedback_doc = {
                "version": 1,
                "updated_at": int(time.time()),
                "job_id": str(state.get("job_id") or ""),
                "repo_url": str(state.get("repo_url") or ""),
                "seed_generation_degraded": bool(out.get("coverage_seed_generation_degraded") or False),
                "seed_generation_failed_count": int(out.get("coverage_seed_generation_failed_count") or 0),
                "seed_generation_failed_fuzzers": list(out.get("coverage_seed_generation_failed_fuzzers") or []),
                "seed_generation_skipped_reason": str(out.get("coverage_seed_generation_skipped_reason") or ""),
                "by_fuzzer": by_fuzzer,
            }
            seed_feedback_path.write_text(
                json.dumps(seed_feedback_doc, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            out["coverage_seed_feedback_path"] = str(seed_feedback_path)
        except Exception as exc:
            _wf_log(cast(dict[str, Any], state), f"run: failed to write seed_feedback.json: {exc}")
        if run_error_kind:
            last_detail = run_details[-1] if run_details else {}
            attempt_index = int(state.get("repair_attempt_index") or 0) + 1
            out["repair_mode"] = True
            out["repair_origin_stage"] = "crash"
            out["repair_error_kind"] = run_error_kind
            out["repair_error_code"] = run_terminal_reason or run_error_kind
            out["repair_signature"] = str(timeout_signature or crash_signature or "")[:12]
            out["repair_stdout_tail"] = str(last_detail.get("stdout_tail") or "")
            out["repair_stderr_tail"] = str(last_detail.get("stderr_tail") or "")
            out["repair_attempt_index"] = attempt_index
            out["repair_strategy_force_change"] = False
            out["repair_error_digest"] = {
                "error_code": str(out.get("repair_error_code") or ""),
                "error_kind": run_error_kind,
                "signature": str(out.get("repair_signature") or ""),
                "failing_files": [],
                "symbols": [],
                "first_seen": int(time.time()),
                "latest_seen": int(time.time()),
                "top_trace": _extract_repair_top_trace(
                    str(out.get("last_error") or ""),
                    str(last_detail.get("stdout_tail") or ""),
                    str(last_detail.get("stderr_tail") or ""),
                ),
            }
            recent = list(state.get("repair_recent_attempts") or [])
            recent.append(
                {
                    "step": "run",
                    "origin": "crash",
                    "error_kind": run_error_kind,
                    "error_code": run_terminal_reason or run_error_kind,
                    "signature": out["repair_signature"],
                    "attempt_index": attempt_index,
                    "message": str(out.get("last_error") or "")[:512],
                }
            )
            out["repair_recent_attempts"] = recent[-5:]
        elif not crash_found:
            out["repair_mode"] = False
            out["repair_origin_stage"] = ""
            out["repair_error_kind"] = ""
            out["repair_error_code"] = ""
            out["repair_signature"] = ""
            out["repair_stdout_tail"] = ""
            out["repair_stderr_tail"] = ""
            out["repair_recent_attempts"] = []
            out["repair_error_digest"] = {}
            out["repair_attempt_index"] = 0
            out["repair_strategy_force_change"] = False
        choose_seed_snapshot = {
            "kind": "choose_seed",
            "seed_profile": str(last_seed_profile or ""),
            "seed_counts_raw": dict(seed_count_raw_total),
            "seed_counts_filtered": dict(seed_count_filtered_total),
            "seed_sources": sorted(seed_sources),
            "seed_generation_failed_fuzzers": list(seed_generation_failed_fuzzers),
            "seed_generation_skipped_reason": seed_generation_skipped_reason,
            "seed_generation_degraded": bool(out.get("coverage_seed_generation_degraded") or False),
            "quality_flags": list(out.get("coverage_quality_flags") or []),
            "degraded_reason": (
                "seed_generation_failed"
                if seed_generation_failed_fuzzers
                else ("low_filtered_seed_count" if int(seed_count_filtered_total.get("total") or 0) <= 1 else "")
            ),
        }
        out = _record_decision_trace(
            out,
            stage="run",
            tool="seed_pipeline",
            model=str(state.get("model") or ""),
            latency_ms=int(max(0.0, (time.perf_counter() - t0) * 1000.0)),
            error_kind=str(run_error_kind or ""),
            error_code=str(run_terminal_reason or run_error_kind or ""),
            retry_count=0,
            decision_snapshot=choose_seed_snapshot,
        )
        _wf_log(
            cast(dict[str, Any], out),
            (
                f"<- run ok crash_found={crash_found} rc={run_rc} evidence={crash_evidence} "
                f"same_crash_repeats={same_crash_repeats} same_timeout_repeats={same_timeout_repeats} "
                f"dt={_fmt_dt(time.perf_counter()-t0)}"
            ),
        )
        _emit_fuzz_metrics(cast(dict[str, Any], out))
        return out
    except Exception as e:
        out = {
            **state,
            "last_step": "run",
            "last_error": str(e),
            "message": "run failed",
            "repair_mode": True,
            "repair_origin_stage": "crash",
            "repair_error_kind": "run_exception",
            "repair_error_code": "run_exception",
            "repair_signature": "",
            "repair_stdout_tail": "",
            "repair_stderr_tail": "",
            "repair_recent_attempts": list(state.get("repair_recent_attempts") or []),
        }
        _wf_log(cast(dict[str, Any], out), f"<- run err={e} dt={_fmt_dt(time.perf_counter()-t0)}")
        return out
