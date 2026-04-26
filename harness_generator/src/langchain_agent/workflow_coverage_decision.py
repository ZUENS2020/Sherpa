from __future__ import annotations

from typing import Any


def evaluate_coverage_decision(
    *,
    run_error_kind: str,
    crash_found: bool,
    failed: bool,
    recoverable_run_error_kinds: set[str],
    plateau_detected: bool,
    current_cov: int,
    prev_cov: int,
    current_ft: int,
    prev_ft: int,
    prev_plateau_streak: int,
    current_seed_profile: str,
    quality_flags: list[str],
    seed_families_missing: list[str],
    cold_start_failure: bool,
    seed_generation_degraded: bool,
    quality_score: float,
    cold_start_quality_threshold: float,
    early_new_units_30s: int,
    cold_start_early_units_threshold: int,
    merge_retained_low: bool,
    configured_parallel_units: int,
    parallel_cpu_budget: int,
    total_execs_per_sec: int,
    underutilized_execs_threshold: int,
    current_depth_class: str,
    coverage_replay_stage_success: bool,
    coverage_replay_manifest_fresh: bool,
    coverage_replay_queue_drained: bool,
    coverage_frontier_input_count: int,
    current_round: int,
    max_rounds: int,
    unlimited_rounds: bool,
) -> dict[str, Any]:
    plateau_no_gain = plateau_detected and current_cov <= prev_cov and current_ft <= prev_ft
    plateau_streak = (prev_plateau_streak + 1) if plateau_no_gain else (1 if plateau_detected else 0)
    requested_replan = bool(
        plateau_no_gain
        and plateau_streak >= 2
        and bool(current_seed_profile)
    )

    can_in_place = unlimited_rounds or (current_round < max_rounds)
    can_replan = unlimited_rounds or ((current_round + 1) < max_rounds)

    cold_start_seed_replan_triggered = bool(
        cold_start_failure
        and quality_score < cold_start_quality_threshold
        and early_new_units_30s <= cold_start_early_units_threshold
    )
    degraded_seed_replan_triggered = bool(
        seed_generation_degraded
        and (
            quality_score < cold_start_quality_threshold
            or early_new_units_30s <= cold_start_early_units_threshold
            or any(
                flag in quality_flags
                for flag in {
                    "low_early_yield",
                    "missing_execution_targets",
                }
            )
        )
    )
    seed_quality_issue = bool(
        any(
            flag in quality_flags
            for flag in {
                "low_retention",
                "low_early_yield",
                "high_homogeneity",
                "seed_noise_high",
                "missing_execution_targets",
            }
        )
        or cold_start_failure
        or merge_retained_low
    )
    resource_underutilized = bool(
        not seed_quality_issue
        and configured_parallel_units < int(parallel_cpu_budget * 0.7)
        and total_execs_per_sec < underutilized_execs_threshold
    )
    strategy_mismatch = bool(
        (not seed_quality_issue)
        and plateau_detected
        and total_execs_per_sec > 0
        and current_cov <= prev_cov
        and current_ft <= prev_ft
    )
    if seed_quality_issue:
        parallel_diagnosis_code = "seed_limited_priority"
        parallel_diagnosis = (
            "seed quality is the primary bottleneck; prioritize seed replan before parallelism changes"
        )
    elif resource_underutilized:
        parallel_diagnosis_code = "resource_underutilized"
        parallel_diagnosis = (
            "exec/s is low while configured parallel units are below cpu budget; "
            "increase outer or inner workers"
        )
    elif strategy_mismatch:
        parallel_diagnosis_code = "strategy_mismatch"
        parallel_diagnosis = (
            "exec/s is healthy but coverage/features are stalled; "
            "reduce parallelism and prioritize target/seed strategy changes"
        )
    else:
        parallel_diagnosis_code = "balanced"
        parallel_diagnosis = "parallelism looks balanced for current coverage signal"

    quality_degraded = bool(
        seed_quality_issue
        or requested_replan
    )
    quality_oracle = "quality_degraded" if quality_degraded else "ok"
    replay_frontier_ready = bool(
        coverage_replay_stage_success
        and coverage_replay_manifest_fresh
        and coverage_replay_queue_drained
    )

    if seed_quality_issue:
        coverage_bottleneck_kind = "seed_limited"
        if cold_start_failure:
            coverage_bottleneck_reason = "cold_start_failure"
        elif seed_generation_degraded:
            coverage_bottleneck_reason = "seed_generation_degraded"
        elif merge_retained_low:
            coverage_bottleneck_reason = "merge_retained_low"
        elif seed_families_missing:
            coverage_bottleneck_reason = "missing_seed_families"
        else:
            coverage_bottleneck_reason = "seed_quality_flags"
    elif requested_replan or (plateau_no_gain and str(current_depth_class or "").lower() == "shallow"):
        coverage_bottleneck_kind = "target_limited"
        coverage_bottleneck_reason = "target_plateau_or_shallow_depth"
    elif plateau_no_gain and replay_frontier_ready and int(coverage_frontier_input_count or 0) <= 0:
        coverage_bottleneck_kind = "harness_limited"
        coverage_bottleneck_reason = "plateau_without_seed_or_target_signal"
    elif plateau_no_gain and not replay_frontier_ready:
        coverage_bottleneck_kind = "none"
        coverage_bottleneck_reason = "replay_frontier_not_ready"
    else:
        coverage_bottleneck_kind = "none"
        coverage_bottleneck_reason = ""

    base_should_improve = (
        (not crash_found)
        and (not failed)
        and (
            (not run_error_kind)
            or (run_error_kind in recoverable_run_error_kinds)
        )
    )
    should_improve = False
    replan_required = False
    replan_reason = ""
    improve_mode = ""
    round_budget_exhausted = False
    stop_reason = ""

    if base_should_improve:
        if current_cov <= 0 and current_round > 0:
            should_improve = True
            replan_required = True
            improve_mode = "replan"
            replan_reason = "zero_coverage_force_replan"
        elif cold_start_seed_replan_triggered or degraded_seed_replan_triggered:
            if can_replan:
                should_improve = True
                replan_required = True
                improve_mode = "seed_replan"
                replan_reason = (
                    "seed_cold_start_failure"
                    if cold_start_seed_replan_triggered
                    else "seed_generation_degraded"
                )
            elif can_in_place:
                should_improve = True
                improve_mode = "in_place"
                replan_reason = (
                    "seed_cold_start_failure_fallback_in_place"
                    if cold_start_seed_replan_triggered
                    else "seed_generation_degraded_fallback_in_place"
                )
            else:
                round_budget_exhausted = True
                stop_reason = "coverage_loop_budget_exhausted"
        elif seed_quality_issue and can_in_place:
            should_improve = True
            improve_mode = "in_place"
            if cold_start_failure:
                replan_reason = "seed_cold_start_failure"
            elif merge_retained_low:
                replan_reason = "seed_merge_retained_low"
            else:
                replan_reason = "seed_quality_issue"
        elif requested_replan:
            if can_replan:
                should_improve = True
                replan_required = True
                improve_mode = "replan"
                replan_reason = (
                    "prefer_deeper_target"
                    if current_depth_class == "shallow"
                    else "stalled_current_target"
                )
            else:
                round_budget_exhausted = True
                stop_reason = "coverage_loop_budget_exhausted"
        elif can_in_place:
            should_improve = True
            improve_mode = "in_place"
        else:
            round_budget_exhausted = True
            stop_reason = "coverage_loop_budget_exhausted"

    return {
        "plateau_no_gain": plateau_no_gain,
        "plateau_streak": plateau_streak,
        "requested_replan": requested_replan,
        "cold_start_seed_replan_triggered": cold_start_seed_replan_triggered,
        "degraded_seed_replan_triggered": degraded_seed_replan_triggered,
        "seed_quality_issue": seed_quality_issue,
        "parallel_diagnosis_code": parallel_diagnosis_code,
        "parallel_diagnosis": parallel_diagnosis,
        "quality_degraded": quality_degraded,
        "quality_oracle": quality_oracle,
        "coverage_replay_frontier_ready": replay_frontier_ready,
        "coverage_bottleneck_kind": coverage_bottleneck_kind,
        "coverage_bottleneck_reason": coverage_bottleneck_reason,
        "should_improve": should_improve,
        "replan_required": replan_required,
        "improve_mode": improve_mode,
        "replan_reason": replan_reason,
        "round_budget_exhausted": round_budget_exhausted,
        "stop_reason": stop_reason,
    }
