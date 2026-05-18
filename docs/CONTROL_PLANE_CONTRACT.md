# TianHeng Control Plane Contract

## Purpose

TianHeng separates execution truth from AI strategy:

- `strict contract`: system-owned fields that may drive routing, build, run, replay, or retry decisions.
- `advisory contract`: agent-suggested fields that may influence strategy, but must be normalized before they can affect execution.
- `derived fields`: runtime-derived facts from repository inspection, build output, run output, replay, or crash artifacts.
- `normalized fields`: advisory inputs after system normalization.

The core rule is simple: raw AI output is never the final truth source for workflow routing or execution.

## Ownership Model

- `system-owned`: only the coordinator/runtime may finalize the field.
- `agent-suggested`: the agent may propose it, but the system normalizes and may drop or rewrite it.
- `runtime-derived`: produced from observed artifacts only.

## Canonical Objects

### `fuzz/context/control_context.json`

Purpose: scheduler and execution controls only.

| Field | Owner | Notes |
| --- | --- | --- |
| `schema_version` | system-owned | metadata |
| `updated_at` | system-owned | metadata |
| `job_id` | system-owned | metadata |
| `time_budget` | system-owned | routing/execution budget |
| `run_time_budget` | system-owned | routing/execution budget |
| `coverage_loop_max_rounds` | system-owned | loop control |
| `max_fix_rounds` | system-owned | loop control |
| `same_error_max_retries` | system-owned | retry control |
| `run_oom_retry_count` | system-owned | execution retry state |
| `run_rss_limit_mb_override` | system-owned | execution override |
| `run_parallel_fuzzers_override` | system-owned | execution override |
| `run_timeout_budget_sec_override` | system-owned | execution override |
| `target_node_name` | system-owned | placement hint |
| `resume_repo_root` | system-owned | resume control |
| `last_fuzzer` | system-owned | operational control only |
| `last_crash_artifact` | system-owned | operational control only |

Forbidden: business/advisory target reasoning, AI draft state, seed taxonomy details.

### `fuzz/context/workflow_context.json`

Purpose: workflow business state and decision snapshots.

| Field | Owner | Notes |
| --- | --- | --- |
| `coverage_target_name` | system-owned | workflow target identity; never replace with observed API |
| `coverage_target_api` | system-owned | code entry/API identity |
| `coverage_target_type` | system-owned | normalized target type |
| `coverage_seed_profile` | normalized field | normalized before persistence |
| `coverage_seed_families_suggested` | agent-suggested | empty list is valid |
| `coverage_seed_families_covered` | runtime-derived | from seed bootstrap/run metadata |
| `coverage_seed_families_missing` | runtime-derived | derived diff: suggested vs covered |
| `coverage_attempted_targets` | system-owned | compare by `coverage_target_name` |
| `coverage_should_improve` | system-owned | routing |
| `coverage_improve_mode` | system-owned | routing |
| `coverage_replan_required` | system-owned | routing |
| `coverage_seed_quality` | runtime-derived | quality diagnostics |
| `coverage_quality_flags` | runtime-derived | quality diagnostics |
| `run_details` | runtime-derived | per-fuzzer run facts |
| `latest_decision_snapshot` | system-owned | normalized decision snapshot |
| `latest_vuln_decision_snapshot` | system-owned | normalized vuln snapshot |
| `vuln_hunt_enabled` | system-owned | internal hunt subphase enabled |
| `vuln_hunt_iteration` | system-owned | hunt subphase invocation count |
| `vuln_hunt_active_candidate_id` | system-owned | top active candidate chosen from candidate worklist |
| `vuln_hunt_candidate_count` | runtime-derived | current `fuzz/vuln_candidates.json` candidate count |
| `vuln_hunt_degraded` | system-owned | hunt degradation flag |
| `vuln_hunt_last_reason` | system-owned | structured hunt degradation/reason text |
| `vuln_hunt_summary_path` | system-owned | path to `fuzz/vuln_hunt_summary.md` |
| `vuln_hunt_events_path` | runtime-derived | path to append-only hunt feedback events |
| `vuln_hunt_rerun_requested` | system-owned | coverage/crash feedback requested another hunt refresh |
| `prompt_render_degraded` | system-owned | degraded observability |
| `prompt_render_issue` | system-owned | degraded observability |

Forbidden: raw agent JSON blobs becoming routing truth without normalization.

### `fuzz/selected_targets.json`

Purpose: normalized ranked targets that the system may execute against.

| Field | Owner | Notes |
| --- | --- | --- |
| `target_name` / `target` / `name` | normalized field | target identity |
| `api` | normalized field | API/code entry |
| `target_type` | normalized field | normalized type |
| `seed_profile` | normalized field | normalized profile |
| `seed_families_suggested` | advisory output | empty is valid |
| `seed_families_optional` | advisory output | non-routing |
| `score_total` | system-owned | computed score |
| `score_breakdown` | system-owned | computed score |
| `security_score_breakdown` | system-owned | computed score |
| `runtime_viability` | normalized field | used for prioritization |
| `api_surface_exception` | system-owned | normalized exception record |

Source chain:

1. `targets.json` is candidate input.
2. `selected_targets.json` is normalized ranked output.
3. `execution_plan.json` must be generated from normalized selected targets only.

### `fuzz/vuln_candidates.json`

Purpose: vulnerability-hunt advisory worklist consumed by plan/materialize.

| Field | Owner | Notes |
| --- | --- | --- |
| `candidate_id` | system-owned | stable candidate identity |
| `target_api` / `api` | agent-suggested | normalized before target materialization |
| `target_name` / `name` | agent-suggested | normalized before target materialization |
| `source_path` / `line` | agent-suggested | evidence location |
| `risk_type` | agent-suggested | vulnerability category signal |
| `priority` | system-owned | normalized hunt priority |
| `vuln_likelihood` | agent-suggested | risk score input |
| `exploitability` | agent-suggested | risk score input |
| `reachability_confidence` | agent-suggested | risk score input |
| `detectability_confidence` | agent-suggested | fuzz validation feasibility |
| `evidence_ids` | agent-suggested | must reference `analysis_evidence.security_evidence[]` when available |
| `attack_hint` | agent-suggested | advisory seed/harness guidance |
| `validation_status` | system-owned | `pending`, `validating`, `validated`, `inconclusive`, `exhausted`, `cooling` |
| `attempt_count` | runtime-derived | validation attempts |
| `last_result` | runtime-derived | latest validation feedback |

### `fuzz/decision_trace.jsonl`

Purpose: audit trail for routing and ranking decisions.

| Field | Owner | Notes |
| --- | --- | --- |
| `stage` | system-owned | execution stage |
| `tool` | system-owned | decision producer |
| `model` | system-owned | model/runtime metadata |
| `latency_ms` | runtime-derived | observed latency |
| `token_usage` | runtime-derived | observed token usage |
| `error_kind` | system-owned | normalized |
| `error_code` | system-owned | normalized |
| `retry_count` | system-owned | normalized |
| `decision_snapshot` | system-owned | normalized snapshot, may include advisory references |

## Target And Seed Contract

### Target

- `coverage_target_name`: workflow target identity and fuzzer identity anchor.
- `coverage_target_api`: code entry/API identity only.
- Target comparisons and attempted-target exclusion must use `coverage_target_name`, not API string fallback.
- `selected_targets.json -> execution_plan.json -> workflow_context.coverage_target_*` must stay aligned.

### Seed

- `coverage_seed_profile` is a controlled enum.
- Non-parser targets must not retain `parser-*` profiles after normalization.
- `image` and `decoder` targets normalize to `decoder-binary`.
- `coverage_seed_families_suggested` may be `[]`; empty means "no advisory requirement", not "inherit previous".
- `coverage_seed_families_covered` comes only from runtime evidence.
- `coverage_seed_families_missing` comes only from derived diff; never inherit stale values across target switches.

## Normalization Boundaries

Normalization must happen at least here:

1. `plan` before selected target state is persisted.
2. `synthesize` after observed target alignment.
3. `run` after seed bootstrap/run metadata is merged.
4. context storage write path for workflow state.

## Advisory Plane

The agent may propose:

- harness shape
- seed strategy
- `seed_profile`
- `seed_families_suggested`
- dictionary hints
- attack hints
- vulnerability rationale

But advisory data must not directly rewrite:

- active workflow target identity
- stage routing
- retry control
- execution plan truth
- runtime-derived diagnostics

## Format-Domain Seed Taxonomy

`decoder-binary` family hints are advisory and format-bounded. Generic decoder/image families include:

- `magic_headers`
- `chunk_layout`
- `length_boundary_values`
- `checksum_crc_variants`
- `truncated_sections`
- `metadata_chunks`
- `compressed_payload_variants`

Format-specific additions are allowed, for example PNG:

- `png_signature`
- `png_chunk_order`
- `png_crc_variants`
- `png_ihdr_dimensions`
- `png_idat_payloads`
- `png_ancillary_chunks`

These guide seed generation and feedback only. They do not directly route stages.
