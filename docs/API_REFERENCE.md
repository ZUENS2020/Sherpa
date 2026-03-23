# Sherpa API Reference

Last updated: 2026-03-23
Backend source of truth: `harness_generator/src/langchain_agent/main.py`

## 1. Base Information

- dev base URL: `https://dev.zuens2020.work`
- API prefix: `/api`
- request/response content type: JSON unless otherwise noted
- authentication: no API auth layer is currently documented in code
- CORS: allow-all

## 2. Common Semantics

### Status normalization

Task-facing APIs normalize internal statuses into uppercase values:

- `QUEUED`
- `RUNNING`
- `SUCCESS`
- `COMPLETED`
- `FAILED`
- `ERROR`

### Time fields

Most task payloads expose both raw timestamps and ISO strings:

- `*_at`
- `*_at_iso`

### Unlimited budgets

Current conventions:

- `-1` is accepted by the request layer as unlimited intent
- `0` is used internally as unlimited for time-budget style fields

## 3. Configuration APIs

### GET `/api/config`

Returns the current persisted runtime-facing configuration view.

Notes:

- secret fields are hidden or blanked
- `*_set` style flags indicate whether secrets exist
- some Docker-related fields still exist for compatibility, but current staged K8s runtime is native-oriented

### PUT `/api/config`

Supported forms:

#### Lightweight frontend update

```json
{
  "apiBaseUrl": "https://dev.zuens2020.work"
}
```

or

```json
{
  "api_base_url": "https://dev.zuens2020.work"
}
```

#### Full config update

The backend merges the request with current config, validates it, preserves provider secret ownership, and persists the result.

Validation rules visible in code:

- `fuzz_time_budget >= 0`
- `sherpa_run_unlimited_round_budget_sec >= 0`

Success response:

```json
{ "ok": true }
```

## 4. System APIs

### GET `/api/system`

Returns system-wide runtime and dashboard aggregates.

Top-level blocks:

- `ok`
- `server_time`
- `server_time_iso`
- `uptime_sec`
- `jobs`
- `jobs_by_kind`
- `workers`
- `active_jobs`
- `logs`
- `memory`
- `config`
- `overview`
- `telemetry`
- `execution`
- `tasks_tab_metrics`

#### `overview`

Current fields in code:

- `avg_fuzz_time`
- `active_agents`
- `cluster_health`
- `cluster_health_trend`
- `crash_triage_rate`
- `crash_triage_rate_trend`
- `harnesses_synthesized`
- `harnesses_synthesized_trend`
- `avg_coverage`
- `avg_coverage_trend`
- `main_tasks_running`
- `main_tasks_queued`
- `child_jobs_running`
- `child_jobs_queued`

#### `telemetry`

Current fields in code:

- `llm_token_usage`
- `llm_token_status`
- `k8s_pod_capacity`
- `k8s_pod_status`
- `fastapi_gateway`
- `fastapi_status`
- `agent_health_matrix`
- `performance_series`

Important note:

- `llm_token_usage` only uses real token-derived job data; if no usable token field exists, it may be `null`

#### `execution.summary`

Current fields:

- `failure_rate`
- `fuzzing_jobs_24h`
- `cluster_load_peak`
- `repos_queued`
- `avg_triage_time_ms`
- `success_ratio`
- `main_tasks_running`
- `main_tasks_queued`
- `child_jobs_running`
- `child_jobs_queued`

#### `tasks_tab_metrics`

Current fields:

- `total_jobs`
- `execs_per_sec`
- `success_rate`
- `failed_tasks`

`execs_per_sec` is derived from recent run-stage exec-rate aggregation, not from a static config value.

### GET `/api/metrics`

Prometheus plaintext endpoint.

Media type:

- `text/plain; version=0.0.4; charset=utf-8`

### GET `/api/health`

Simple liveness endpoint:

```json
{ "ok": true }
```

## 5. Task APIs

### POST `/api/task`

Creates one parent task (`kind=task`) and one child fuzz job for each submitted job entry.

Request shape:

```json
{
  "jobs": [
    {
      "code_url": "https://github.com/owner/repo",
      "email": "optional",
      "model": "optional",
      "temperature": 0.5,
      "timeout": 10,
      "max_tokens": 0,
      "time_budget": 900,
      "total_time_budget": 900,
      "run_time_budget": 900,
      "total_duration": 900,
      "single_duration": 900,
      "unlimited_round_limit": 7200,
      "docker": false,
      "docker_image": ""
    }
  ],
  "auto_init": true,
  "build_images": true,
  "images": ["cpp"],
  "force_build": false,
  "oss_fuzz_repo_url": "optional",
  "force_clone": false
}
```

Behavior notes:

- `code_url` is the critical per-job field
- `total_duration` and `single_duration` are compatibility aliases used by the frontend
- `unlimited_round_limit` is accepted and bridged into runtime budget semantics
- `max_tokens=0` means no explicit token cap

Success response:

```json
{
  "job_id": "parent-task-id",
  "status": "queued"
}
```

### GET `/api/task/{job_id}`

Returns the parent-task view if `job_id` is a task.

Possible error responses:

```json
{ "error": "job_not_found" }
```

```json
{ "error": "job_not_task" }
```

Current response content includes aggregated child state, for example:

- `status`
- `children_status`
- `children`
- `phase`
- `runtime_mode`
- `error_code`
- `error_kind`
- `error_signature`

### POST `/api/task/{job_id}/resume`

Resumes a task or fuzz job depending on the stored `kind`.

Current response includes:

- `job_id`
- `kind`
- `accepted`
- `reason`
- `resume_attempts`
- `status`

### POST `/api/task/{job_id}/stop`

Requests cancellation of a task or fuzz job.

Current response includes:

- `job_id`
- `kind`
- `accepted`
- `reason`
- `status`
- `details`

## 6. Task List API

### GET `/api/tasks`

Returns parent task rows for the task table.

Query:

- `limit` default `50`, clamped to `[1, 200]`

Current item fields:

- `job_id`
- `id`
- `status`
- `status_raw`
- `stage`
- `repo`
- `repo_raw`
- `created_at`
- `created_at_iso`
- `updated_at`
- `updated_at_iso`
- `started_at`
- `started_at_iso`
- `finished_at`
- `finished_at_iso`
- `error`
- `error_code`
- `error_kind`
- `error_signature`
- `phase`
- `runtime_mode`
- `result`
- `children_status`
- `child_count`
- `progress`
- `active_child_id`
- `active_child_status`
- `active_child_phase`

Response shape:

```json
{
  "items": [
    {
      "job_id": "parent-task-id",
      "id": "parent-task-id",
      "status": "RUNNING",
      "stage": "BUILD",
      "repo": "batch",
      "progress": 66
    }
  ]
}
```

Notes:

- this endpoint lists parent tasks, not every fuzz child
- `repo` is a display label; frontend may derive a repository name from it
- `progress` is an aggregate task signal, not a strict workflow percentage

## 7. Frontend-Relevant Semantics

The local/Next frontends primarily depend on:

- `POST /api/task`
- `POST /api/task/{job_id}/stop`
- `GET /api/tasks`
- `GET /api/system`
- `PUT /api/config`

Recommended polling model:

1. poll `/api/tasks`
2. poll `/api/system`
3. fetch `/api/task/{job_id}` for task detail drill-down when needed

## 8. Source-of-Truth Reminder

This document describes current code behavior. If a field here and the implementation diverge, `harness_generator/src/langchain_agent/main.py` is authoritative.
