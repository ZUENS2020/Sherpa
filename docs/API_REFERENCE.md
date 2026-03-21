# Sherpa API Reference

Last updated: 2026-03-21
Backend source of truth: `harness_generator/src/langchain_agent/main.py`

## 1. Base Information

- Base URL (dev): `https://dev.zuens2020.work`
- API prefix: `/api`
- Content type:
  - Request: `application/json` (except metrics endpoint)
  - Response: `application/json` (except `/api/metrics`)
- Auth: currently no API auth
- CORS: enabled for all origins (`allow_origins=["*"]`, methods `*`, headers `*`)

## 2. Common Semantics

### 2.1 Time fields

Most task/job payloads include Unix timestamps (seconds) and ISO UTC mirrors:

- `*_at`: float seconds
- `*_at_iso`: ISO-8601 UTC string (or `null`)

### 2.2 Status mapping used by task list

Internal status is normalized to uppercase API status:

- `queued` -> `QUEUED`
- `running`, `resuming` -> `RUNNING`
- `success` -> `SUCCESS`
- `resumed` -> `COMPLETED`
- `error` -> `ERROR`
- `recoverable`, `resume_failed` -> `FAILED`

### 2.3 Unlimited budget convention

For task submission budget fields:

- `-1` means unlimited (converted to internal `0`)
- `0` is also treated as unlimited internally

### 2.4 Frontend dynamic fields

The dashboard consumes these backend blocks directly:

- `overview`
- `telemetry`
- `execution.summary`
- `tasks_tab_metrics`

The backend does not inject demo placeholders. Missing metrics must be returned as `null` or a safe empty value.

## 3. Configuration APIs

## 3.1 GET `/api/config`

Returns current runtime config (public/sanitized):

- Secret fields are masked/blanked:
  - `openai_api_key`, `openrouter_api_key`, provider `api_key`
- Secret set flags are provided:
  - `openai_api_key_set`, `openrouter_api_key_set`, `api_key_set`
- Docker fields are forced for UI compatibility:
  - `fuzz_use_docker=false`, `fuzz_docker_image=""`

Example:

```json
{
  "openrouter_api_key": "",
  "openrouter_api_key_set": false,
  "openrouter_base_url": "https://openrouter.ai/api/v1",
  "openrouter_model": "anthropic/claude-3.5-sonnet",
  "openai_api_key": "",
  "openai_api_key_set": true,
  "openai_base_url": "https://api.minimaxi.com/anthropic/v1",
  "openai_model": "MiniMax-M2.7-highspeed",
  "opencode_model": "MiniMax-M2.7-highspeed",
  "opencode_providers": [
    {
      "name": "minimax",
      "enabled": true,
      "base_url": "https://api.minimaxi.com/anthropic/v1",
      "api_key": "",
      "api_key_set": true,
      "clear_api_key": false,
      "models": ["MiniMax-M2.7-highspeed"],
      "headers": {},
      "options": {}
    }
  ],
  "fuzz_time_budget": 900,
  "sherpa_run_unlimited_round_budget_sec": 7200,
  "fuzz_use_docker": false,
  "fuzz_docker_image": "",
  "oss_fuzz_dir": "",
  "sherpa_git_mirrors": "",
  "sherpa_docker_http_proxy": "",
  "sherpa_docker_https_proxy": "",
  "sherpa_docker_no_proxy": "",
  "sherpa_docker_proxy_host": "host.docker.internal",
  "apiBaseUrl": "",
  "version": 1
}
```

## 3.2 PUT `/api/config`

Supports two write modes.

1) Lightweight mode (for frontend settings only):  
Request contains only one of:

- `apiBaseUrl`
- `api_base_url`

Example:

```json
{ "apiBaseUrl": "https://dev.zuens2020.work" }
```

2) Full config mode:  
Request is merged into current config and validated by `WebPersistentConfig`.

Important behavior:

- `openai_api_key` / `openrouter_api_key` / provider API keys are not accepted from frontend payload as authoritative control fields in this route.
- `fuzz_time_budget` must be `>= 0`.
- `sherpa_run_unlimited_round_budget_sec` must be `>= 0`.
- config file is persisted to `config/web_config.json`.

Success response:

```json
{ "ok": true }
```

Error response:

- `400` invalid payload or invalid numeric constraints
- `500` persistence failure

## 4. OpenCode Provider Model Discovery APIs

## 4.1 GET `/api/opencode/providers/{provider}/models`

Path param:

- `provider`: currently normalized alias maps to `minimax`

Response:

```json
{
  "provider": "minimax",
  "models": ["MiniMax-M2.7-highspeed", "MiniMax-Text-01"],
  "source": "remote",
  "warning": ""
}
```

Notes:

- `source` is one of: `remote`, `builtin`, `none`
- `warning` is optional and may explain fallback reason

Errors:

- `400` provider empty/invalid
- `404` unsupported provider

## 4.2 POST `/api/opencode/providers/{provider}/models`

Same behavior as GET, but allows temporary override input:

```json
{
  "api_key": "optional",
  "base_url": "optional"
}
```

## 5. System and Health APIs

## 5.1 GET `/api/system`

Returns backend runtime status plus frontend-facing overview blocks.

Top-level fields:

- `ok`, `server_time`, `server_time_iso`, `uptime_sec`
- `jobs`: `{ total, queued, running, success, error }`
- `jobs_by_kind`: object map (e.g. `task`, `fuzz`)
- `workers`: `{ max }`
- `active_jobs`: concise active job list
- `logs`: log directory metadata
- `memory`: process/cgroup memory telemetry
- `config`: selected public runtime config

Frontend dynamic blocks:

- `overview`
- `telemetry`
- `execution.summary`
- `tasks_tab_metrics`

Metric contract:

- Values are computed from live job/runtime data only.
- `fastapi_gateway` is SLI-only and intentionally omits p95 / rps.
- `llm_token_usage` only uses real token accounting fields; if none exist, it is `null`.
- `tasks_tab_metrics.execs_per_sec` is derived from run-stage metrics/logs, not from task count.
- `repos_queued` tracks queued main tasks.
- If a metric has no reliable source at query time, it is returned as `null`.

Example (abridged):

```json
{
  "ok": true,
  "jobs": { "total": 4, "queued": 0, "running": 1, "success": 2, "error": 1 },
  "overview": {
    "avg_fuzz_time": "17m 8s",
    "active_agents": "1",
    "cluster_health": "80.0",
    "cluster_health_trend": "+5.0% ▲",
    "crash_triage_rate": "1",
    "crash_triage_rate_trend": "-1.0 ▼",
    "harnesses_synthesized": "2",
    "harnesses_synthesized_trend": "+1.0 ▲",
    "avg_coverage": null,
    "avg_coverage_trend": null
  },
  "telemetry": {
    "llm_token_usage": null,
    "llm_token_status": null,
    "k8s_pod_capacity": "61% CAP",
    "k8s_pod_status": "Normal",
    "fastapi_gateway": "100.00% SLI",
    "fastapi_status": "UP",
    "agent_health_matrix": [1, 0, 1],
    "performance_series": [{ "time": "00:00", "throughput": 33, "latency": 45 }]
  },
  "execution": {
    "summary": {
      "failure_rate": "33.33%",
      "fuzzing_jobs_24h": "4",
      "cluster_load_peak": "61%",
      "repos_queued": "0",
      "avg_triage_time_ms": null,
      "success_ratio": "66.67"
    }
  },
  "tasks_tab_metrics": {
    "total_jobs": "4",
    "execs_per_sec": "84.0",
    "success_rate": "66.7",
    "failed_tasks": "01"
  }
}
```

Field details:

- `overview.avg_fuzz_time`: rolling average fuzz runtime for successful or active jobs.
- `overview.active_agents`: current active main task count shown in the dashboard.
- `overview.cluster_health`: aggregate health score derived from failure rate, load and error ratio.
- `overview.crash_triage_rate`: recent crash-handling throughput signal.
- `overview.harnesses_synthesized`: synthesized harness count metric.
- `overview.avg_coverage`: average coverage percentage when available.
- `telemetry.llm_token_usage`: hourly token usage summary, only if real token fields exist.
- `telemetry.k8s_pod_capacity`: cluster resource pressure indicator.
- `telemetry.fastapi_gateway`: FastAPI gateway SLI only.
- `telemetry.fastapi_status`: gateway health text (`UP`, `DEGRADED`, `ERROR`).
- `telemetry.agent_health_matrix`: compact binary health grid used by the UI.
- `telemetry.performance_series`: time series rendered as throughput/latency charts.
- `execution.summary.failure_rate`: recent failure ratio for completed jobs.
- `execution.summary.fuzzing_jobs_24h`: 24h fuzz job count.
- `execution.summary.cluster_load_peak`: peak cluster load summary.
- `execution.summary.repos_queued`: queued main-task count.
- `tasks_tab_metrics.total_jobs`: task list total count.
- `tasks_tab_metrics.execs_per_sec`: run-stage exec/s rollup.
- `tasks_tab_metrics.success_rate`: task success ratio.
- `tasks_tab_metrics.failed_tasks`: failed task count.

## 5.2 GET `/api/metrics`

Prometheus plaintext endpoint.  
Response media type:

- `text/plain; version=0.0.4; charset=utf-8`

Includes metrics such as:

- `sherpa_jobs_total`
- `sherpa_jobs_status{status="..."}`
- `sherpa_jobs_failure_rate_window`
- `sherpa_process_resident_memory_bytes`
- `sherpa_cgroup_memory_*`

## 5.3 GET `/api/health`

Simple liveness:

```json
{ "ok": true }
```

## 6. Task APIs

## 6.1 POST `/api/task`

Creates one parent task job (`kind=task`), then submits child fuzz jobs for each item in `jobs`.

Request schema:

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
  "images": ["cpp", "java"],
  "force_build": false,
  "oss_fuzz_repo_url": "optional",
  "force_clone": false
}
```

Field notes:

- `code_url` required per job
- `jobs` is an array of parent task entries; each entry may spawn one or more fuzz children internally.
- Budget aliases:
  - `total_duration` -> `total_time_budget`
  - `single_duration` -> `run_time_budget`
  - `-1` means unlimited
- `max_tokens = 0` means no explicit token cap.
- `unlimited_round_limit` maps to the run-stage unlimited round cap used by the backend worker.
- In current k8s native mode, docker controls are kept for compatibility but not used as runtime selector.

Success response:

```json
{ "job_id": "task-parent-id", "status": "queued" }
```

## 6.2 GET `/api/task/{job_id}`

Returns a derived view for a parent task:

- If `job_id` not found: `{ "error": "job_not_found" }`
- If id is not a `task` kind: `{ "error": "job_not_task" }`
- Else returns parent task object with:
  - normalized `status` (`running/success/error`)
  - `children_status`: `{ total, queued, running, success, error }`
  - `children`: child job detailed snapshots
  - error metadata: `error_code`, `error_kind`, `error_signature`
  - phase metadata: `phase`, `runtime_mode`

Note: these two error cases are returned as JSON body (not HTTP 404/400).

## 6.3 POST `/api/task/{job_id}/resume`

Resumes a stopped/failed job.

- If `{job_id}` is fuzz kind: resumes that fuzz job
- Else task kind: tries to resume resumable child fuzz jobs

Response:

```json
{
  "job_id": "id",
  "kind": "task",
  "accepted": true,
  "reason": "resuming",
  "resume_attempts": 2,
  "status": "resuming"
}
```

## 6.4 POST `/api/task/{job_id}/stop`

Requests cancellation.

- Task kind: cancels parent + iterates children
- Fuzz kind: cancels single fuzz

Response:

```json
{
  "job_id": "id",
  "kind": "task",
  "accepted": true,
  "reason": "stopped",
  "status": "error",
  "details": {
    "accepted": true,
    "reason": "stopped"
  }
}
```

## 6.5 GET `/api/tasks?limit={n}`

Returns list for task board.

- Query `limit` default `50`, clamped to `[1, 200]`
- `repo` remains the backend job repo label; the frontend may display a derived repo name for readability.
- `stage` is the current phase shown in the task row.
- `active_child_status` mirrors the most recent child state and is useful when a parent task fan-outs to fuzz jobs.
- `progress` is a soft indicator; it is not a strict completion percentage.

Response:

```json
{
  "items": [
    {
      "job_id": "5c9f10ed1e384b86a655c3505cbc340b",
      "id": "5c9f10ed1e384b86a655c3505cbc340b",
      "status": "RUNNING",
      "status_raw": "running",
      "stage": "SYNTHESIZE",
      "repo": "https://github.com/user/repo",
      "created_at": 1710000000.0,
      "created_at_iso": "2026-03-20T12:00:00+00:00",
      "updated_at": 1710000050.0,
      "updated_at_iso": "2026-03-20T12:00:50+00:00",
      "started_at": 1710000005.0,
      "started_at_iso": "2026-03-20T12:00:05+00:00",
      "finished_at": null,
      "finished_at_iso": null,
      "error": null,
      "error_code": "",
      "error_kind": "",
      "error_signature": "",
      "phase": "synthesize",
      "runtime_mode": "native",
      "result": null,
      "children_status": { "total": 1, "queued": 0, "running": 1, "success": 0, "error": 0 },
      "child_count": 1,
      "progress": 66,
      "active_child_id": "child-fuzz-id",
      "active_child_status": "RUNNING",
      "active_child_phase": "synthesize"
    }
  ]
}
```

## 7. Service Root

## 7.1 GET `/`

Simple service descriptor:

```json
{
  "service": "sherpa-web",
  "role": "api-backend-only",
  "entrypoint": "Use Ingress at / for UI and /api/* for API"
}
```

## 8. Frontend Polling Recommendation

For dashboard-style frontend:

1. Poll `/api/tasks` every 5s for task table/state.
2. Poll `/api/system` every 5s for overview and metrics cards.
3. Use `status` (uppercase) + `stage` for main display.
4. Use `progress` only as soft indicator; derive strict completion from status.

## 9. cURL Examples

Submit task:

```bash
curl -X POST "https://dev.zuens2020.work/api/task" \
  -H "Content-Type: application/json" \
  -d '{
    "jobs": [{
      "code_url": "https://github.com/fmtlib/fmt.git",
      "total_duration": -1,
      "single_duration": -1,
      "max_tokens": 1000,
      "unlimited_round_limit": 7200
    }]
  }'
```

List tasks:

```bash
curl "https://dev.zuens2020.work/api/tasks?limit=50"
```

System status:

```bash
curl "https://dev.zuens2020.work/api/system"
```

Lightweight config save:

```bash
curl -X PUT "https://dev.zuens2020.work/api/config" \
  -H "Content-Type: application/json" \
  -d '{"apiBaseUrl":"https://dev.zuens2020.work"}'
```
