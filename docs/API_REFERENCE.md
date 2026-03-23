# Sherpa API 参考

最后更新：2026-03-23
后端事实来源：`harness_generator/src/langchain_agent/main.py`

## 1. 基础信息

- dev 基础地址：`https://dev.zuens2020.work`
- API 前缀：`/api`
- 请求 / 响应内容类型：除非特别说明，否则为 JSON
- 鉴权：当前代码中尚未记录独立的 API 鉴权层
- CORS：全开放

## 2. 通用语义

### 状态归一化

面向任务的 API 会把内部状态归一成大写值：

- `QUEUED`
- `RUNNING`
- `SUCCESS`
- `COMPLETED`
- `FAILED`
- `ERROR`

### 时间字段

大多数任务载荷同时暴露原始时间戳与 ISO 字符串：

- `*_at`
- `*_at_iso`

### 无限预算

当前约定：

- 请求层接受 `-1` 表示“无限”意图
- 对时间预算类字段，内部使用 `0` 表示无限

## 3. 配置 API

### GET `/api/config`

返回当前持久化的运行时配置视图。

说明：

- secret 字段会被隐藏或清空
- `*_set` 风格字段表示对应 secret 是否已存在
- 部分 Docker 相关字段仍保留用于兼容，但当前分阶段 K8s 运行时以原生执行为主

### PUT `/api/config`

支持以下形式。

#### 轻量前端更新

```json
{
  "apiBaseUrl": "https://dev.zuens2020.work"
}
```

或者：

```json
{
  "api_base_url": "https://dev.zuens2020.work"
}
```

#### 完整配置更新

后端会将请求与现有配置合并，完成校验、保留 provider secret 所有权后再持久化。

代码中可见的校验规则：

- `fuzz_time_budget >= 0`
- `sherpa_run_unlimited_round_budget_sec >= 0`

成功响应：

```json
{ "ok": true }
```

## 4. 系统 API

### GET `/api/system`

返回系统级运行时与仪表盘聚合信息。

顶层字段块：

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

代码中的当前字段：

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

代码中的当前字段：

- `llm_token_usage`
- `llm_token_status`
- `k8s_pod_capacity`
- `k8s_pod_status`
- `fastapi_gateway`
- `fastapi_status`
- `agent_health_matrix`
- `performance_series`

重要说明：

- `llm_token_usage` 仅使用真实 token 衍生的作业数据；如果没有可用 token 字段，则可能为 `null`

#### `execution.summary`

当前字段：

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

当前字段：

- `total_jobs`
- `execs_per_sec`
- `success_rate`
- `failed_tasks`

`execs_per_sec` 来源于近期 run 阶段执行速率聚合，而不是静态配置值。

### GET `/api/metrics`

Prometheus 纯文本指标端点。

媒体类型：

- `text/plain; version=0.0.4; charset=utf-8`

### GET `/api/health`

简单存活探针：

```json
{ "ok": true }
```

## 5. 任务 API

### POST `/api/task`

为每个提交的 job 条目创建一个父任务（`kind=task`）以及一个子 fuzz job。

请求格式：

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

行为说明：

- `code_url` 是每个 job 中最关键的字段
- `total_duration` 与 `single_duration` 是前端仍在使用的兼容别名
- `unlimited_round_limit` 会被接受并桥接到运行时预算语义
- `max_tokens=0` 表示没有显式 token 上限

成功响应：

```json
{
  "job_id": "parent-task-id",
  "status": "queued"
}
```

### GET `/api/task/{job_id}`

若 `job_id` 对应任务，则返回父任务视图。

可能的错误响应：

```json
{ "error": "job_not_found" }
```

```json
{ "error": "job_not_task" }
```

当前响应中会包含聚合后的子任务状态，例如：

- `status`
- `children_status`
- `children`
- `phase`
- `runtime_mode`
- `error_code`
- `error_kind`
- `error_signature`

### POST `/api/task/{job_id}/resume`

根据持久化的 `kind` 恢复任务或 fuzz job。

当前响应包含：

- `job_id`
- `kind`
- `accepted`
- `reason`
- `resume_attempts`
- `status`

### POST `/api/task/{job_id}/stop`

请求取消一个任务或 fuzz job。

当前响应包含：

- `job_id`
- `kind`
- `accepted`
- `reason`
- `status`
- `details`

## 6. 任务列表 API

### GET `/api/tasks`

返回任务表使用的父任务行。

查询参数：

- `limit` 默认 `50`，会被约束在 `[1, 200]`

当前条目字段：

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

响应格式：

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

说明：

- 该端点列出的是父任务，而不是每个子 fuzz job
- `repo` 是展示标签；前端可能会进一步从中推导仓库名
- `progress` 是聚合任务信号，并非严格的工作流百分比

## 7. 与前端相关的语义

本地 / Next 前端主要依赖以下接口：

- `POST /api/task`
- `POST /api/task/{job_id}/stop`
- `GET /api/tasks`
- `GET /api/system`
- `PUT /api/config`

推荐轮询模型：

1. 轮询 `/api/tasks`
2. 轮询 `/api/system`
3. 在需要查看详情时，再请求 `/api/task/{job_id}`

## 8. 事实来源提醒

本文档描述的是当前代码行为。如果这里的字段与实现不一致，应以 `harness_generator/src/langchain_agent/main.py` 为准。
