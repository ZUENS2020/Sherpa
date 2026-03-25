# Sherpa API 参考

最后更新：2026-03-24
事实来源：[`harness_generator/src/langchain_agent/main.py`](../harness_generator/src/langchain_agent/main.py)

本文档只描述当前真实实现，不描述理想化契约。

## 1. 基础信息

- dev 基础地址：`https://dev.zuens2020.work`
- API 前缀：`/api`
- 返回格式：默认 JSON
- 鉴权：当前实现未暴露独立 API 鉴权层
- CORS：当前部署口径为全开放

## 2. 通用规则

### 状态归一化

任务相关接口会把内部状态归一成大写值：

- `QUEUED`
- `RUNNING`
- `SUCCESS`
- `COMPLETED`
- `FAILED`
- `ERROR`

### 时间字段

不少返回同时提供：

- 原始时间戳
- ISO 字符串

### 无限预算语义

当前实现里，时间预算类字段的无限意图通常会被桥接为 `0`。

## 3. 配置 API

### GET `/api/config`

返回当前运行时配置的持久化视图。

说明：

- secret 字段会被隐藏或清空
- `*_set` 风格字段表示对应 secret 是否已存在
- 部分 Docker 相关字段仍保留用于兼容

### PUT `/api/config`

支持轻量前端配置更新和完整配置更新。

#### 轻量更新

```json
{
  "apiBaseUrl": "https://dev.zuens2020.work"
}
```

或：

```json
{
  "api_base_url": "https://dev.zuens2020.work"
}
```

#### 完整更新

后端会把请求与现有配置合并后保存。

当前代码中能直接看到的校验示例包括：

- `fuzz_time_budget >= 0`
- `sherpa_run_unlimited_round_budget_sec >= 0`

成功返回：

```json
{ "ok": true }
```

## 4. 系统 API

### GET `/api/system`

返回系统级运行态与仪表盘聚合数据。

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

当前字段：

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

当前字段：

- `llm_token_usage`
- `llm_token_status`
- `k8s_pod_capacity`
- `k8s_pod_status`
- `fastapi_gateway`
- `fastapi_status`
- `agent_health_matrix`
- `performance_series`

重要说明：

- `llm_token_usage` 只应基于真实 token 数据；如果没有可用 token 字段，应显示为空或占位，不要用 `max_tokens` 估算。

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

`execs_per_sec` 来源于近期任务的真实执行速率聚合，而不是静态配置值。

### GET `/api/metrics`

Prometheus 文本指标端点。

媒体类型：

- `text/plain; version=0.0.4; charset=utf-8`

### GET `/api/health`

简单存活探针：

```json
{ "ok": true }
```

## 5. 任务 API

### POST `/api/task`

创建一个或多个父任务，并为每个 job 派生子 fuzz job。

请求体示例：

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

字段说明：

- `code_url` 是最关键字段
- `total_duration` / `single_duration` 是前端兼容别名
- `unlimited_round_limit` 会被桥接到运行时预算语义
- `max_tokens=0` 表示没有显式 token 上限

成功返回：

```json
{
  "job_id": "parent-task-id",
  "status": "queued"
}
```

### GET `/api/task/{job_id}`

返回父任务视图。

常见错误：

```json
{ "error": "job_not_found" }
```

```json
{ "error": "job_not_task" }
```

父任务视图会包含任务级状态以及子任务聚合信息，例如：

- `status`
- `children_status`
- `active_child_status`
- `error_code`
- `repo`
- `created_at`
- `updated_at`

### POST `/api/task/{job_id}/resume`

恢复一个暂停或失败后可继续的任务。

### POST `/api/task/{job_id}/stop`

停止任务。

### GET `/api/tasks`

返回任务列表，供 Tasks 面板轮询。

列表项会做前端友好归一化，常见字段包括：

- `job_id`
- `id`
- `repo`
- `status`
- `stage`
- `active_child_status`
- `progress`

说明：

- `status` 已归一成前端可直接消费的大写枚举
- `stage` 优先显示当前 active child stage
- `progress` 是任务进度百分比，通常只在运行中有意义

## 6. 前端消费建议

前端应优先使用：

- `GET /api/system` 做 Overview / Tasks 顶部总览
- `GET /api/tasks` 做任务表格
- `GET /api/task/{job_id}` 做任务详情页

不要用静态文案模拟动态指标。

## 7. 额外路由

当前后端还暴露：

- `GET /`
- `GET /api/opencode/providers/{provider}/models`
- `POST /api/opencode/providers/{provider}/models`

这些路由是实现的一部分，但前端联调通常优先关注前述核心 API。
