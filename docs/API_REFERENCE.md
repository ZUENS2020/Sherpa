# Sherpa API Reference

Last updated: 2026-03-22
Backend source of truth: `harness_generator/src/langchain_agent/main.py`

## 1. Base Information

- Base URL (dev): `https://dev.zuens2020.work/`
- API prefix: `/api`
- Content type:
  - Request: `application/json`（`/api/metrics` 除外）
  - Response: `application/json`（`/api/metrics` 返回 Prometheus 文本）
- Auth: 当前无 API 鉴权
- CORS: 允许全部来源

## 2. Common Semantics

### 2.1 Time fields

大部分 task/job payload 同时提供 Unix 秒时间戳和 ISO UTC 字符串：

- `*_at`: float seconds
- `*_at_iso`: ISO-8601 UTC string，或 `null`

### 2.2 Task status mapping

后台内部状态会规范化成大写 API 状态：

- `queued` -> `QUEUED`
- `running`, `resuming` -> `RUNNING`
- `success` -> `SUCCESS`
- `resumed` -> `COMPLETED`
- `error` -> `ERROR`
- `recoverable`, `resume_failed` -> `FAILED`

### 2.3 Unlimited budget convention

预算字段里：

- `-1` 表示无限
- `0` 也会在内部被视为无限

### 2.4 Frontend dynamic blocks

前端仪表盘直接消费以下后端块：

- `overview`
- `telemetry`
- `execution.summary`
- `tasks_tab_metrics`

如果某个指标当前没有可靠来源，后端返回 `null` 或安全空值，而不是 demo 占位数据。

## 3. Configuration APIs

### 3.1 GET `/api/config`

返回当前运行时配置的公开视图。

要点：

- secret 字段会被清空或掩码：
  - `openai_api_key`
  - `openrouter_api_key`
  - provider `api_key`
- secret 是否存在会通过 `*_set` 标记给前端
- `fuzz_use_docker`、`fuzz_docker_image` 会保留给 UI 兼容，但在当前 native k8s 模式下不是主要运行开关

### 3.2 PUT `/api/config`

支持两种写法：

1. 轻量模式，只更新前端设置：

```json
{ "apiBaseUrl": "https://dev.zuens2020.work" }
```

或：

```json
{ "api_base_url": "https://dev.zuens2020.work" }
```

2. 完整配置模式：

- 请求体会与当前配置合并后再校验
- `fuzz_time_budget >= 0`
- `sherpa_run_unlimited_round_budget_sec >= 0`
- secret 控制字段不会由前端作为最终可信来源

成功响应：

```json
{ "ok": true }
```

错误响应：

- `400`：payload 无效或数值约束不合法
- `500`：持久化失败

## 4. OpenCode Provider Model Discovery

### 4.1 GET `/api/opencode/providers/{provider}/models`

查询 provider 的可用模型列表。

常见返回：

```json
{
  "provider": "minimax",
  "models": ["MiniMax-M2.7-highspeed", "MiniMax-Text-01"],
  "source": "remote",
  "warning": ""
}
```

`source` 可能是：

- `remote`
- `builtin`
- `none`

### 4.2 POST `/api/opencode/providers/{provider}/models`

与 GET 行为相同，但允许临时覆盖：

```json
{
  "api_key": "optional",
  "base_url": "optional"
}
```

## 5. System and Health APIs

### 5.1 GET `/api/system`

返回后端运行状态与前端仪表盘用的动态块。

顶层字段通常包括：

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

#### 动态块语义

- `overview.avg_fuzz_time`：成功或活跃任务的平均 fuzz 时长
- `overview.active_agents`：当前活跃主任务数
- `overview.cluster_health`：综合健康分
- `overview.crash_triage_rate`：近期 crash triage 吞吐
- `overview.harnesses_synthesized`：已合成 harness 统计
- `overview.avg_coverage`：可用时的平均覆盖率
- `telemetry.llm_token_usage`：只使用真实 token 统计；如果没有真实字段，返回 `null`
- `telemetry.k8s_pod_capacity`：集群资源压力
- `telemetry.fastapi_gateway`：FastAPI 网关 SLI
- `telemetry.fastapi_status`：网关状态文本
- `telemetry.agent_health_matrix`：紧凑健康矩阵
- `telemetry.performance_series`：吞吐 / 延迟时序图
- `execution.summary.failure_rate`：近期失败率
- `execution.summary.fuzzing_jobs_24h`：24h fuzz 任务数
- `execution.summary.cluster_load_peak`：峰值负载
- `execution.summary.repos_queued`：队列中的主任务数
- `tasks_tab_metrics.total_jobs`：任务面板总数
- `tasks_tab_metrics.execs_per_sec`：run 阶段的 exec/s 汇总
- `tasks_tab_metrics.success_rate`：任务成功率
- `tasks_tab_metrics.failed_tasks`：失败任务数

### 5.2 GET `/api/metrics`

Prometheus plaintext 端点。

返回媒体类型：

- `text/plain; version=0.0.4; charset=utf-8`

常见指标：

- `sherpa_jobs_total`
- `sherpa_jobs_status{status="..."}`
- `sherpa_jobs_failure_rate_window`
- `sherpa_process_resident_memory_bytes`
- `sherpa_cgroup_memory_*`

### 5.3 GET `/api/health`

简单存活检查：

```json
{ "ok": true }
```

## 6. Task APIs

### 6.1 POST `/api/task`

创建一个 parent task（`kind=task`），再为每个 job 创建 child fuzz jobs。

#### Request schema

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

#### 约定

- `code_url` 是每个 job 必填
- `total_duration` 会映射到内部总预算字段
- `single_duration` 会映射到内部单轮预算字段
- `-1` 表示无限，后台会按无限预算处理
- `max_tokens=0` 表示不设置显式 token 上限
- `unlimited_round_limit` 是 run 阶段无限轮次的上限桥接字段

#### Success response

```json
{ "job_id": "task-parent-id", "status": "queued" }
```

### 6.2 GET `/api/task/{job_id}`

返回 parent task 视图。

#### Error cases

- `{ "error": "job_not_found" }`
- `{ "error": "job_not_task" }`

#### 成功返回

包含：

- `status`
- `children_status`
- `children`
- `phase`
- `runtime_mode`
- `error_code`
- `error_kind`
- `error_signature`

### 6.3 POST `/api/task/{job_id}/resume`

恢复暂停或失败任务。

- fuzz kind：恢复该 fuzz job
- task kind：尝试恢复可恢复的 child fuzz jobs

### 6.4 POST `/api/task/{job_id}/stop`

请求取消任务。

- task kind：取消 parent + 迭代 children
- fuzz kind：取消单个 fuzz

### 6.5 GET `/api/tasks`

返回任务列表，供任务面板轮询。

#### Query

- `limit`：默认 `50`，会被 clamp 到 `[1, 200]`

#### 单项字段

- `job_id` / `id`
- `repo`
- `status`
- `status_raw`
- `stage`
- `phase`
- `progress`
- `children_status`
- `active_child_status`
- `active_child_phase`

`repo` 只是后端保存的仓库标签，前端可以根据需要显示仓库名或 URL 的派生值。

#### 典型响应

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

### GET `/`

简单服务描述：

```json
{
  "service": "sherpa-web",
  "role": "api-backend-only",
  "entrypoint": "Use Ingress at / for UI and /api/* for API"
}
```

## 8. Frontend Polling Recommendation

对于仪表盘式前端：

1. 每 5 秒轮询 `/api/tasks`
2. 每 5 秒轮询 `/api/system`
3. 主列表使用 `status`（大写）+ `stage`
4. `progress` 只作为软指标，不要把它当成严格完成度

## 9. cURL Examples

### 提交任务

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

### 查看任务列表

```bash
curl "https://dev.zuens2020.work/api/tasks?limit=50"
```

### 查看系统状态

```bash
curl "https://dev.zuens2020.work/api/system"
```
