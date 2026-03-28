# Sherpa API Reference

本文档以当前分支代码实现为准，覆盖前端联调所需的主要 API。  
后端入口实现位于 `harness_generator/src/langchain_agent/main.py`。

## 1. 通用说明

- 基础前缀：`/api`
- 数据格式：`application/json`
- CORS：当前服务端允许所有来源（`*`）
- 时间字段：主用 Unix 时间戳（秒，`float`），部分接口附带 `*_iso` 字段（ISO8601）

任务模型说明：
- 主任务（`kind=task`）：批量提交入口，聚合多个子任务状态
- 子任务（`kind=fuzz`）：实际执行工作流的阶段任务

## 2. 配置接口

### GET `/api/config`

用途：读取当前 Web 配置（去敏后的公开配置）。

响应：`200 OK`

```json
{
  "fuzz_time_budget": 0,
  "fuzz_use_docker": false,
  "fuzz_docker_image": "",
  "openrouter_model": "MiniMax-M2.7-highspeed",
  "api_base_url": "https://dev.example.com"
}
```

说明：
- 实际返回字段以 `as_public_dict(cfg)` 为准。
- 敏感字段（API Key）不会以明文形式返回。

### PUT `/api/config`

用途：更新配置，支持轻量模式与完整模式。

请求体（轻量模式）：

```json
{
  "apiBaseUrl": "https://dev.zuens2020.work/",
  "sherpa_run_plateau_idle_growth_sec": 600
}
```

请求体（完整模式）：可传递配置对象。后端会做校验并保留受控字段。

响应：`200 OK`

```json
{
  "ok": true
}
```

约束：
- `fuzz_time_budget >= 0`（`0` 表示 unlimited）
- `sherpa_run_unlimited_round_budget_sec >= 0`（`0` 表示 fully unlimited）
- `sherpa_run_plateau_idle_growth_sec` 范围 `[30, 86400]`（秒）
- `apiBaseUrl` 与 `api_base_url` 均可用，最终统一存为 `api_base_url`
- 若 payload 非法返回 `400`

## 3. 任务提交与控制

### POST `/api/task`

用途：提交一批 fuzz 子任务，创建一个主任务。

请求体：

```json
{
  "jobs": [
    {
      "code_url": "https://github.com/fmtlib/fmt.git",
      "model": "MiniMax-M2.7-highspeed",
      "max_tokens": 0,
      "total_time_budget": 0,
      "run_time_budget": 0,
      "total_duration": -1,
      "single_duration": -1,
      "unlimited_round_limit": 7200
    }
  ],
  "auto_init": true,
  "build_images": true,
  "images": [],
  "force_build": false,
  "force_clone": false
}
```

关键兼容字段：
- `total_duration` -> `total_time_budget`
- `single_duration` -> `run_time_budget`
- `-1` 会被转换为 `0`（unlimited）

响应：`200 OK`

```json
{
  "job_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "status": "queued"
}
```

### GET `/api/task/{job_id}`

用途：查询主任务详情（聚合状态 + 子任务列表 +增强字段）。

响应：`200 OK`

```json
{
  "job_id": "xxxxxxxx",
  "kind": "task",
  "status": "running",
  "children_status": {
    "total": 3,
    "queued": 0,
    "running": 1,
    "success": 1,
    "error": 1
  },
  "children": [
    {
      "job_id": "child_xxx",
      "kind": "fuzz",
      "status": "running",
      "phase": "build",
      "runtime_mode": "native",
      "fuzz_total_execs_per_sec": 8123.5
    }
  ],
  "phase": "task",
  "runtime_mode": "native",
  "error": {
    "stage": "build",
    "kind": "build",
    "code": "missing_llvmfuzzer_entrypoint",
    "message": "build failed rc=1",
    "detail": "build failed rc=1",
    "signature": "9f23ab41c2de",
    "retryable": true,
    "terminal": false,
    "at": 1770000000
  },
  "error_code": "missing_llvmfuzzer_entrypoint",
  "error_kind": "build",
  "error_signature": "9f23ab41c2de"
}
```

增强字段（父子任务都可能出现）：
- 工作流与恢复：`k8s_phase`、`cancel_requested`、`workflow_active_step`、`workflow_last_step`、`recoverable`、`resume_attempts`、`resume_error_code`、`last_resume_reason` 等
- fuzz 指标：`fuzz_fuzzers`、`fuzz_max_cov`、`fuzz_max_ft`、`fuzz_total_execs_per_sec`、`fuzz_crash_found`、`fuzz_coverage_*`
- 错误字段：`error` 为统一错误对象；`error_code/error_kind/error_signature` 为兼容字段（deprecated）

### POST `/api/task/{job_id}/resume`

用途：手动恢复任务（task/fuzz 都支持）。

响应：`200 OK`

```json
{
  "job_id": "xxxxxxxx",
  "kind": "task",
  "accepted": true,
  "reason": "resume_started",
  "resume_attempts": 2,
  "status": "running"
}
```

### POST `/api/task/{job_id}/stop`

用途：手动停止任务（task/fuzz 都支持）。

响应：`200 OK`

```json
{
  "job_id": "xxxxxxxx",
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

## 4. 任务列表（前端 Tasks 面板）

### GET `/api/tasks?limit=50`

用途：分页/轮询任务列表，返回主任务视图（每项已聚合子任务状态）。

响应：`200 OK`

```json
{
  "items": [
    {
      "job_id": "xxxxxxxx",
      "id": "xxxxxxxx",
      "status": "RUNNING",
      "status_raw": "running",
      "stage": "BUILD",
      "repo": "fmt",
      "repo_raw": "https://github.com/fmtlib/fmt.git",
      "progress": 66,
      "active_child_id": "child_xxx",
      "active_child_status": "RUNNING",
      "active_child_phase": "build",
      "children_status": {
        "total": 3,
        "queued": 0,
        "running": 1,
        "success": 1,
        "error": 1
      },
      "error": {
        "stage": "build",
        "kind": "build",
        "code": "missing_llvmfuzzer_entrypoint",
        "message": "build failed rc=1",
        "detail": "build failed rc=1",
        "signature": "9f23ab41c2de",
        "retryable": true,
        "terminal": false,
        "at": 1770000000
      },
      "error_code": "missing_llvmfuzzer_entrypoint",
      "error_kind": "build",
      "error_signature": "9f23ab41c2de",
      "fuzz_fuzzers": {},
      "fuzz_max_cov": 0,
      "fuzz_max_ft": 0,
      "fuzz_total_execs_per_sec": 8123.5,
      "fuzz_crash_found": false,
      "fuzz_coverage_loop_round": 1,
      "fuzz_coverage_loop_max_rounds": 3,
      "fuzz_coverage_plateau_streak": 0,
      "fuzz_coverage_seed_profile": "parser-format",
      "fuzz_coverage_quality_flags": []
    }
  ]
}
```

字段口径：
- `status`：大写标准状态（`RUNNING/SUCCESS/COMPLETED/FAILED/ERROR/QUEUED` 兼容前端）
- `stage`：优先 active child 的阶段；无 active child 时回退主任务阶段
- `progress`：由子任务完成度估算（0-100）
- `repo`：用于 UI 展示的仓库名/短名；`repo_raw` 保留原始 URL
- `fuzz_*`：来自 active child（若存在），否则来自主任务自身
- `fuzz_coverage_plateau_streak`：按固定 30 秒无增长窗口统计的连续平台期轮次（`idle_no_growth=30s`）
- `error`：统一错误对象；`error_code/error_kind/error_signature` 为兼容字段（deprecated）

## 5. 系统总览（前端 Overview/Tasks 顶部）

### GET `/api/system`

用途：系统级统计与前端总览聚合。

响应：`200 OK`

```json
{
  "ok": true,
  "server_time": 1770000000.123,
  "server_time_iso": "2026-03-25T20:00:00+08:00",
  "uptime_sec": 12345.6,
  "jobs": {
    "total": 42,
    "queued": 2,
    "running": 5,
    "success": 31,
    "error": 4
  },
  "jobs_by_kind": {
    "task": 12,
    "fuzz": 30
  },
  "overview": {
    "avg_fuzz_time": "12m 34s",
    "active_agents": "3",
    "cluster_health": "97.2",
    "cluster_health_trend": "+0.8% ▲",
    "crash_triage_rate": "1",
    "harnesses_synthesized": "9",
    "avg_coverage": "61.23"
  },
  "telemetry": {
    "llm_token_usage": "2.4M / hr",
    "llm_token_status": "Active",
    "k8s_pod_capacity": "68% CAP",
    "k8s_pod_status": "Normal",
    "fastapi_gateway": "99.95% SLI",
    "fastapi_status": "UP",
    "agent_health_matrix": [1, 1, 0, 1],
    "performance_series": []
  },
  "execution": {
    "summary": {
      "failure_rate": "2.56%",
      "fuzzing_jobs_24h": "128",
      "cluster_load_peak": "68%",
      "repos_queued": "2",
      "success_ratio": "97.44"
    }
  },
  "tasks_tab_metrics": {
    "total_jobs": "12",
    "execs_per_sec": "84.2",
    "success_rate": "91.7",
    "failed_tasks": "1"
  }
}
```

字段口径说明：
- `llm_token_usage`：仅基于任务结果里的真实 token 字段聚合；无数据时为 `null`/`--` 状态
- `tasks_tab_metrics.execs_per_sec`：来自近期 fuzz job 的执行速率聚合（非配置常量）
- `overview.avg_coverage`：从 fuzz 结果中提取覆盖率相关字段并归一到 0-100 后求均值

## 6. 监控与健康

### GET `/api/metrics`

用途：Prometheus 文本指标。

响应类型：`text/plain; version=0.0.4; charset=utf-8`

包含指标示例：
- `sherpa_jobs_total`
- `sherpa_jobs_status{status="running"}`
- `sherpa_jobs_failure_rate_window`
- `sherpa_process_resident_memory_bytes`
- `sherpa_cgroup_memory_*`

### GET `/api/health`

用途：基础健康探针。

响应：

```json
{
  "ok": true
}
```

## 7. OpenCode Provider 模型接口（配置页可选）

### GET `/api/opencode/providers/{provider}/models`
### POST `/api/opencode/providers/{provider}/models`

用途：获取 provider 可用模型列表（支持带临时 `api_key/base_url` 查询）。

响应示例：

```json
{
  "provider": "openai",
  "models": ["gpt-5.4", "gpt-5.4-mini"],
  "source": "provider-config",
  "warning": null
}
```

## 8. 前端联调建议

- Tasks 列表优先使用 `/api/tasks`，避免前端自行拼装主子任务状态。
- 任务详情页面使用 `/api/task/{job_id}`，可直接读取 `children` 和 `fuzz_*`。
- Overview 与 Tasks 顶部统计都由 `/api/system` 提供，建议 5s 轮询一次。
- 如果只改前端 API Base URL，优先调用 `PUT /api/config` 轻量模式（`apiBaseUrl`）。
