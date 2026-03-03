# zlib E2E 验收报告（SHE-58）

- 日期：2026-03-03
- 环境：`k8s/base`（Postgres-only）
- 仓库：`https://github.com/madler/zlib.git`

## 1. 验收目标

1. 成功场景：任务可完成 `plan -> synthesize -> build -> run`，并产出 `fuzz/out` 结果。
2. 失败场景：输入无效仓库时，任务进入 `error`，并返回可追踪错误信息。
3. 可观测性：`/api/task/{job_id}` 与 `/api/tasks` 暴露 `runtime_mode`、`phase`、`error_code`、`error_kind`、`error_signature`，`/api/metrics` 暴露核心计数指标。

## 2. 执行步骤

1. 提交成功任务（zlib）：
```bash
curl -sS -X POST http://127.0.0.1:18001/api/task \
  -H 'Content-Type: application/json' \
  -d '{"jobs":[{"code_url":"https://github.com/madler/zlib.git","total_time_budget":900,"run_time_budget":900,"max_tokens":1000}]}'
```
2. 提交失败任务（不存在仓库）：
```bash
curl -sS -X POST http://127.0.0.1:18001/api/task \
  -H 'Content-Type: application/json' \
  -d '{"jobs":[{"code_url":"https://github.com/example/not-exist-repo.git","total_time_budget":300,"run_time_budget":120,"max_tokens":200}]}'
```
3. 轮询详情与列表：
```bash
curl -sS http://127.0.0.1:18001/api/task/<job_id>
curl -sS http://127.0.0.1:18001/api/tasks?limit=20
```
4. 拉取指标：
```bash
curl -sS http://127.0.0.1:18001/api/metrics
```

## 3. 结果摘要

1. 成功场景：任务可进入终态，状态聚合正确，子任务信息可回放。
2. 失败场景：返回 `status=error`，并附带 `error` 与 `error_code`。
3. 字段可观测性：
   - 详情：`job_id`、`runtime_mode`、`phase`、`error_code`、`error_kind`、`error_signature`、`children_status`
   - 列表：`job_id`、`runtime_mode`、`phase`、`error_code`、`error_kind`、`error_signature`、`active_child_*`
4. 指标可观测性：
   - `sherpa_jobs_total`
   - `sherpa_jobs_status{status=*}`
   - `sherpa_jobs_recoverable_total`
   - `sherpa_jobs_failure_rate_window`

## 4. 风险与后续

1. 当前 metrics 为内存快照，实例重启后窗口统计会重置。
2. 建议下阶段将指标接入 Prometheus + Grafana，并追加告警策略（失败率阈值、任务积压阈值）。
