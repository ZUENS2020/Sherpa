# Kubernetes 映射关系

本文档把 Sherpa 的概念映射到当前运行时对象。

## 1. 运行时对象映射

| 逻辑组件 | 当前实现 |
|---|---|
| API / 控制面 | backend Deployment / service |
| UI | frontend Deployment / service |
| 状态存储 | Postgres |
| 阶段执行 | Kubernetes Job |
| 任务输出根目录 | `/shared/output` |
| 聚合作业日志 | `/app/job-logs/jobs/*.log` |

## 2. 关键产物映射

| 路径或文件 | 含义 |
|---|---|
| `fuzz/PLAN.md` | 规划产物 |
| `fuzz/targets.json` | 候选目标 |
| `fuzz/selected_targets.json` | 带执行元数据的已选目标 |
| `fuzz/execution_plan.json` | 执行目标契约 |
| `fuzz/harness_index.json` | 目标到 harness 的映射 |
| `run_summary.json` | 任务级运行摘要 |
| `repro_context.json` | 崩溃复现上下文 |
| `stage-*.json` | 阶段结果记录 |

## 3. 当前主线路由

| 阶段 | 可能的下一阶段 |
|---|---|
| `plan` | `synthesize` |
| `synthesize` | `build` |
| `build` | `run` 或 `plan` |
| `run` | 根据结果进入 `coverage-analysis`、`crash-triage` 或 `plan` |
| `coverage-analysis` | `improve-harness` 或 `stop` |
| `improve-harness` | `build`、`plan` 或 `stop` |
| `crash-triage` | `fix-harness`、`re-build`、`plan` 或 `stop` |
| `fix-harness` | `build` 或 `plan` |
| `re-build` | `re-run`、`plan` 或 `stop` |
| `re-run` | `plan` 或 `stop` |

该表描述的是文档口径下的当前主线行为，并非代码中残留的每一条兼容分支。
