# 组件映射

## 运行时映射

| 逻辑组件 | 当前实现 |
|---|---|
| API | `sherpa-web` Deployment |
| UI | `frontend-next` Deployment |
| 状态存储 | Postgres |
| 阶段执行 | Kubernetes Job |
| 工作目录 | `/shared/output` |
| 聚合日志 | `/app/job-logs/jobs/*.log` |

## 状态文件映射

| 文件 | 作用 |
|---|---|
| `fuzz/PLAN.md` | plan 输出 |
| `fuzz/targets.json` | target 列表与元数据 |
| `fuzz/target_analysis.json` | 工具辅助 target 分析 |
| `run_summary.json` | 本轮汇总状态 |
| `repro_context.json` | crash 复现上下文 |
| `stage-*.json` | 单阶段结果 |

## 当前阶段路由

| 阶段 | 可能下一步 |
|---|---|
| `plan` | `synthesize` |
| `synthesize` | `build` |
| `build` | `run` / `fix_build` / `build`(env rebuild) |
| `run` | `coverage-analysis` / `re-build` / `stop` |
| `coverage-analysis` | `improve-harness` / `stop` |
| `improve-harness` | `build` / `plan` / `stop` |
| `re-build` | `re-run` / `plan` |
| `re-run` | `stop` / `plan` |
