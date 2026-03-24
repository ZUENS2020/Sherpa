# Kubernetes 运行手册

本文档用于排查当前已部署 Sherpa 环境。

## 1. 第一时间查看的位置

1. 后端聚合日志：`/app/job-logs/jobs/<job_id>.log`
2. 阶段结果文件：`/shared/output/_k8s_jobs/<job_id>/stage-*.json`
3. 阶段错误文件：`/shared/output/_k8s_jobs/<job_id>/stage-*.error.txt`
4. 任务工作目录：`/shared/output/<repo>-<id>/`
5. 对应阶段的 Kubernetes Pod 日志

## 2. 以阶段为中心的排障

### `plan`

检查：

- `PLAN.md` 是否存在
- `targets.json` 与 `selected_targets.json` 是否一致
- 规划错误是否持久化到了 stage 文件

### `synthesize`

检查：

- harness 源码是否存在
- build 脚手架是否存在
- `execution_plan.json` 与 `harness_index.json` 是否一致

### `build`

检查：

- `build_error_code`
- `build_error_kind`
- `target_build_matrix`
- `missing_targets`
- `repair_error_digest`

### `run`

检查：

- `run_summary.json`
- `run_error_kind`
- 覆盖率和 exec/s 是否有变化
- 种子质量与输入家族缺口

### `coverage-analysis`

检查：

- `coverage_should_improve`
- `coverage_improve_mode`
- `coverage_quality_oracle`

### `crash-triage`

检查：

- `crash_info.md`
- `crash_triage.json`
- 分类是否与日志一致

### `re-build` / `re-run`

检查：

- `repro_context.json`
- 复现工作目录路径
- rebuild / rerun 日志
- crash 是否真的可复现

### `crash-analysis`

检查：

- `crash_analysis.md`
- `crash_analysis.json`
- verdict 是否是 `false_positive` / `real_bug` / `unknown`

## 3. 高价值问题

优先问这几个问题：

- 该阶段是否已经持久化了结构化输出？
- 任务失败是规划问题、生成问题，还是运行问题？
- 崩溃路径是不是 harness 误报？
- 真正瓶颈是不是种子质量或执行目标覆盖不足？

## 4. 运维原则

不要只信阶段状态。排障时必须交叉验证：

- stage JSON
- 任务工作目录产物
- 聚合日志
