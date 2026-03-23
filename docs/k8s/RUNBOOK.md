# Kubernetes 运行手册

本文档是当前已部署 Sherpa 环境的故障排查指南。

## 1. 第一时间查看的位置

1. 后端聚合日志：`/app/job-logs/jobs/<job_id>.log`
2. 阶段结果文件：`/shared/output/_k8s_jobs/<job_id>/stage-*.json`
3. 阶段错误文件：`/shared/output/_k8s_jobs/<job_id>/stage-*.error.txt`
4. 任务工作目录产物：`/shared/output/<repo>-<id>/`
5. 对应阶段的 Kubernetes Pod 日志

## 2. 以阶段为中心的排障

### `plan` 问题

检查：

- 目标规划产物是否存在
- selected / execution targets 是否一致
- 错误文本是否已持久化到阶段结果文件中

### `synthesize` 问题

检查：

- harness 源码是否存在
- build 脚手架是否存在
- `execution_plan.json` 与 `harness_index.json` 是否一致

### `build` 问题

检查：

- `build_error_code`
- `build_error_kind`
- `target_build_matrix`
- `missing_targets`
- `repair_error_digest`

### `run` 问题

检查：

- `run_summary.json`
- `run_error_kind`
- `terminal_reason`
- 覆盖率与 feature 是否有变化
- 种子质量与输入家族缺口

### `crash-triage` 问题

检查：

- `crash_info.md`
- `crash_analysis.md`
- `crash_triage.json`

### `re-build` / `re-run` 问题

检查：

- `repro_context.json`
- 复现工作目录路径
- rebuild 日志
- 产物是否实际存在

## 3. 高价值问题

优先问这几个问题：

- 该阶段是否持久化了结构化输出？
- 任务失败是由规划 / 生成漂移导致，还是由运行时执行导致？
- 崩溃路径是否其实是 harness bug？
- 真正瓶颈是不是种子质量或执行目标覆盖不足？

## 4. 运维原则

不要只信阶段状态。在线排障时，必须交叉验证：

- stage JSON
- 任务工作目录产物
- 聚合日志

这三者结合起来，才是最接近事实的依据。
