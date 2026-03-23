# Zlib 端到端验证模板

本文档是一个可复用模板，用于验证 Sherpa 是否能够对真实仓库完成一次完整运行。虽然文件名保留了 zlib 历史背景，但模板适用于任何仓库级 smoke test。

## 目标

通过一次端到端任务确认当前工作流是否能够：

- 合理选择目标
- 产出 harness 与构建脚手架
- 构建出可运行的 fuzzer
- 生成可用的种子语料
- 运行 fuzzer
- 输出预期产物

## 建议记录的字段

- Job ID
- 仓库 URL
- 选中的目标
- 生成的 harness 文件
- `fuzz/execution_plan.json`
- `fuzz/harness_index.json`
- `run_summary.json`
- 终止阶段与停止原因
- 任意崩溃或 coverage-analysis 输出

## 结果模板

```text
job_id:
repository:
selected_targets:
harness_files:
seed_profile:
execution_plan_targets:
built_targets:
terminal_reason:
coverage_stop_reason:
crash_detected:
notes:
```

## 如何解读

使用此模板回答三个问题：

1. 工作流是否创建了连贯的“目标 -> harness”映射？
2. build 与 run 是否操作的是同一组目标？
3. 任务是否以有意义的产物结束，而不是静默循环？
