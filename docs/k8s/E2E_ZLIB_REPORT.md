# Zlib E2E 验证模板

## 目标
验证以下问题是否已被当前代码主线修复：

- target 不再优先落到浅层 `adler32` 类 utility target
- `repo_examples` 不再把 `.c/.h/.html` 当 seed
- `build/fix_build` 对 `LLVMFuzzerTestOneInput` 与 `-lz` 问题分类更准确

## 建议记录项

- 任务 ID
- `fuzz/targets.json`
- 实际落成的 harness
- `run_summary.json`
- `coverage_loop.target_depth_class`
- `coverage_loop.repo_examples_*`
- 最终 `build_error_code` / `final_build_error_code`

## 结果记录模板

```text
job_id:
selected_target:
selected_depth_class:
seed_profile:
repo_examples_accepted_count:
repo_examples_rejected_count:
terminal_reason:
coverage_stop_reason:
final_build_error_code:
notes:
```
