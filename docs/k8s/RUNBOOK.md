# Runbook

## 先看哪里

1. 聚合日志：`/app/job-logs/jobs/<job_id>.log`
2. 阶段文件：`/shared/output/_k8s_jobs/<job_id>/stage-*.json`
3. 任务目录：`/shared/output/<repo>-<id>/run_summary.json`
4. stage pod 日志

## 常见排障路径

### `plan` 失败
看：
- `targets.json` schema
- `target_analysis.json`
- 是否 fallback 成功

### `synthesize` 失败
看：
- 是否 partial scaffold
- 是否至少有 harness + build script
- 是否触发 completion prompt

### `build` 失败
看：
- `build_error_code`
- `error_signature_before/after`
- `fix_action_type`
- `fix_effect`
- `final_build_error_code`

### `run` 很久不结束
看：
- `terminal_reason`
- `coverage_loop.round`
- `coverage_loop.stop_reason`
- `coverage_loop.target_depth_class`
- `coverage_loop.repo_examples_*`

### `re-run` 失败
看：
- `repro_context.json`
- `.repro_crash/`
- `last_crash_artifact`
- `re_workspace_root`

## 线上重点判断标准

- 如果 `run` 已 plateau 且预算耗尽，应正常 stop
- 如果 `replan` 没有 material change，应 stop，不应继续空转
- 如果 `build` 只是 env rebuild 问题，应该 fresh build 收口
