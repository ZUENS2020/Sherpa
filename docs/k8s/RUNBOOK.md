# Kubernetes Runbook

This is the current troubleshooting guide for a deployed Sherpa environment.

## 1. First Places to Look

1. backend aggregate log: `/app/job-logs/jobs/<job_id>.log`
2. stage result files: `/shared/output/_k8s_jobs/<job_id>/stage-*.json`
3. stage error files: `/shared/output/_k8s_jobs/<job_id>/stage-*.error.txt`
4. task workspace artifacts: `/shared/output/<repo>-<id>/`
5. Kubernetes pod logs for the specific stage

## 2. Stage-Oriented Triage

### `plan` issues

Check:

- target planning artifacts exist
- selected/execution targets are coherent
- error text persisted in stage result files

### `synthesize` issues

Check:

- harness source exists
- build scaffold exists
- `execution_plan.json` and `harness_index.json` are aligned

### `build` issues

Check:

- `build_error_code`
- `build_error_kind`
- `target_build_matrix`
- `missing_targets`
- `repair_error_digest`

### `run` issues

Check:

- `run_summary.json`
- `run_error_kind`
- `terminal_reason`
- coverage and feature movement
- seed quality and family gaps

### `crash-triage` issues

Check:

- `crash_info.md`
- `crash_analysis.md`
- `crash_triage.json`

### `re-build` / `re-run` issues

Check:

- `repro_context.json`
- repro workspace paths
- rebuild logs
- artifact existence

## 3. High-Value Questions

Ask these first:

- did the stage persist structured output?
- is the task failing because of planning/synthesis drift, or runtime execution?
- is the crash path actually a harness bug?
- are seed quality and execution-target coverage the real bottleneck?

## 4. Operational Principle

Do not trust stage status alone. Always cross-check:

- stage JSON
- task workspace artifacts
- aggregate logs

That combination is the closest thing to ground truth during live debugging.
