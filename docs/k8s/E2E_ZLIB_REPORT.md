# Zlib E2E Validation Template

This document is a reusable template for validating a full Sherpa run against a
real repository. The filename is historical, but the template applies to any
repository-level smoke test.

## Goal

Use one end-to-end job to confirm that the current workflow can:

- pick targets reasonably,
- materialize harnesses and build scaffolding,
- build runnable fuzzers,
- generate usable seed corpus,
- run the fuzzers,
- and emit the expected artifacts.

## Suggested Fields To Record

- Job ID
- Repository URL
- Selected targets
- Generated harness files
- `fuzz/execution_plan.json`
- `fuzz/harness_index.json`
- `run_summary.json`
- terminal stage and stop reason
- any crash or coverage-analysis outputs

## Result Template

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

## Interpretation

Use this template to answer three questions:

1. Did the workflow create a coherent target-to-harness mapping?
2. Did the build and run stages operate on the same target set?
3. Did the task terminate with meaningful artifacts instead of silent looping?
