# Kubernetes Mapping

This file maps Sherpa concepts to current runtime objects.

## 1. Runtime Object Mapping

| Logical component | Current implementation |
|---|---|
| API / control plane | backend Deployment / service |
| UI | frontend Deployment / service |
| state store | Postgres |
| stage execution | Kubernetes Job |
| task output root | `/shared/output` |
| aggregated job logs | `/app/job-logs/jobs/*.log` |

## 2. Important Artifact Mapping

| Path or file | Meaning |
|---|---|
| `fuzz/PLAN.md` | planning artifact |
| `fuzz/targets.json` | candidate targets |
| `fuzz/selected_targets.json` | selected targets with execution metadata |
| `fuzz/execution_plan.json` | execution-target contract |
| `fuzz/harness_index.json` | target-to-harness mapping |
| `run_summary.json` | task-level run summary |
| `repro_context.json` | crash repro context |
| `stage-*.json` | stage result record |

## 3. Current Mainline Routing

| Stage | Possible next stage |
|---|---|
| `plan` | `synthesize` |
| `synthesize` | `build` |
| `build` | `run` or `plan` |
| `run` | `coverage-analysis`, `crash-triage`, or `plan` depending on outcome |
| `coverage-analysis` | `improve-harness` or `stop` |
| `improve-harness` | `build`, `plan`, or `stop` |
| `crash-triage` | `fix-harness`, `re-build`, `plan`, or `stop` |
| `fix-harness` | `build` or `plan` |
| `re-build` | `re-run`, `plan`, or `stop` |
| `re-run` | `plan` or `stop` |

This table describes the documented current mainline behavior, not every compatibility branch left in code.
