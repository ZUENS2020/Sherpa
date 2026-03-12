# Sherpa 后端技术交接（中文）

这是 `harness_generator/` 的中文交接摘要。完整解析见：

- `docs/CODEBASE_TECHNICAL_ANALYSIS.md`

## 当前主线概况

- Web API 与阶段调度：`src/langchain_agent/main.py`
- 工作流状态机：`src/langchain_agent/workflow_graph.py`
- 底层仓库处理与 seed/bootstrap：`src/fuzz_unharnessed_repo.py`
- OpenCode 调用：`src/codex_helper.py`

## 当前真实阶段

- `plan`
- `synthesize`
- `build`
- `fix_build`
- `run`
- `coverage-analysis`
- `improve-harness`
- `re-build`
- `re-run`
- `fix_crash`

## 当前重要实现点

1. `targets.json` 强制包含 `target_type` 与 `seed_profile`
2. `plan` 额外写 `target_analysis.json`
3. `run` 的 seed bootstrap = repo examples + AI + `radamsa`
4. plateau 后优先 `in_place` 改进，连续无收益才 `replan`
5. `replan` 没有 material change 会直接 stop
6. crash 复现依赖 `repro_context.json`
7. k8s worker 原生执行 `opencode`

## 当前接手时优先看哪里

- `run_summary.json`
- `stage-*.json`
- `workflow_graph.py` 的 route 逻辑
- `fuzz_unharnessed_repo.py` 的 seed/bootstrap 与 run 实现
