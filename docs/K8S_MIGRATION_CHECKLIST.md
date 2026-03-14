# K8s 迁移检查清单

本清单按当前代码状态维护，目的是确认线上执行链是否仍然与仓库实现一致。

## 已完成

- [x] stage-per-job 调度
- [x] Postgres 持久化
- [x] `coverage-analysis` / `improve-harness`
- [x] `re-build` / `re-run`
- [x] `repro_context.json` 持久化 crash 上下文
- [x] k8s worker 原生执行 `opencode`
- [x] `run` typed seed bootstrap
- [x] `plan` 产出 `target_analysis.json`
- [x] `targets.json` 强制 `target_type + seed_profile`

## 持续验证项

- [ ] target 深度排序是否足够稳定
- [ ] plateau 后最后一轮预算收口是否总能正常停止
- [ ] `fix_build` 规则修复覆盖更多链接错误与 build system 变体
- [ ] `repo_examples` 过滤是否仍会混入低价值文件

## 每次部署后建议验收

1. 跑一条 `libyaml`
2. 跑一条 `zlib`
3. 检查：
- `run_summary.json`
- 是否有 `seed_profile`
- 是否有 `corpus_sources`
- `coverage_loop.stop_reason`
- 是否没有无限空转到 dispatch limit
