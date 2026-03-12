# 项目当前交付状态

本文档记录 Sherpa 当前代码主线的交付状态，适合作为接手人快速判断系统是否与线上行为一致。

## 当前已落地能力

- `k8s_job` 主执行链稳定运行
- `plan` 输出 `target_type + seed_profile + target_analysis.json`
- `synthesize` 支持 partial scaffold completion
- `build/fix_build` 支持 quick-check、env rebuild、link/source 细分类
- `run` 支持：
  - AI seed 生成
  - repo examples 过滤复用
  - `radamsa` 变异
  - plateau 检测
  - stop-on-first-crash
- `coverage-analysis` / `improve-harness` 形成闭环
- `re-build` / `re-run` 独立 crash 复现链路
- `repro_context.json` 用于 crash 上下文持久化
- `k8s` worker 内 `opencode` 原生执行

## 当前已知重点问题

1. target 仍可能过浅
- 尤其是 utility/checksum 类 API
- 已加入 depth bias，但仍需继续验证

2. plateau 收口依赖 target 与 seed 质量
- 虽然已加入 plateau 检测与 budget 门控
- 但对部分仓库仍可能需要更 aggressive 的 target 选择

3. `fix_build` 仍需持续扩展规则修复覆盖
- 尤其是更复杂的 linker/ABI/build system 变体

## 当前产物与状态文件

- `/shared/output/<repo>-<id>/run_summary.json`
- `/shared/output/<repo>-<id>/repro_context.json`
- `/shared/output/_k8s_jobs/<job_id>/stage-*.json`
- `/app/job-logs/jobs/<job_id>.log`

## 当前推荐排障入口

1. 看 `run_summary.json`
2. 看 `stage-*.json` 和 `stage-*.error.txt`
3. 看聚合日志
4. 看具体 stage pod 日志

## 当前不再使用的旧机制

- GitNexus 运行时预分析
- k8s worker 内 `docker run opencode`
- 固定阶段尾序列调度
