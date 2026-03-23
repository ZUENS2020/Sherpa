# OpenCode 稳定性现状与约束

本文档记录当前已经落地的 OpenCode 稳定性机制，而不是未来的空计划。

## 已落地机制

### 1. sentinel 与 idle timeout
- 通过 `codex_helper.py` 管理 `./done`
- idle timeout 会终止长时间无输出的 OpenCode 调用

### 2. plan 阶段 schema 护栏
- `targets.json` 必须满足 schema 校验
- 第一次不合法会自动重试
- fallback 也必须写出合法的 `target_type + seed_profile`

### 3. synthesize 阶段的残缺脚手架补全
- 当已有 harness 但 scaffold 不完整时，不再直接失败
- 会发送 completion prompt，仅补齐缺失项

### 4. build / fix_build 护栏
- error signature
- quick-check build
- env rebuild
- `max_fix_rounds`
- `same_error_max_retries`

### 5. k8s 原生执行
- `k8s_job` 路径下不再依赖 inner Docker `opencode`
- worker 内直接调用本地 `opencode`

### 6. 命令白名单
- `grep/rg` 永远允许，不会被 blocklist 环境变量误伤

## 当前仍需关注

- OpenCode 在大型仓库的 synthesize 阶段仍可能产生 partial output
- 目标选择过浅时，即使 scaffold 正常，fuzz 收益仍然不高
- seed prompt 质量与真实 coverage 的相关性仍需持续迭代
