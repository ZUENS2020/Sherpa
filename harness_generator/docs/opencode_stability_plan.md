# OpenCode 稳定性现状与约束

本文档记录当前已经落地的 OpenCode 稳定性机制，不是未来空计划。

## 已落地机制

### 1. sentinel 与 idle timeout
- 通过 `codex_helper.py` 管理 `./done`
- idle timeout 会终止长时间无输出的 OpenCode 调用

### 2. plan 阶段 schema 护栏
- `targets.json` 必须 schema-valid
- 第一次不合法会重试
- fallback 也必须写合法 `target_type + seed_profile`

### 3. synthesize partial scaffold completion
- 有 harness 但 scaffold 不完整时，不再直接失败
- 会发 completion prompt 只补缺失项

### 4. build/fix_build 护栏
- error signature
- quick-check build
- env rebuild
- `max_fix_rounds`
- `same_error_max_retries`

### 5. k8s 原生执行
- `k8s_job` 路径下不再依赖 inner Docker `opencode`
- worker 内直接调用本地 `opencode`

### 6. 命令白名单
- `grep/rg` 永远允许，不受 blocklist 环境变量误伤

## 当前仍需关注

- OpenCode 在大仓库 synthesize 时仍可能 partial output
- 目标选择过浅时，即使 scaffold 正常，fuzz 效益仍不高
- seed prompt 质量与真实 coverage 仍需持续迭代
