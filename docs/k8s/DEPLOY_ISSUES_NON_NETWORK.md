# 非网络类部署问题清单

## 1. OpenCode 仍走 Docker CLI
症状：
- `Docker CLI not found; cannot build opencode image`

当前正确口径：
- k8s worker 必须原生执行 `opencode`
- 不能再依赖 inner Docker

## 2. 镜像内缺少运行时工具
症状：
- seed/bootstrap 工具缺失
- `radamsa` 不存在
- Python 分析依赖缺失

处理：
- 优先修 `docker/Dockerfile.web`
- 不在运行时动态安装工具

## 3. `fix_build` 误判
症状：
- 明明是 link error，却被归类成 source error
- env rebuild 后仍重复原错误

处理：
- 看 `build_error_code`
- 看 `error_signature_before/after`
- 看是否命中规则修复还是 OpenCode 修复

## 4. plateau 空转
症状：
- `run` 多轮 plateau
- 最后打满 dispatch limit

处理：
- 看 `coverage_loop.stop_reason`
- 看 `coverage_replan_effective`
- 看是否最后一轮预算门控失效
