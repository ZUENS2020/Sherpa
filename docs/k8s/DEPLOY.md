# K8s 部署说明（简版）

## 常用覆盖层

- 开发环境：`k8s/overlays/dev`
- 生产环境：`k8s/overlays/prod`

## 核心组件

- `sherpa-web`
- `frontend-next`
- `postgres`

## 部署步骤

1. 构建并导入 `sherpa-web` 镜像
2. 构建并导入 `frontend-next` 镜像
3. `kubectl apply -k` 对应 overlay
4. 等待 rollout 完成
5. 用真实仓库任务做 smoke test

## smoke test 推荐仓库

- `https://github.com/yaml/libyaml.git`
- `https://github.com/fmtlib/fmt.git`
- `https://github.com/madler/zlib.git`

## 成功标准

- `plan` 不再依赖 Docker CLI
- `run` 能生成 seed 并写入 `run_summary.json`
- plateau 任务在预算内正常收口
- crash 任务能进入 `re-build/re-run`
