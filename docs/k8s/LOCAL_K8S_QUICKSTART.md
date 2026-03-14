# 本地 K8s 快速启动

适合做最小链路验证，不追求生产一致性。

## 前提

- `kubectl`
- 可用的本地集群（如 OrbStack / kind / minikube）
- 本地可构建 `sherpa-web` 与 `frontend-next` 镜像

## 最小步骤

1. 准备镜像
2. 导入到本地集群运行时
3. `kubectl apply -k k8s/overlays/dev`
4. 检查：
   - `sherpa-web`
   - `frontend-next`
   - `postgres`
5. 提交一条任务做 smoke test

## 推荐 smoke test 仓库

- `https://github.com/yaml/libyaml.git`
- `https://github.com/fmtlib/fmt.git`

## 本地排障顺序

1. `kubectl get pods`
2. `kubectl logs deploy/sherpa-web`
3. 对应 stage pod 日志
4. `/shared/output/<repo>-<id>/run_summary.json`
