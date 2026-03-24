# 本地 Kubernetes 快速开始

本文档面向希望在本地 Kubernetes 集群上拉起最小 Sherpa 栈并做 smoke test 的开发者。它不是生产部署指南。

## 目标

启动控制面，提交一个任务，并确认各阶段 Pod 可以端到端运行。

## 前置条件

- 可用的本地 Kubernetes 集群，例如 OrbStack、kind 或 minikube
- `kubectl`
- 能够获取或构建以下镜像：
  - `sherpa-web`
  - `frontend-next`
- 可写的阶段产物输出目录

## 最小启动步骤

1. 将所需镜像构建或导入到本地集群运行时中。
2. 应用开发环境 overlay：

```bash
kubectl apply -k k8s/overlays/dev
```

3. 验证核心服务：

```bash
kubectl get pods
kubectl get svc
kubectl get ingress
```

至少应看到以下服务处于健康状态：

- `sherpa-web`
- `frontend-next`
- `postgres`

4. 打开 Web UI，提交一个 smoke-test 任务。

## 推荐的 smoke-test 仓库

- `https://github.com/yaml/libyaml.git`
- `https://github.com/fmtlib/fmt.git`

这些仓库体量较小，能较快给出反馈，同时仍能覆盖规划、生成、构建、运行与产物链路。

## 需要检查什么

提交任务后，请验证：

- `/api/tasks` 中出现了任务记录
- 各阶段作业按顺序创建
- 阶段日志可查看
- 仓库工作目录已创建在共享输出根目录下
- 作业产生了 `run_summary.json` 或阶段结果文件
- 如果触发崩溃，还应能看到 `crash_info.md`、`crash_triage.json`，以及复现成功后的 `crash_analysis.md`

## 本地排障顺序

1. 查看集群对象：

```bash
kubectl get pods
kubectl get jobs
kubectl get events --sort-by=.metadata.creationTimestamp
```

2. 查看 Web 服务日志：

```bash
kubectl logs deploy/sherpa-web
```

3. 查看活跃阶段 Pod 或 Job 的日志：

```bash
kubectl logs <pod-name>
```

4. 查看生成的产物目录：

```bash
ls -la /shared/output/<repo>-<id>
cat /shared/output/<repo>-<id>/run_summary.json
```

## 相关文档

- [`DEPLOY.md`](DEPLOY.md)
- [`DEPLOYMENT_DETAILED.md`](DEPLOYMENT_DETAILED.md)
- [`RUNBOOK.md`](RUNBOOK.md)
- [`../API_REFERENCE.md`](../API_REFERENCE.md)
