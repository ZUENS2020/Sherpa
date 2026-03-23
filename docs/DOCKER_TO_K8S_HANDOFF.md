# Docker 到 K8s 的交接说明

本文件是历史背景，用于解释系统为何脱离旧的执行假设，不是当前运行手册。

## 当前运行现实

Sherpa 现在默认假设：

- Kubernetes 是主要的分阶段执行环境
- worker 在阶段 Pod 内原生执行
- 共享输出根目录是 `/shared/output`
- 控制面服务长期运行，阶段作业短生命周期执行

## 为什么还保留这份历史说明

部分旧讨论与旧产物中仍可能出现：

- inner Docker 假设
- 迁移检查点
- Kubernetes 之前的执行模式

保留此文件只是为了说明，这些旧假设已不再是参考模型。

## 请改看这些文档

- [README.md](README.md)
- [k8s/DEPLOY.md](k8s/DEPLOY.md)
- [k8s/DEPLOYMENT_DETAILED.md](k8s/DEPLOYMENT_DETAILED.md)
