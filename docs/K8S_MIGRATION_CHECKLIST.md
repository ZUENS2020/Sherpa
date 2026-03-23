# K8s 迁移检查清单

该文件仅保留为历史迁移背景，不再是当前部署手册。

当前主要文档请查看：

- [k8s/DEPLOY.md](k8s/DEPLOY.md)
- [k8s/DEPLOYMENT_DETAILED.md](k8s/DEPLOYMENT_DETAILED.md)
- [k8s/RUNBOOK.md](k8s/RUNBOOK.md)

## 历史迁移结果

从高层角度看，迁移目标已经达成：

- 已具备分阶段的 Kubernetes Job 执行模型
- 输出与日志可以脱离 Pod 生命周期独立持久化
- 前后端服务与阶段作业已解耦
- 覆盖率改进与崩溃复现已经纳入工作流模型

## 变更后需要验证什么

当基础设施或工作流发生变更时，应验证：

- 阶段作业仍会产出 `stage-*.json` 与 `stage-*.error.txt`
- 任务工作目录仍落在 `/shared/output`
- `/api/tasks` 与 `/api/system` 仍能正确反映实时任务状态
- 至少一个真实仓库任务能够跑通主线链路

## 本文件不再尝试描述的内容

- 不再详细说明当前工作流阶段图
- 不再记录部署命令
- 不再定义发布准入标准
