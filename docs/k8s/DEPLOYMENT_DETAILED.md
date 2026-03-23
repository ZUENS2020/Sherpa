# Kubernetes 部署详解

本文档比简版部署指南更详细地说明当前 Sherpa 的部署模型。

## 1. 组件模型

### 常驻服务

- 后端 API / 控制面
- 前端 UI
- Postgres

### 阶段作业

工作流会为以下阶段分发短生命周期 Job：

- `plan`
- `synthesize`
- `build`
- `run`
- `crash-triage`
- `fix-harness`
- `coverage-analysis`
- `improve-harness`
- `re-build`
- `re-run`

## 2. 数据与产物模型

Sherpa 依赖持久化输出路径，而不是 Pod 本地状态。

重要持久化位置：

- `/shared/output/<repo>-<id>/`
- `/shared/output/_k8s_jobs/<job_id>/`
- `/app/job-logs/jobs/<job_id>.log`

## 3. 控制流

```mermaid
sequenceDiagram
  participant U as 用户
  participant FE as 前端
  participant API as 后端
  participant DB as Postgres
  participant K8S as 阶段作业

  U->>FE: 提交仓库
  FE->>API: POST /api/task
  API->>DB: 创建父任务与子作业
  API->>K8S: 分发阶段作业
  K8S->>API: 持久化阶段结果
  API->>K8S: 分发下一阶段
```

## 4. 环境要求

- worker 与 backend 版本必须对齐
- 共享输出路径必须对所有阶段作业可访问
- metrics 可提升可观测性，但不是工作流事实来源
- 应保持非 root 运行与临时目录相关假设

## 5. 部署后要验证什么

- 后端路由可正常响应
- 前端可以加载并读取实时 API 数据
- 阶段作业能够被成功分发
- 持久化输出出现在预期位置
- 至少一个真实仓库任务能成功跨越多个阶段

## 6. 本文不覆盖的内容

- 从零开始引导集群
- 云厂商特定的 ingress / 负载均衡配置
- 历史迁移笔记

如需查看集群引导，请阅读 [ORIGINAL_K8S_CLUSTER_DEPLOYMENT.md](ORIGINAL_K8S_CLUSTER_DEPLOYMENT.md)。
