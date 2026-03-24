# Kubernetes 部署指南

这是 Sherpa 当前在 Kubernetes 上的简版部署说明。

## 1. 会部署什么

常驻服务：

- 后端 API
- 前端 UI
- Postgres

短生命周期负载：

- 每个工作流阶段对应一个 Kubernetes Job

## 2. 当前运行形态

- 执行器模式应为 `k8s_job`
- 输出根目录应对后端与阶段作业都可见
- worker 镜像与后端发布版本要保持一致
- 默认按非 root 运行假设配置

## 3. 部署步骤

1. 构建或引用目标 backend / frontend 镜像
2. 应用对应 overlay 或 manifests
3. 等待常驻服务 ready
4. 核对配置与 worker 镜像引用
5. 提交一个真实仓库任务作为 smoke test

## 4. 推荐 smoke test 仓库

- `https://github.com/fmtlib/fmt.git`
- `https://github.com/yaml/libyaml.git`
- `https://github.com/madler/zlib.git`
- `https://github.com/libarchive/libarchive.git`

## 5. 成功标准

- 任务提交成功
- 阶段作业可以被正确创建
- 失败时能留下持久化报告
- `/api/tasks` 和 `/api/system` 能反映实时状态
- 任务产物出现在 `/shared/output`

## 6. 下一步阅读

- [`DEPLOYMENT_DETAILED.md`](DEPLOYMENT_DETAILED.md)
- [`RUNBOOK.md`](RUNBOOK.md)
- [`MAPPING.md`](MAPPING.md)
