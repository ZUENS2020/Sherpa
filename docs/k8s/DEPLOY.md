# Kubernetes 部署指南

这是 Sherpa 当前在 Kubernetes 上部署的简版指南。

## 1. 会部署什么

常驻服务：

- 后端 API 服务
- 前端 UI 服务
- Postgres

短生命周期负载：

- 每个工作流阶段对应一个 Kubernetes Job

## 2. 期望运行形态

- 执行器模式应为 `k8s_job`
- 输出根目录应为共享路径，并且对阶段作业与后端都可见
- 后端与 worker 运行时镜像版本必须一致
- 默认假设为非 root 执行

## 3. 基本部署流程

1. 构建或引用目标 backend / frontend 镜像
2. 应用对应 overlay 或 manifests
3. 等待常驻服务 ready
4. 验证配置与 worker 镜像引用
5. 提交一个真实仓库任务作为 smoke test

## 4. 建议的 smoke test 仓库

推荐：

- `https://github.com/fmtlib/fmt.git`
- `https://github.com/yaml/libyaml.git`
- `https://github.com/madler/zlib.git`
- `https://github.com/libarchive/libarchive.git`

## 5. 成功标准

- 任务提交成功
- 阶段作业被创建，并能完成或在失败时留下持久化报告
- `/api/tasks` 与 `/api/system` 能反映实时状态
- 任务产物出现在 `/shared/output` 下

## 6. 下一步阅读

- [DEPLOYMENT_DETAILED.md](DEPLOYMENT_DETAILED.md)
- [RUNBOOK.md](RUNBOOK.md)
- [MAPPING.md](MAPPING.md)
