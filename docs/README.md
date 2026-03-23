# 文档索引

本目录用于说明 Sherpa 当前实际行为，组织方式优先服务于技术学习，而不是历史还原。

## 建议先阅读

1. [../README.md](../README.md)
   系统概览、当前工作流以及产物模型。

2. [CODEBASE_TECHNICAL_ANALYSIS.md](CODEBASE_TECHNICAL_ANALYSIS.md)
   模块边界、控制面与执行面的关系，以及状态机语义。

3. [TECHNICAL_DEEP_DIVE.md](TECHNICAL_DEEP_DIVE.md)
   如何高效阅读代码库，以及各项核心能力分别在做什么。

4. [API_REFERENCE.md](API_REFERENCE.md)
   当前后端 API 契约，供前端集成与任务控制使用。

5. [STANDARD_CHANGE_PROCESS.md](STANDARD_CHANGE_PROCESS.md)
   分支、验证、文档同步与发布要求。

## 部署与运维

- [k8s/DEPLOY.md](k8s/DEPLOY.md)
- [k8s/DEPLOYMENT_DETAILED.md](k8s/DEPLOYMENT_DETAILED.md)
- [k8s/RUNBOOK.md](k8s/RUNBOOK.md)
- [k8s/MAPPING.md](k8s/MAPPING.md)
- [k8s/ORIGINAL_K8S_CLUSTER_DEPLOYMENT.md](k8s/ORIGINAL_K8S_CLUSTER_DEPLOYMENT.md)

## 历史 / 遗留背景

以下文件保留为历史上下文，而不是当前主要操作手册：

- [PROJECT_HANDOFF_STATUS.md](PROJECT_HANDOFF_STATUS.md)
- [K8S_MIGRATION_CHECKLIST.md](K8S_MIGRATION_CHECKLIST.md)
- [DOCKER_TO_K8S_HANDOFF.md](DOCKER_TO_K8S_HANDOFF.md)
- [k8s/DEPLOY_ISSUES_NON_NETWORK.md](k8s/DEPLOY_ISSUES_NON_NETWORK.md)
- [k8s/E2E_ZLIB_REPORT.md](k8s/E2E_ZLIB_REPORT.md)

## 当前文档规则

- 工作流描述应遵循当前主线阶段：
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
- API 文档必须与 `harness_generator/src/langchain_agent/main.py` 中的 FastAPI 实现保持一致。
- 文档中的链接应使用仓库相对路径。
- 历史设计说明必须明确标注为“历史背景”，不能被误解为当前运行事实。
