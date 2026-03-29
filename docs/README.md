# 文档索引

这里是 Sherpa 的技术文档入口。目标是帮助你按“当前真实实现”理解系统，而不是按历史交接材料理解系统。

## 建议阅读顺序

1. [`../README.md`](../README.md)
   系统总览、主工作流、API 与部署模型。

2. [`CODEBASE_TECHNICAL_ANALYSIS.md`](CODEBASE_TECHNICAL_ANALYSIS.md)
   代码库分层、主入口和工作流状态机。

3. [`TECHNICAL_DEEP_DIVE.md`](TECHNICAL_DEEP_DIVE.md)
   推荐阅读顺序、核心循环与常见失败来源。

4. [`API_REFERENCE.md`](API_REFERENCE.md)
   当前后端 API 契约，供前端联调与任务控制使用。

5. [`PROMEFUZZ_MCP_TECHNICAL_SPEC.md`](PROMEFUZZ_MCP_TECHNICAL_SPEC.md)
   PromeFuzz companion + HTTP MCP + embedding/RAG + 新增字段技术细节。

6. [`STANDARD_CHANGE_PROCESS.md`](STANDARD_CHANGE_PROCESS.md)
   变更、验证、发布与文档同步流程。

## 部署与运行

- [`k8s/DEPLOY.md`](k8s/DEPLOY.md)
- [`k8s/DEPLOYMENT_DETAILED.md`](k8s/DEPLOYMENT_DETAILED.md)
- [`k8s/RUNBOOK.md`](k8s/RUNBOOK.md)
- [`k8s/MAPPING.md`](k8s/MAPPING.md)
- [`k8s/RELEASE_GATE.md`](k8s/RELEASE_GATE.md)
- [`k8s/ORIGINAL_K8S_CLUSTER_DEPLOYMENT.md`](k8s/ORIGINAL_K8S_CLUSTER_DEPLOYMENT.md)

## 历史材料

以下文件保留为历史背景，不能作为当前主手册：

- [`PROJECT_HANDOFF_STATUS.md`](PROJECT_HANDOFF_STATUS.md)
- [`K8S_MIGRATION_CHECKLIST.md`](K8S_MIGRATION_CHECKLIST.md)
- [`DOCKER_TO_K8S_HANDOFF.md`](DOCKER_TO_K8S_HANDOFF.md)
- [`k8s/DEPLOY_ISSUES_NON_NETWORK.md`](k8s/DEPLOY_ISSUES_NON_NETWORK.md)
- [`k8s/E2E_ZLIB_REPORT.md`](k8s/E2E_ZLIB_REPORT.md)

## 当前文档规则

- 工作流描述以当前阶段流转为准：`plan`、`synthesize`、`build`、`run`、`coverage-analysis`、`improve-harness`、`crash-triage`、`fix-harness`、`re-build`、`re-run`、`crash-analysis`
- 历史修复节点 `fix_build` / `fix_crash` 如果出现，只能作为兼容实现说明，不能写成主线推荐路径
- API 文档必须与 [`harness_generator/src/langchain_agent/main.py`](../harness_generator/src/langchain_agent/main.py) 保持一致
- 所有链接使用仓库相对路径
- 所有历史说明必须显式标注为“历史背景”或“遗留材料”
