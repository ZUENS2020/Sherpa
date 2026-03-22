# Sherpa 文档入口

本目录记录 Sherpa 当前代码实现的真实口径。这里不复述历史设计，只保留当前能落地、可联调、可接手的内容。

## 推荐阅读顺序

1. [仓库总览](/README.md)
   - 先看项目定位、当前工作流和模块边界。

2. [代码级技术分析](/docs/CODEBASE_TECHNICAL_ANALYSIS.md)
   - 面向开发者，解释控制面、执行面、工作流状态机和 crash triage 逻辑。

3. [API Reference](/docs/API_REFERENCE.md)
   - 面向前端联调，按当前真实实现整理 `/api/task`、`/api/tasks`、`/api/system`、`/api/config` 等接口。

4. [技术学习指南](/docs/TECHNICAL_DEEP_DIVE.md)
   - 面向开发者系统学习 Sherpa 的工作流、状态机、API、K8s 和 crash triage 逻辑。

5. [标准变更流程](/docs/STANDARD_CHANGE_PROCESS.md)
   - 描述 `codex/* -> dev -> main` 的标准验证和发布路径。

6. [原版 K8s 主控/Worker 集群部署](/docs/k8s/ORIGINAL_K8S_CLUSTER_DEPLOYMENT.md)
   - 面向 kubeadm，说明如何把多台服务器组成集群并部署 Sherpa。

## 当前统一口径

- 线上执行环境是 Kubernetes，工作流阶段由短生命周期 Job 执行。
- 常驻服务是 `sherpa-web`、`sherpa-frontend`、`postgres`。
- `main.py` 负责外层 API 与调度，`workflow_graph.py` 负责状态机，`fuzz_unharnessed_repo.py` 负责底层 clone/build/run。
- 当前主线工作流包含：
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
- `fix_build` / `fix_crash` 仍保留兼容逻辑，但不是当前主线的唯一修复路径。
- 默认不复用仓库自带 fuzz target，而是统一生成外部 harness 与 build scaffold。
- non-root 是默认运行假设，运行时临时文件优先使用容器内 `/tmp`。
- 前端联调用到的动态指标优先看 `/api/system` 和 `/api/tasks`，其契约以 `docs/API_REFERENCE.md` 为准。
