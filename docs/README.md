# Sherpa 文档入口

本目录记录 Sherpa 当前代码实现的真实口径。本文档入口只做导航，不复述历史设计。

## 推荐阅读顺序

1. [仓库总览](/README.md)
   - 适合第一次了解项目，先建立对系统目标、模块边界和工作流的整体认知。

2. [代码级技术分析](/docs/CODEBASE_TECHNICAL_ANALYSIS.md)
   - 面向开发者，按模块和状态文件解释系统如何工作。

3. [比赛展示版技术解读](/docs/COMPETITION_TECHNICAL_BRIEF.md)
   - 面向展示、答辩和演示，重点解释设计价值、系统闭环和工程亮点。

4. [API Reference](/docs/API_REFERENCE.md)
   - 面向前端联调与接口对齐，按当前真实实现整理 `/api/task`、`/api/tasks`、`/api/system`、`/api/config` 等接口。

5. [标准变更流程](/docs/STANDARD_CHANGE_PROCESS.md)
   - 描述 `codex/* -> dev -> main` 的标准验证和发布路径。

## 其他保留文档

- `/docs/k8s/`
  - 部署、运维、发布门禁、Cloudflare、Runbook 等操作型文档。
- `/docs/`
  - 后端工具链与交接类资料。

## 当前统一口径

- 线上执行环境是 Kubernetes，工作流阶段由短生命周期 Job 执行。
- 常驻服务是 `sherpa-web`、`sherpa-frontend`、`postgres`。
- `main.py` 负责外层 API 与调度，`workflow_graph.py` 负责状态机，`fuzz_unharnessed_repo.py` 负责底层 clone/build/run。
- 默认不复用仓库自带 fuzz target，而是统一生成外部 harness 与 build scaffold。
- non-root 是默认运行假设，运行时临时文件优先使用容器内 `/tmp`。
- 前端联调用到的动态指标优先看 `/api/system` 和 `/api/tasks`，其契约以 `docs/API_REFERENCE.md` 为准。
