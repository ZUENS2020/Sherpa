# Sherpa 项目技术报告（SHE-91）

> 目标：提供一份可直接作为论文写作输入材料的“实现级技术报告”，使未参与项目的人能够理解该系统的设计目标、核心机制、工程实现、验证路径与边界条件。

---

## 1. 项目概览

Sherpa 是一个面向 C/C++ 仓库的自动化 fuzz 编排系统。系统输入是仓库 URL，输出是可审计的 fuzz 执行结果（包括阶段日志、构建结果、崩溃样本与复现状态），并通过 Kubernetes Job 执行多阶段流水线。

当前仓库对外口径是：

- Kubernetes-only 执行模式
- Postgres 持久化任务状态
- stage-per-job（每个阶段单独 Job）
- 前后端分离（FastAPI + Next.js）

---

## 2. 业务问题与研究价值

### 2.1 解决的问题

该项目要解决的核心问题是：对于“没有现成 fuzz harness”的仓库，如何从零到一完成自动化 fuzz 过程，并保持结果可回溯、可重放、可诊断。

传统难点包括：

1. 目标选择依赖经验，自动化难度高。
2. harness 初版构建失败率高，修复链路长。
3. fuzz 运行异常（超时、空跑、崩溃）难以统一建模。
4. CI/CD 与运行环境耦合，容易“本地可跑、线上失败”。

### 2.2 对论文写作的可提炼方向

可从以下方向展开学术化论述：

1. **多阶段自治工作流**：计划、生成、构建、运行、复现的状态机建模。
2. **LLM 与确定性执行的分工**：哪些步骤允许模型介入，哪些步骤严格禁用模型（例如 fuzz 验证阶段）。
3. **云原生可观测性**：Job 级执行结果标准化落盘与聚合 API。
4. **工程稳态机制**：错误分类、回流（restart-to-plan）、环境隔离（dev/prod）。

---

## 3. 系统架构

### 3.1 逻辑架构

系统由四层组成：

1. **交互层**：`frontend-next`（Next.js）用于参数配置、任务提交、进度与日志展示。
2. **控制层**：`sherpa-web`（FastAPI）负责任务生命周期、阶段调度、状态聚合、配置管理。
3. **执行层**：Kubernetes Job Worker 按阶段执行 pipeline。
4. **存储与产物层**：Postgres 保存任务状态，`/shared/output` 保存阶段结果与中间产物。

### 3.2 关键设计点

- API 端保持“任务编排职责”，将具体执行拆到 k8s Job，降低 web 容器阻塞风险。
- 使用阶段产物 JSON（`stage-*.json`）作为控制层与执行层之间的契约格式。
- 通过 `re-build` 与 `re-run` 将“崩溃发现”与“崩溃复现”分离，便于根因定位。

---

## 4. 核心执行模型

### 4.1 阶段序列

标准阶段序列为：

1. `plan`
2. `synthesize`
3. `build`
4. `run`
5. `coverage-analysis`
6. `improve-harness`
7. `re-build`
8. `re-run`

其中 `fix_build`/`fix_crash` 属于工作流内部修复节点，外层 stage 队列以 `build/run/re-*` 结果驱动。

### 4.2 状态与恢复

系统维护父任务与子任务两级状态：

- 父任务：负责聚合多个子任务的总状态。
- 子任务：对应单仓库 fuzz 执行。

当阶段失败且满足恢复条件时，可触发回流：

- `re-build` 或 `re-run` 失败 -> 回流到 `plan`（受计数上限约束）
- `resume` API 可从推断阶段继续执行（存在 fallback 策略）

### 4.3 执行超时策略

运行阶段与非运行阶段使用不同超时策略：

- `run` 阶段使用更大的 grace window（由 `SHERPA_K8S_RUN_TIMEOUT_GRACE_SEC` 控制）。
- 其他阶段使用默认 grace（由 `SHERPA_K8S_STAGE_TIMEOUT_GRACE_SEC` 控制）。

---

## 5. LLM 参与边界与质量控制

### 5.1 LLM 参与节点

LLM（经 OpenAI/OpenRouter 配置）主要参与：

1. `plan`：分析目标与策略。
2. `synthesize`：生成 harness 与构建脚手架。
3. `fix_build`：在规则热修失败后进行补救编辑。
4. `improve-harness`：基于覆盖率反馈提出下一轮改进。

### 5.2 禁止 AI 介入的验证阶段

项目流程明确规定：`run` 与 `repro_crash` 阶段默认禁止 AI 参与验证判定，默认环境变量为 `SHERPA_VERIFY_STAGE_NO_AI=1`。该策略用于保证验证可重复性与结果可信度。

### 5.3 敏感信息脱敏

Web 服务对日志文本执行敏感信息清洗：

- API Key / Token / Secret / Password 模式匹配脱敏
- Bearer Token 脱敏
- 环境变量值替换脱敏

该策略降低了日志外泄风险，适合多角色协作排障。

---

## 6. 代码结构与职责

### 6.1 后端关键模块

1. `harness_generator/src/langchain_agent/main.py`
   - FastAPI 生命周期、路由、任务调度、k8s Job 启停、系统指标。
2. `harness_generator/src/langchain_agent/workflow_graph.py`
   - LangGraph 状态定义与节点路由，覆盖 build/run/coverage/re-* 关键决策。
3. `harness_generator/src/langchain_agent/k8s_job_worker.py`
   - 阶段执行器入口，负责读取 payload、执行单阶段并输出 result 文档。
4. `harness_generator/src/langchain_agent/job_store.py`
   - JobStore 抽象与 Postgres/SQLite 实现。
5. `harness_generator/src/fuzz_unharnessed_repo.py`
   - 与仓库构建和 fuzz 运行相关的底层执行逻辑。

### 6.2 前端关键模块

`frontend-next` 基于 Next 14 + React 18，典型职责包括：

- 配置输入（如 budget、模型、覆盖率循环参数）
- 实时状态查询与展示
- 日志与系统概览面板

核心组件包含 `ConfigPanel`、`TaskProgressPanel`、`LogPanel`、`SystemOverviewCard`。

### 6.3 基础设施与配置

- `k8s/base`：基础资源定义（web/frontend/postgres/service/ingress/configmap 等）
- `k8s/overlays/dev` 与 `k8s/overlays/prod`：环境差异化配置
- `.github/workflows`：dev/prod 部署、保护规则、PR 质量门禁

---

## 7. API 设计与外部契约

### 7.1 核心接口

后端暴露关键接口：

1. `POST /api/task`：提交任务。
2. `GET /api/task/{job_id}`：查询任务状态。
3. `POST /api/task/{job_id}/resume`：恢复任务。
4. `POST /api/task/{job_id}/stop`：停止任务。
5. `GET /api/tasks`：任务列表。
6. `GET /api/system` / `GET /api/metrics` / `GET /api/health`：系统状态与监控。
7. `GET/PUT /api/config`：运行时配置管理。

### 7.2 兼容性策略

接口保留了部分历史字段（例如 docker 相关字段）用于兼容，但在 k8s 原生模式下主要作为兼容钩子，不再是核心执行路径依赖。

---

## 8. 数据持久化与产物组织

### 8.1 数据库存储

`jobs` 表存储统一任务记录，关键字段包括：

- `job_id`（主键）
- `kind`（task/fuzz）
- `status`
- `repo`
- `created_at` / `updated_at`
- `payload_json`（JSONB，保存完整任务快照）

### 8.2 文件产物

关键产物路径包括：

1. `/shared/output/<repo>-<id>`：仓库级执行目录。
2. `/shared/output/_k8s_jobs/<job_id>/stage-*.json`：阶段结果契约。
3. `/app/job-logs/jobs/<job_id>.log`：任务聚合日志。

这些文件构成“执行证据链”，是论文中复现实验章节的重要材料。

---

## 9. CI/CD 与发布治理

### 9.1 发布路径

项目采用三段式路径：

1. 功能分支（`codex/*`）-> PR 到 `dev`
2. `dev` 验证通过 -> PR 到 `main`
3. `main` 工作流通过后进入生产

### 9.2 关键保护策略

- 禁止直接 push 到 `dev` / `main`
- `main` 仅接受来自 `dev` 的 PR
- 默认禁止管理员绕过保护规则强制合并

### 9.3 环境策略

- Dev：允许验证期重置（按工作流参数控制）
- Prod：强调稳定，禁用重置

---

## 10. 可观测性与排障方法

建议排障顺序：

1. 查看阶段结果 `stage-*.json` 与错误文本。
2. 查看任务聚合日志（job log）。
3. 查看 Pod 事件和容器日志。

常见问题类型：

- 构建失败（编译器/链接器/依赖）
- 运行阶段 idle 或 timeout
- k8s API 不可达导致部署步骤跳过
- 覆盖率提升未触发回环（与外层阶段调度策略相关）

---

## 11. 当前能力边界与风险

### 11.1 边界

1. 覆盖率改进轮次由参数允许，但是否多轮回跑受外层固定 stage 队列影响。
2. 不同目标仓库的 build 系统差异大，自动修复不保证收敛。
3. 复现链路依赖 run 阶段产物完整性，对共享存储可靠性有要求。

### 11.2 风险点

1. LLM 生成质量波动可能导致 harness 质量不稳定。
2. 大型仓库编译与运行成本高，对资源配额和超时参数敏感。
3. kubeconfig 可达性问题会直接影响 CI 部署闭环。

### 11.3 缓解策略

1. 规则热修优先、LLM 补救兜底。
2. 结构化错误分类与签名，减少重复无效修复。
3. dev/prod 分层发布与门禁约束，降低线上风险。

---

## 12. 复现实验建议（论文可直接引用）

建议以 `zlib` 等中等规模仓库作为基线样例，采用以下实验维度：

1. **成功率**：任务成功率、阶段成功率。
2. **效率**：从提交到首个可执行 fuzzer 的时间。
3. **鲁棒性**：失败后恢复成功率（resume/restart-to-plan）。
4. **质量**：崩溃复现率、无效 crash 比例。
5. **工程可维护性**：单任务日志可追溯性与故障定位时延。

对比实验可设置：

- 无 LLM 修复 vs 启用 LLM 修复
- 单阶段串行（历史方案）vs stage-per-job（当前方案）
- 无覆盖率改进 vs 启用覆盖率改进

---

## 13. 论文写作建议章节映射

可按以下章节映射本报告：

1. **引言**：自动化 fuzz 的工程痛点与研究价值。
2. **系统设计**：分层架构、状态机、数据契约。
3. **关键机制**：LLM 参与边界、错误恢复策略、复现链路。
4. **实现细节**：API、JobStore、K8s Job、CI/CD。
5. **实验与评估**：效率、成功率、复现率、稳定性。
6. **讨论**：边界、风险、未来工作。

---

## 14. 总结

Sherpa 的核心贡献在于：将“LLM 驱动的 harness 生成能力”与“云原生可审计执行链路”融合为一条可工程化落地的流水线。其价值不仅是“能自动跑 fuzz”，更在于“能以可追踪、可恢复、可发布治理的方式持续运行”。

对于论文作者而言，本项目提供了一个较完整的研究-工程结合样本：

- 在方法层展示“人机协同程序修复 + 状态机编排”的可行性；
- 在工程层展示“可观测、可回滚、可多环境发布”的生产化路径。
