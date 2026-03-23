# Harness Generator 后端

`harness_generator/` 是 Sherpa 的后端实现目录，包含 Web API、stage worker、工作流状态机、OpenCode 封装，以及非 OSS-Fuzz 仓库的处理逻辑。

当前线上主路径是：

- `FastAPI + Postgres + Kubernetes stage jobs`
- 非 OSS-Fuzz workflow
- `OpenCode` 原生运行于 k8s worker

## 目录结构

```text
harness_generator/
├── src/
│   ├── codex_helper.py
│   ├── fuzz_unharnessed_repo.py
│   └── langchain_agent/
│       ├── main.py
│       ├── workflow_graph.py
│       ├── workflow_common.py
│       ├── workflow_summary.py
│       └── prompts/
├── docs/
└── README.md
```

## 当前主职责

### `src/langchain_agent/main.py`
- 提供 FastAPI API
- 调度 stage job
- 记录任务、子任务、日志与状态
- 根据 `workflow_recommended_next` 决定下一阶段，而不是固定死链路
- 向前端暴露 `/api/task`、`/api/tasks`、`/api/system`、`/api/config` 等接口；接口口径以 `docs/API_REFERENCE.md` 为准

### `src/langchain_agent/workflow_graph.py`
- 定义工作流节点与路由
- 当前节点包括：
  - `init`
  - `plan`
  - `synthesize`
  - `build`
  - `fix_build`
  - `run`
  - `coverage-analysis`
  - `improve-harness`
  - `re-build`
  - `re-run`
  - `fix_crash`
- 负责：
  - `targets.json` schema 校验
  - coverage loop
  - env rebuild / build / rerun
  - plateau 收口
  - replan 有效性校验

### `src/fuzz_unharnessed_repo.py`
- 负责仓库 clone、build、run 的底层执行
- 实现 seed bootstrap 三段式逻辑
- 处理 repo examples 过滤
- 执行 OpenCode scaffolding passes
- 执行本地 build / run 命令并解析日志

### `src/codex_helper.py`
- 封装 OpenCode CLI 调用
- 管理 sentinel 与 idle timeout
- 原生执行 `opencode`
- 对只读命令（含 `grep/rg`）做白名单控制

## 当前数据产物

任务输出目录中的典型内容：

- `fuzz/PLAN.md`
- `fuzz/targets.json`
- `fuzz/target_analysis.json`
- `fuzz/build.py`
- `fuzz/*.cc`
- `fuzz/corpus/<fuzzer>/`
- `run_summary.json`
- `repro_context.json`
- `.repro_crash/`

## 当前关键约束

### 1. Target 元数据
`fuzz/targets.json` 当前要求每个 target 至少包含：

- `name`
- `api`
- `lang`
- `target_type`
- `seed_profile`

### 2. Seed profile
当前固定枚举：

- `parser-structure`
- `parser-token`
- `parser-format`
- `parser-numeric`
- `decoder-binary`
- `archive-container`
- `serializer-structured`
- `document-text`
- `network-message`
- `generic`

### 3. Coverage loop
- plateau 后优先在当前 target 上做原地 improve
- 连续无收益且预算允许时才 replan
- replan 必须有实质变化，否则停止

### 4. Build resilience
- `max_fix_rounds`
- `same_error_max_retries`
- `error_signature_before/after`
- `requires_env_rebuild`
- quick-check build

## 当前与历史实现的差异

旧版实现曾经存在：
- inner Docker OpenCode
- 固定阶段尾序列
- 缺少 `seed_profile`
- 缺少 `target_analysis.json`
- plateau 后直接反复回 `plan`

这些都不是当前主线行为。请以当前代码和本文档为准。
