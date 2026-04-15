# Sherpa 代码结构整治路线图（可复用拆分版）

## 1. 当前结构问题（按风险排序）

### P0：超大文件耦合过高
- `harness_generator/src/langchain_agent/workflow_graph.py`（13k+ 行）
  - 节点逻辑、路由、提示词渲染、决策追踪、指标上报、修复策略混在一起。
- `harness_generator/src/langchain_agent/main.py`（5k+ 行）
  - API、k8s 调度、payload 组装、超时重试、上下文落盘耦合。
- `harness_generator/src/fuzz_unharnessed_repo.py`（6.9k+ 行）
  - seed 推断、质量评估、语料引导、构建/运行辅助交织。

### P1：跨模块边界不清
- 观测逻辑（decision trace / wf-metrics）散落在 `workflow_graph.py`。
- stage payload 构造是内联字典，复用性低，修改容易漏字段。

### P2：可测试性偏弱
- 关键逻辑大量内联，不利于单元测试和契约测试。

## 2. 拆分原则

1. **先抽纯函数与观测层**，不改状态机语义。  
2. **每次拆分只做一层边界**，确保可回归验证。  
3. **主流程拓扑不变**，先提升可维护性再改策略。  
4. **字段契约先固定再复用**，避免“拆分导致字段漂移”。  

## 3. 本轮已完成拆分

### 3.1 新增模块：`workflow_observability.py`
- 路径：
  - `harness_generator/src/langchain_agent/workflow_observability.py`
- 职责：
  - `record_decision_trace(...)`
  - `emit_fuzz_metrics(...)`
  - `decision_snapshot_from_state(...)`
- 收益：
  - 观测逻辑从 `workflow_graph.py` 抽离，后续其他节点/执行器可直接复用。

### 3.2 `workflow_graph.py` 对接新模块
- 保留原调用入口（`_record_decision_trace` / `_emit_fuzz_metrics`），内部委托到新模块。
- 状态机行为与字段不变，降低回归风险。

### 3.3 `main.py` 抽取 stage payload 组装
- 新增函数：
  - `_build_stage_payload(...)`
- 原先内联大字典改为函数调用。
- 收益：
  - 统一 payload 构造入口，后续加减字段不需要在主流程中找散点改动。

### 3.4 新增测试
- `tests/test_workflow_observability.py`
- `tests/test_main_stage_payload.py`
- `tests/test_workflow_target_scoring.py`
- `tests/test_workflow_coverage_decision.py`
- `tests/test_workflow_target_selection.py`
- `tests/test_workflow_selected_target_row.py`

### 3.5 Batch 2 进展（main.py 主循环瘦身）
- 已抽取并稳定：
  - `_handle_k8s_job_failure(...)`
  - `_handle_stage_dispatch_exception(...)`
  - `_finalize_stage_result(...)`
  - `_update_stage_node_pin(...)`
  - `_next_stage_from_result(...)`
- 主循环职责更聚焦为：阶段编排 + 调度。

### 3.6 Batch 3 进展（workflow_graph 热点拆分）
- 新增模块：
  - `harness_generator/src/langchain_agent/workflow_target_scoring.py`
  - `harness_generator/src/langchain_agent/workflow_coverage_decision.py`
  - `harness_generator/src/langchain_agent/workflow_target_selection.py`
- 已完成迁移：
  - target 评分纯逻辑（component/breakdown/seed penalty）
  - coverage improve 决策内核（seed/cold-start/plateau/replan）
  - selected targets 排序与执行优先级分配
  - selected target 单行构建 helper（缩短 `_build_selected_targets_doc`）
- 结果：`workflow_graph.py` 仍保留状态机与节点装配，但算法细节已外移。

## 4. 下一批拆分（建议顺序）

### Batch 2（低风险，高收益）
- 已完成“函数级”抽离，下一步可选“文件级”拆分：
  - 将 `main.py` 中 stage helper 进一步迁移到 `workflow_stage_runner.py`
  - 保持 `main.py` 只做 API + orchestration
- 目标：把主循环压缩成“调度编排”视图。

### Batch 3（中风险，核心收益）
- 已完成第一阶段：
  - `coverage` 决策、target scoring、selected-target 排序已抽离为独立模块
- 剩余建议：
  - 路由函数族 `_route_after_*` → `workflow_routes.py`
  - prompt 渲染与 degrade 处理 → `workflow_prompting.py`
  - repair snapshot/constraint memory → `workflow_repair_state.py`
- 目标：节点函数只保留业务决策，不再混杂通用基础逻辑。

### Batch 4（中高风险，收益大）
- 从 `fuzz_unharnessed_repo.py` 抽出：
  - seed profile / target inference → `seed_inference.py`
  - seed quality scoring → `seed_quality.py`
  - seed family classification → `seed_families.py`
- 目标：seed 质量与策略可独立迭代，避免牵动生成器主流程。

## 5. 验证门禁

每个 batch 合入前必须通过：

1. 目标单测 + 相关回归：
   - `tests/test_workflow_prompt_render_safe.py`
   - `tests/test_workflow_antlr_context.py`
   - `tests/test_k8s_job_worker.py`
2. 最小语法门禁：
   - `python -m py_compile` 覆盖改动文件。
3. 字段契约门禁：
   - decision trace、workflow_context、payload 字段不得回归丢失。

## 6. 预期结果

- `workflow_graph.py` 降到“节点 + 路由”核心职责。
- `main.py` 降到“API + 调度编排”核心职责。
- seed 体系可独立演进，不再绑死在单一巨文件。
- 新功能接入成本下降，字段回归风险降低。

## 7. 当前执行状态（2026-04-15）

### 7.1 已完成（本轮）
- `main.py`
  - stage payload 构建、失败重试、结果收口、节点 pin、next stage 解析均已函数化。
- `workflow_graph.py`
  - target scoring 已模块化（`workflow_target_scoring.py`）。
  - coverage improve 决策已模块化（`workflow_coverage_decision.py`）。
  - selected targets 排序/执行优先级已模块化（`workflow_target_selection.py`）。
  - selected target 单行构建已抽出 `_build_selected_target_row(...)`。
- 文档与测试
  - 新增/扩展测试覆盖上述模块化边界，确保行为等价。

### 7.2 验证结果（本轮）
- 通过（重点子集）：
  - `tests/test_main_stage_payload.py`
  - `tests/test_workflow_observability.py`
  - `tests/test_workflow_target_scoring.py`
  - `tests/test_workflow_coverage_decision.py`
  - `tests/test_workflow_target_selection.py`
  - `tests/test_workflow_selected_target_row.py`
  - `tests/test_workflow_run_detection.py`（selected targets 相关关键用例）
  - `tests/test_workflow_prompt_render_safe.py`
  - `tests/test_workflow_field_forwarding_contract.py`
  - `tests/test_k8s_job_worker.py::test_worker_forwards_restart_and_decision_trace_fields`
- 通过（语法门禁）：
  - `python -m py_compile` 覆盖 `main.py`、`workflow_graph.py` 与新拆分模块。
- 未通过（环境阻塞）：
  - 全量 `pytest tests -q` 中 `tests/test_api_stability.py` 相关用例因 PostgreSQL 不可达失败。
  - 错误形态：`psycopg.OperationalError: connection to 127.0.0.1:55432 refused`。
  - 该问题属于本地测试环境依赖，不是本轮重构逻辑回归。

### 7.3 剩余拆分任务（按优先级）
1. P0：`workflow_graph.py` 路由函数族拆分到 `workflow_routes.py`。
2. P1：prompt 渲染/降级处理拆分到 `workflow_prompting.py`。
3. P1：repair snapshot/constraint memory 拆分到 `workflow_repair_state.py`。
4. P2：`main.py` helper 文件级迁移到 `workflow_stage_runner.py`（当前仅函数级收口）。
5. P2：`fuzz_unharnessed_repo.py` 的 seed inference/quality/families 三段拆分（Batch 4）。

### 7.4 合入前建议
- 先确保本地 PostgreSQL（`127.0.0.1:55432`）可连通，再跑一次全量 `pytest tests -q`。
- 合入 PR 时附两类证据：
  - 子集回归通过截图/日志
  - 全量测试结果（或明确 DB 依赖阻塞说明）
