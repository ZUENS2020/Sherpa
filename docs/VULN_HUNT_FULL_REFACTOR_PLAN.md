# Sherpa 全面改造计划：持续漏洞挖掘主引擎 + 异步 Fuzz 验证引擎

## 1. 目标与范围

### 1.1 目标
- 把 Sherpa 从单线性 `analysis -> plan -> synthesize -> build -> run -> coverage-analysis` 模式，升级为“持续漏洞挖掘优先”的双引擎系统。
- 主引擎持续发现漏洞候选，验证引擎异步消费并验证，主引擎不中断继续发现下一个候选。
- 复用现有 fuzz 执行与修复能力，避免推倒重写。

### 1.2 非目标
- 本次不引入全新外部服务依赖（可优先使用现有 DB/队列/k8s 基础设施）。
- 不改变现有 API 路径语义（只做兼容扩展字段）。

## 2. 目标架构

### 2.1 双引擎架构
- `Vuln-Hunt Engine`（主引擎，持续运行）
  - 持续分析代码，产出 `vuln_candidate`。
  - 负责候选去重、排序、派发、回收与重评分。
- `Fuzz-Validation Engine`（验证引擎，异步并发）
  - 消费候选，执行 `plan/synthesize/build/run/coverage/crash-triage`。
  - 输出 `validation_result`，回写候选状态。

### 2.2 关键层
- `Orchestrator`
  - 预算、并发、优先级、退避、重试调度。
- `Knowledge/Memory Layer`
  - 候选库、签名库、失败经验、约束记忆。
- `Artifacts & Observability`
  - decision trace、评分拆解、降级原因、验证证据链全量落盘。

## 3. 主流程（目标态）

1. `analysis`：构建代码事实与索引（调用图、路径、接口、历史上下文）。
2. `vuln-hunt`：持续循环生成漏洞候选（主循环）。
3. `dispatch`：候选入 `validation_queue`。
4. `validation`：fuzz 工作流验证候选。
5. `feedback`：验证结果回写候选库与策略。
6. `vuln-hunt` 继续下一轮，不阻塞于单个候选。

## 4. 数据契约（必须统一）

### 4.1 `vuln_candidate`
- `candidate_id: string`（幂等主键）
- `repo: string`
- `target_api: string`
- `target_file: string`
- `signal_type: string`（如 `mem_oob_candidate`）
- `security_signals: string[]`
- `evidence: Array<{evidence_id, signal_id, severity, confidence, source_path, line, summary}>`
- `vuln_likelihood: number (0-1)`
- `exploitability: number (0-1)`
- `reachability_confidence: number (0-1)`
- `detectability_confidence: number (0-1)`
- `priority: number`
- `status: enum[pending, validating, confirmed, rejected, inconclusive, cooling]`
- `created_at/updated_at: timestamp`

### 4.2 `validation_result`
- `candidate_id: string`
- `status: enum[confirmed, rejected, inconclusive]`
- `crash_signature: string`
- `repro_artifacts: object`
- `coverage_delta: object`
- `failure_reason: string`
- `validator_round: number`
- `created_at: timestamp`

### 4.3 `signature_cluster`
- `signature: string`（`sanitizer + stack_top + key_frame_hash`）
- `count: number`
- `last_seen: timestamp`
- `sources: string[]`

## 5. 队列与调度

### 5.1 队列
- `validation_queue`：待验证候选。
- `retry_queue`：可重试候选（带退避）。
- `deadletter_queue`：多次失败候选。

### 5.2 并发与预算
- 主引擎并发：按 repo/path shard 分片。
- 验证并发：按 worker 资源动态调度。
- 限流策略：
  - 同 `crash_signature` 限流，避免重复烧资源。
  - 连续低收益候选降权并冷却。
- 预算硬阈值：
  - token/time/cost/round 数都需有上限。

## 6. 评分与选择（风险优先）

### 6.1 目标排序
- 主排序以风险为主，覆盖率为辅：
  - `vuln_likelihood`、`exploitability`、`reachability_confidence` 主导。
  - `coverage_gap`、`complexity_depth`、`api_relevance` 仅参考。

### 6.2 评分可解释
- 每个候选必须落盘：
  - `score_total`
  - `security_score_breakdown`
  - `penalty_reason`
- `decision_trace.jsonl` 必含：
  - `choose_candidate`
  - `choose_seed`
  - `choose_repair`
  - `strategy_delta`

## 7. 与现有 Sherpa 功能的关系

### 7.1 保留
- 现有 `synthesize/build/run/coverage/crash-triage/repair` 执行能力。
- k8s worker、日志、工件管理、API 主体路径。

### 7.2 升级
- 新增 `vuln-hunt` 阶段（`analysis` 之后）。
- plan 输入从“coverage-first”改为“candidate-first”。
- coverage loop 增加“候选验证进展”语义，不再只看覆盖率。

### 7.3 清理
- 移除冲突旧契约（旧字段透传、旧 seed 键、静默降级）。
- 所有降级必须可观测（禁止 silent fallback）。

## 8. 上下文与状态管理

### 8.1 双文件上下文继续沿用
- `fuzz/context/control_context.json`：调度硬参数。
- `fuzz/context/workflow_context.json`：业务状态与决策上下文。

### 8.2 业务字段归一
- 新增命名空间：
  - `security_*`
  - `vuln_*`
  - `candidate_*`
  - `validation_*`
- 保证跨阶段保真，不回退旧 payload 透传。

## 9. Prompt/Skill 合同

### 9.1 `vuln-hunt` 合同
- 必须输出：
  - `VULN_HYPOTHESES`
  - `security_evidence[]`
  - `vuln_candidate_inventory[]`
- 每条假设必须引用 `evidence_id`。

### 9.2 `plan` 合同
- 必须声明：风险优先，coverage 仅参考。
- 必须输出候选排序拆解与例外说明。

### 9.3 安全渲染
- 所有节点必须走安全渲染路径。
- 模板异常只降级，不中断；必须写：
  - `prompt_render_degraded`
  - `prompt_render_issue`

## 10. 实施阶段（不做 PoC，直接全量改造）

### Phase 1：架构落地
- 引入 `vuln-hunt` 阶段与候选队列。
- 建立 `vuln_candidate`/`validation_result` 契约与存储。
- 路由接入 candidate-first 验证流。

### Phase 2：策略落地
- 风险优先打分切换。
- internal/private API 例外策略与证据记录。
- 签名去重全链路统一。

### Phase 3：执行落地
- 验证引擎并行消费队列。
- retry/deadletter/冷却策略上线。
- seed/repair 与候选状态联动。

### Phase 4：可观测与治理
- API 字段扩展（兼容新增）。
- trace、snapshot、评分、降级全量可见。
- 监控告警（空转、重复签名、无增量循环）。

### Phase 5：收口清理
- 删除 legacy 透传与冲突契约。
- 更新文档、测试与运维手册。
- 统一开关与默认策略。

## 11. 验收标准

- 主引擎持续运行，验证引擎持续消费，无单点阻塞。
- 候选验证端到端链路稳定（可追踪、可复现、可回写）。
- 同签名重复空转显著下降。
- 在长任务中可稳定观察“发现 -> 验证 -> 继续发现”闭环。
- 全量回归测试通过，且无旧契约残留。

## 12. 风险与控制

### 12.1 风险
- 状态复杂度提升，跨阶段一致性风险变大。
- 并发提升后资源竞争与成本上升。
- 候选质量不足时可能导致验证队列拥塞。

### 12.2 控制
- 幂等主键 + 签名去重 + 限流冷却。
- 多维预算硬阈值（time/token/cost/round）。
- 强制结构化降级与死信隔离。
- 关键阶段回滚开关（策略层可回退，执行层不回退）。

## 13. 推荐默认配置

- `SHERPA_VULN_HUNTING_ENABLED=1`
- `SHERPA_VULN_SCORE_MODE=risk_first_v1`
- `SHERPA_VULN_INTERNAL_API_MIN_SCORE=0.75`
- `SHERPA_VULN_MIN_EVIDENCE_CONFIDENCE=0.45`
- `SHERPA_VULN_TOPK=24`

## 14. 立即执行清单（首轮）

1. 增加 `vuln-hunt` 阶段与路由入口。  
2. 定义并落盘 `vuln_candidate` 与 `validation_result` 契约。  
3. 接通 `validation_queue` 消费并复用现有 fuzz 验证流程。  
4. 切换 `plan` 为 candidate-first 排序。  
5. 补齐 decision trace 与降级可观测字段。  

## 15. 参考项目路径（本计划依据）

### 15.1 上游参考仓库（原始来源）
- GitHub：
  - [o2lab/afc-crs-all-you-need-is-a-fuzzing-brain](https://github.com/o2lab/afc-crs-all-you-need-is-a-fuzzing-brain)

### 15.2 本机已下载的分析副本路径（当前会话）
- 本地目录：
  - `/tmp/fb_docs`
- 本次重点对照文件（示例）：
  - `/tmp/fb_docs/01_architecture.md`
  - `/tmp/fb_docs/02_worker_strategy.md`
  - `/tmp/fb_docs/06_suspicious_point_lifecycle.md`
  - `/tmp/fb_docs/07_agent_design.md`
  - `/tmp/fb_docs/tools.yaml`
  - `/tmp/fb_docs/suspicious_point_agent.py`
  - `/tmp/fb_docs/function_analysis_agent.py`
  - `/tmp/fb_docs/pov_agent.py`
  - `/tmp/fb_docs/seed_agent.py`
  - `/tmp/fb_docs/mcp_factory.py`
  - `/tmp/fb_docs/analyzer.py`
  - `/tmp/fb_docs/coverage.py`
  - `/tmp/fb_docs/pov.py`
