# PromeFuzz MCP 技术说明（当前实现）

本文档只描述当前分支已实现的 PromeFuzz MCP 集成能力，覆盖：

1. 任务级 companion Pod 生命周期  
2. HTTP MCP 接入 OpenCode 的方式  
3. 分析工件与状态文件契约  
4. 所有新增字段（任务 API 视图）语义与来源  
5. OpenRouter embedding 配置与降级策略  

## 1. 架构总览

PromeFuzz 采用“任务级 companion + 阶段内调用”模式：

1. 控制面（FastAPI，`main.py`）在任务启动时创建 `sherpa-promefuzz-<job_id[:10]>` Pod 与同名 Service。
2. companion Pod 同时运行：
   - `promefuzz_companion.py`（周期产出分析工件）
   - `python -m promefuzz_mcp.server start --transport streamable-http`（HTTP MCP 服务）
3. stage worker（`k8s_job_worker.py`）在执行前把 companion MCP URL 注入 OpenCode runtime config。
4. `analysis/plan/synthesize`（含 repair 分轨）通过同一 MCP 连接消费证据。
5. 任务结束/失败/取消时，控制面统一回收 companion Pod + Service。

## 2. 生命周期细节

### 2.1 创建与复用

- 函数：`_k8s_start_analysis_companion(job_id)`
- 行为：
  1. 先检查同名 Pod/Service。
  2. 若 Pod Running 且 Service 存在，则直接复用。
  3. 否则删除旧资源并重建。

### 2.2 停止与回收

- 函数：`_k8s_stop_analysis_companion(pod_name, service_name)`
- 调用时机：任务主执行 `finally` 路径（无论成功、失败、取消都会执行）。

### 2.3 fail-open 语义

- companion 不可用不会阻断主流程。
- 任务继续执行并写 `analysis_companion_error` / `analysis_degraded` 相关状态。

## 3. MCP 接入与传输协议

### 3.1 协议与地址

- 传输：`streamable-http`
- 默认监听：`0.0.0.0:18080`
- 默认路径：`/mcp`
- 任务内 URL 形态：
  `http://sherpa-promefuzz-<jobid10>.<namespace>.svc.cluster.local:18080/mcp`

### 3.2 OpenCode 注入

- worker 入口在执行 stage 前调用：
  1. `_resolve_analysis_companion_url(...)`
  2. `_merge_opencode_mcp_servers(url)`
- 注入结果：
  - `SHERPA_OPENCODE_MCP_SERVERS_JSON` 增加 `promefuzz` remote server
  - `SHERPA_OPENCODE_MCP_URL` 同步写入
- `persistent_config.build_opencode_runtime_config(...)` 会把 MCP 配置落到 runtime config，供 `opencode run` 使用。

## 4. MCP 工具能力（当前）

## 4.1 预处理工具（稳定可用）

1. `run_ast_preprocessor`
2. `extract_api_functions`
3. `build_library_callgraph`

## 4.2 RAG 与语义工具（已接入）

1. `init_knowledge_base`
2. `retrieve_documents`
3. `comprehend_library_purpose`
4. `comprehend_function_usage`
5. `comprehend_all_functions`
6. `comprehend_function_relevance`

### 4.3 开关

- `SHERPA_PROMEFUZZ_ENABLE_RAG`（默认 `1`）
- `SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER`（默认 `1`）

## 5. OpenRouter embedding 契约

## 5.1 Secret 与字段

K8s secret（默认名 `sherpa-openrouter-embedding`）仅两项：

1. `OPENROUTER_EMBEDDING_API_KEY`
2. `OPENROUTER_EMBEDDING_MODEL`

可通过环境变量覆盖 secret 名：

- `SHERPA_K8S_OPENROUTER_EMBEDDING_SECRET_NAME`

### 5.2 固定地址

- embedding API base：`https://openrouter.ai/api/v1`
- embedding endpoint：`/embeddings`

### 5.3 降级策略

当 key/model 缺失、API 请求失败、返回向量异常时：

1. 不中断任务（fail-open）。
2. `KnowledgeBase.retrieve()` 自动回退 lexical 检索。
3. 状态字段标记：
   - `embedding_ok=false`
   - `rag_degraded=true`
   - `rag_degraded_reason=<具体错误>`

## 6. companion 工件契约

任务目录：

`/shared/output/_k8s_jobs/<job_id>/promefuzz/`

产物：

1. `status.json`：companion 运行状态、MCP 状态、RAG/embedding 指标
2. `preprocess.json`：预处理分析结果（inventory、api、rag 摘要）
3. `coverage_hints.json`：候选目标与覆盖改进线索

另外，workflow analysis 阶段会生成：

`<repo_root>/fuzz/analysis_context.json`

用于 `plan/synthesize` 消费。

## 7. 新增字段（任务 API 视图）

以下字段由 `main.py::_enrich_job_view` 注入，出现在 `GET /api/task/{id}` 与 `GET /api/tasks` 项目视图中。

| 字段 | 类型 | 来源 | 语义 |
|---|---|---|---|
| `analysis_companion_pod` | string\|null | job state | companion Pod 名 |
| `analysis_companion_service` | string\|null | job state | companion Service 名 |
| `analysis_companion_url` | string | status/job state | MCP URL |
| `analysis_companion_ready` | bool | status/job state | MCP 是否可用 |
| `analysis_companion_active` | bool | job state | companion 是否处于活动期 |
| `analysis_companion_error` | string\|null | job state | companion 生命周期错误 |
| `analysis_companion_last_error` | string\|null | status | 最近一次 companion 错误 |
| `analysis_companion_stopped_at` | number\|null | job state | companion 停止时间戳 |
| `analysis_companion_state` | string | status.json | companion 状态（starting/running/ready/idle/degraded 等） |
| `analysis_companion_backend` | string | status.json | 当前分析后端标识 |
| `analysis_companion_candidate_count` | int | status.json | 推荐候选数 |
| `analysis_companion_updated_at` | string | status.json | 状态更新时间 |
| `analysis_companion_repo_root` | string | status.json | 解析出的仓库根目录 |
| `analysis_companion_status_error` | string | status.json | 当前状态错误文本 |
| `analysis_companion_preprocess_path` | string | status.json | preprocess 工件路径 |
| `analysis_companion_coverage_hints_path` | string | status.json | coverage hints 工件路径 |
| `analysis_companion_rag_ok` | bool | status.json | RAG 流程是否成功 |
| `analysis_companion_rag_knowledge_base_path` | string | status.json | 知识库目录 |
| `analysis_companion_rag_document_count` | int | status.json | 文档数 |
| `analysis_companion_rag_chunk_count` | int | status.json | chunk 数 |
| `analysis_companion_embedding_provider` | string | status.json | embedding provider（当前为 openrouter） |
| `analysis_companion_embedding_model` | string | status.json | embedding 模型名 |
| `analysis_companion_embedding_ok` | bool | status.json | embedding 通道是否可用 |
| `analysis_companion_rag_degraded` | bool | status.json | 是否进入降级 |
| `analysis_companion_rag_degraded_reason` | string | status.json | 降级原因 |
| `analysis_companion_semantic_query_count` | int | status.json | 语义查询次数 |
| `analysis_companion_semantic_hit_count` | int | status.json | 语义检索命中次数 |
| `analysis_companion_semantic_hit_rate` | float | status.json | 语义命中率 |
| `analysis_companion_cache_hit_rate` | float | status.json | 知识库缓存命中率（当前为 0/1 近似） |

## 8. `status.json` 关键字段说明

常见字段：

1. 运行态：`state`, `updated_at`, `error`, `last_error`
2. MCP：`mcp_url`, `mcp_ready`, `mcp_port`, `mcp_path`
3. 分析结果：`analysis_backend`, `candidate_count`, `preprocess_path`, `coverage_hints_path`
4. RAG/embedding：`rag_ok`, `rag_degraded`, `rag_degraded_reason`, `embedding_provider`, `embedding_model`, `embedding_ok`
5. 语义统计：`semantic_query_count`, `semantic_hit_count`, `semantic_hit_rate`, `cache_hit_rate`

## 9. Prompt/Skill 约束（与 MCP 的关系）

当前约束已经统一为：

1. `analysis`：先预处理 MCP 证据，再补充语义证据；不可用时写 degraded 原因。
2. `plan/synthesize`（含 repair）：优先消费 companion 证据，语义证据可用时必须引用具体证据行；不可用时必须显式 degraded 说明。
3. 不允许静默跳过 MCP 状态。

## 10. 排障指引

## 10.1 companion 起不来

优先检查：

1. `analysis_companion_error`
2. `analysis_companion_state`
3. companion Pod 日志与 `status.json.error`

## 10.2 MCP 不可用

检查：

1. `analysis_companion_url` 是否可解析
2. `analysis_companion_ready` 是否为 `true`
3. worker 是否正确注入 `SHERPA_OPENCODE_MCP_SERVERS_JSON`

## 10.3 embedding 不生效

检查：

1. secret `sherpa-openrouter-embedding` 是否存在并包含两项键
2. `analysis_companion_embedding_ok` 是否为 `true`
3. `analysis_companion_rag_degraded_reason` 的具体错误

