# PromeFuzz MCP（当前实现口径）

本 README 只描述**当前代码已实现并可用**的能力，以及尚未完成的 TODO。

## 当前定位

`promefuzz-mcp` 是一个面向 C/C++ 库分析的 MCP 服务与工具集合，当前在 Sherpa 中主要承担：

1. 预处理源码/头文件，提取候选 API 与基础调用关系。
2. 产出可被工作流消费的分析工件（`preprocess.json`、`coverage_hints.json`）。
3. 作为任务级 companion 的 HTTP MCP 服务，供 OpenCode 在 `analysis/plan/synthesize` 阶段读取证据。

## 当前已实现能力

### 1) MCP 服务启动模式

`promefuzz_mcp.server` 当前支持两种启动模式：

1. `stdio`（兼容本地 CLI/调试）
2. `streamable-http`（用于集群内 HTTP MCP）

示例：

```bash
# stdio
python -m promefuzz_mcp.server start --skip-build --transport stdio

# HTTP MCP
python -m promefuzz_mcp.server start \
  --skip-build \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 18080 \
  --mcp-path /mcp
```

### 2) 预处理相关工具（可用）

以下工具有实际实现并可产出结果：

1. `run_ast_preprocessor`：调用 Clang AST 预处理二进制生成 `meta.json`。
2. `extract_api_functions`：从 header + meta 中提取 API 函数集合。
3. `build_library_callgraph`：构建基础调用边集合并导出 JSON。

### 3) RAG 能力（OpenRouter Embedding，已实现）

1. `init_knowledge_base`：真实初始化知识库（文档收集、切片、索引落盘）。
2. `retrieve_documents`：优先使用 OpenRouter embedding 相似度检索；不可用时自动降级 lexical 检索。
3. companion 会把知识库状态写入任务级状态文件：
   - `embedding_provider/embedding_model/embedding_ok`
   - `rag_degraded/rag_degraded_reason`
   - `semantic_query_count/semantic_hit_rate/cache_hit_rate`
   - `rag_ok`、文档数、chunk 数、知识库路径。

默认运行策略：

1. 默认仅启用上述预处理主线工具。
2. `SHERPA_PROMEFUZZ_ENABLE_RAG=1`（默认）时启用 `init_knowledge_base/retrieve_documents`。
3. `SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER=1`（默认）时启用 `comprehend_*` 语义工具。
4. 语义工具输出统一结构：`claim/evidence[]/confidence/limitations/degraded/degraded_reason`。

### 3) 与 Sherpa 工作流的实际接入（已接入）

当前主流程接线（代码已接入）：

1. 每个任务启动时创建独立 PromeFuzz companion Pod + Service（任务级生命周期）。
2. companion 周期性产出：
   - `/shared/output/_k8s_jobs/<job_id>/promefuzz/status.json`
   - `/shared/output/_k8s_jobs/<job_id>/promefuzz/preprocess.json`
   - `/shared/output/_k8s_jobs/<job_id>/promefuzz/coverage_hints.json`
3. worker 会把任务级 MCP URL 注入 OpenCode runtime config（`mcp`）。
4. `analysis` 阶段会读取 companion 工件并合并到 `fuzz/analysis_context.json`。
5. `plan/synthesize`（含 repair 分轨）已加 MCP 证据优先约束；MCP 不可用时 degraded 继续。

## 当前运行限制

1. 主要针对 C/C++ 分析路径；其他语言能力未完善。
2. 调用图与相关性计算目前是基础版，不等同于完整语义分析。
3. MCP 服务可用不代表所有工具都达到生产级精度（见下方 TODO）。

## TODO（未完成/占位能力）

以下能力在代码中仍是 TODO、占位实现或返回固定/空结果，当前不应当按“已实现功能”对外承诺：

### A. 预处理/相关性

1. `calculate_type_relevance`：当前为 TODO/占位。
2. `calculate_class_relevance`：当前为占位实现。
3. `calculate_call_relevance`：当前为占位实现。
4. complexity/incidental 等模块仍有 placeholder。

### B. Comprehender（剩余增强项）

1. 目前已提供证据化输出，但仍属于轻量启发式总结，尚未接完整推理模型链路。
2. `comprehend_function_relevance` 当前基于 usage overlap 近似计算，后续可替换为更强语义模型。

### C. 其他工具

1. `get_function_info`：当前返回示例值（`example_func`），未接真实查询逻辑。
2. `llm/client` 仍有 placeholder 路径未落地。

## 推荐使用方式（当前阶段）

如果你在 Sherpa 中使用 PromeFuzz，建议按以下口径：

1. 把它当作“分析增强与候选信号来源”，而不是最终漏洞判定引擎。
2. 关键决策仍结合 build/run/crash 证据与 workflow 状态机。
3. 对 README 中 TODO 区块列出的能力，不作为生产承诺依赖。

## 开发与调试

系统依赖（C++ 处理器）：

```bash
sudo apt-get install -y clang llvm libclang-dev nlohmann-json3-dev
```

在 Sherpa 的 k8s 默认运行镜像中，这些依赖应在镜像构建阶段预装；`init` 仅做依赖检查，不在运行期安装。

RAG embedding 运行时依赖（K8s secret）：

1. `OPENROUTER_EMBEDDING_API_KEY`
2. `OPENROUTER_EMBEDDING_MODEL`

默认安装不依赖本地 embedding 模型栈（不需要 `sentence-transformers/torch`，因此不会拉取 CUDA 相关依赖）。

默认 secret 名：`sherpa-openrouter-embedding`（可通过 `SHERPA_K8S_OPENROUTER_EMBEDDING_SECRET_NAME` 覆盖）。

```bash
cd promefuzz-mcp
pip install -e .

# 构建处理器二进制（首次或缺失时）
python -m promefuzz_mcp.server build

# 检查二进制是否可用
python -m promefuzz_mcp.server check
```
