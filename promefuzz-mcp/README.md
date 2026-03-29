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

### B. Comprehender / RAG

1. `retrieve_documents`：当前未完成，返回空结果。
2. `KnowledgeBase.retrieve(...)`：当前为占位，未形成真实向量检索闭环。
3. `comprehend_library_purpose`：当前为固定模板输出。
4. `comprehend_function_usage`：当前为固定模板输出。
5. `comprehend_all_functions`：当前仅进度框架。
6. `comprehend_function_relevance`：当前仅框架。

### C. 其他工具

1. `get_function_info`：当前返回示例值（`example_func`），未接真实查询逻辑。
2. `llm/client` 仍有 placeholder 路径未落地。

## 推荐使用方式（当前阶段）

如果你在 Sherpa 中使用 PromeFuzz，建议按以下口径：

1. 把它当作“分析增强与候选信号来源”，而不是最终漏洞判定引擎。
2. 关键决策仍结合 build/run/crash 证据与 workflow 状态机。
3. 对 README 中 TODO 区块列出的能力，不作为生产承诺依赖。

## 开发与调试

```bash
cd promefuzz-mcp
pip install -e .

# 构建处理器二进制（首次或缺失时）
python -m promefuzz_mcp.server build

# 检查二进制是否可用
python -m promefuzz_mcp.server check
```

