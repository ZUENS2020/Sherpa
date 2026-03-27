---
name: "promefuzz-preprocessor-pipeline"
description: "Runs PromeFuzz preprocessor analysis pipeline (AST, API, callgraph, relevance). Invoke when user needs structural C/C++ code analysis outputs."
---

# PromeFuzz Preprocessor Pipeline

## 适用场景

- 需要对 C/C++ 库做结构化分析
- 需要产出 `meta.json`、API 函数列表、调用图
- 需要为后续 fuzz 或语义理解准备基础数据

## 输入参数模板

- `source_paths`: 源代码目录或文件列表
- `header_paths`: 头文件目录或文件列表
- `compile_commands_path`: `compile_commands.json` 路径（可选）
- `output_root`: 输出根目录（建议按库名隔离）

## 标准工具调用顺序

1. `run_ast_preprocessor`
   - `source_paths`
   - `compile_commands_path`（可选）
   - `output_dir` = `<output_root>/meta`
2. `extract_api_functions`
   - `header_paths`
   - `meta_path` = `<output_root>/meta/meta.json`
   - `output_path` = `<output_root>/api/api_functions.json`
3. `build_library_callgraph`
   - `source_paths`
   - `compile_commands_path`（可选）
   - `api_collection`（可从上一步结果传入）
   - `output_path` = `<output_root>/callgraph/callgraph.json`
4. `calculate_type_relevance`
   - `api_collection`
   - `meta_path`
   - `output_path` = `<output_root>/relevance/type_relevance.json`

## 结果校验

- `meta/meta.json` 存在且包含 `functions` 或 `classes`
- `api/api_functions.json` 中 `count > 0`
- `callgraph/callgraph.json` 中 `edges` 字段存在
- 若 `type_relevance` 为空，标记为“当前实现占位结果”

## 输出报告建议

- 源文件数量、头文件数量
- AST 抽取的函数/类总数
- API 数量 Top N
- 调用边数量
- 下一步建议（如进入语义理解阶段）

## 参考来源

- `prompts.py` 中 Preprocessor 阶段提示词
