---
name: "promefuzz-full-analysis-orchestrator"
description: "Executes end-to-end PromeFuzz analysis across preprocessor and comprehender stages. Invoke when user asks for complete library analysis workflow."
---

# PromeFuzz Full Analysis Orchestrator

## 适用场景

- 用户要求“完整分析一个 C/C++ 库”
- 需要串联结构分析与语义分析
- 需要统一输出阶段结果与风险说明

## 输入清单

- 库源码路径（`source_paths`）
- 头文件路径（`header_paths`）
- 文档路径（`document_paths`）
- `compile_commands.json` 路径（可选）
- 输出根目录（如 `./output/<library_name>`）

## 执行编排

1. 执行 Preprocessor 阶段
   - `run_ast_preprocessor`
   - `extract_api_functions`
   - `build_library_callgraph`
   - `calculate_type_relevance`
2. 执行 Comprehender 阶段
   - `init_knowledge_base`
   - `retrieve_documents`（至少一次面向库用途的问题）
   - `comprehend_library_purpose`
   - `comprehend_function_usage`（关键函数）
   - `comprehend_function_relevance`（可选）
3. 汇总最终报告
   - 数据产物路径
   - 关键统计
   - 可用结论与占位能力说明

## 报告输出模板

- 分析对象：`<library_name>`
- 阶段一结果：函数数、API 数、调用边数
- 阶段二结果：文档数、用途总结、函数语义摘要
- 产物清单：`meta/api/callgraph/relevance/knowledge`
- 风险与限制：哪些工具当前是占位实现
- 下一步动作：可执行的验证或补强任务

## 失败回退策略

- 二进制不可用时，仅执行 Comprehender 阶段并标注缺失原因
- LLM 或向量检索不可用时，仅输出 Preprocessor 结构化分析
- 任一步骤失败都保留已成功产物，避免整链路中断

## 参考来源

- `prompts.py` 中完整工作流提示词
- `DEPLOY.md` 的部署与排障流程
