---
name: "promefuzz-comprehender-pipeline"
description: "Runs PromeFuzz comprehender workflow (knowledge base, retrieval, purpose, usage, semantic relevance). Invoke when user needs document-driven semantic analysis."
---

# PromeFuzz Comprehender Pipeline

## 适用场景

- 需要基于文档理解库用途与函数语义
- 需要构建知识库并做检索
- 需要批量补充 API 函数的语义说明

## 输入参数模板

- `document_paths`: 文档文件/目录/URL 列表
- `output_path`: 知识库输出目录
- `knowledge_base_id`: 知识库标识（由初始化结果或约定命名）
- `api_collection`: API 集合（可选，用于批量函数理解）

## 标准工具调用顺序

1. `init_knowledge_base`
   - `document_paths`
   - `output_path`
2. `retrieve_documents`
   - `query`
   - `knowledge_base_id`
   - `top_k`（默认 3）
3. `comprehend_library_purpose`
   - `knowledge_base_id`
4. `comprehend_function_usage`
   - `function_name`
   - `knowledge_base_id`
5. `comprehend_all_functions`（可选）
   - `api_collection`
   - `knowledge_base_id`
6. `comprehend_function_relevance`（可选）
   - `api_collection`
   - `library_purpose`
   - `function_usages`

## 使用限制提示

- 当前仓库中部分 Comprehender 能力仍为占位实现
- 如果返回内容明显模板化，需要在报告中明确“结果为骨架能力输出”
- 先给出可用事实（文档数量、路径、工具返回结构），再给结论

## 输出报告建议

- 知识库初始化是否成功、文档数量
- 典型检索 query 与返回样例
- 库用途总结（若有）
- 函数用法理解结果（若有）
- 占位能力说明与后续改进建议

## 参考来源

- `prompts.py` 中 Comprehender 阶段提示词
