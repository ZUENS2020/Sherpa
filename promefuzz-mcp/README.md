# PromeFuzz MCP Tools

MCP (Model Context Protocol) tools for code analysis and comprehension based on PromeFuzz.

## Features

- **Preprocessor Module**: Extract code metadata from C/C++ libraries
  - AST preprocessing
  - API extraction
  - Call graph building
  - Relevance calculation
  - Complexity analysis

- **Comprehender Module**: Understand library semantics via LLMs
  - Document RAG knowledge base
  - Library purpose comprehension
  - Function usage understanding
  - Semantic relevance analysis

## Installation

```bash
# Clone the repository
cd Tools/promefuzz-mcp

# Install dependencies
pip install -e .

# Build processor binaries (automatic on first run)
python -m promefuzz_mcp build
```

## Configuration

Copy `config.template.toml` to `config.toml` and modify the settings:

```bash
cp config.template.toml config.toml
```

Required configurations:
- LLM settings (API keys, endpoints)
- Binary tool paths
- Embedding model for RAG

## Usage

### As MCP Server

```bash
# Start the MCP server
promefuzz-mcp

# Or with custom config
promefuzz-mcp --config /path/to/config.toml
```

### As Python Library

```python
from promefuzz_mcp.preprocessor import ASTPreprocessor
from promefuzz_mcp.comprehender import KnowledgeBase

# Preprocess a library
preprocessor = ASTPreprocessor(
    source_paths=["/path/to/source"],
    compile_commands_path="/path/to/compile_commands.json"
)
meta = preprocessor.run()

# Initialize knowledge base
kb = KnowledgeBase(document_paths=["/path/to/docs"])
kb.initialize()
```

## Available Tools

### Preprocessor

The Preprocessor module extracts code metadata from C/C++ libraries using Clang AST.

#### 1. run_ast_preprocessor
- **作用**: 使用 Clang AST 解析器处理 C/C++ 源代码，提取元数据
- **输入**: `source_paths`: 源文件/目录列表；`compile_commands_path`: 编译命令文件路径
- **输出**: 包含类和函数数量的元数据字典
- **依赖**: `processor/bin/preprocessor` 二进制工具

#### 2. extract_api_functions
- **作用**: 从头文件中识别并提取公共 API 函数
- **输入**: `header_paths`: 头文件路径列表；`meta_path`: AST预处理生成的meta.json
- **输出**: API函数集合（包含名称、位置、声明位置等）
- **依赖**: 需要先运行 `run_ast_preprocessor`

#### 3. build_library_callgraph
- **作用**: 构建库源代码的函数调用图
- **输入**: 源文件路径、compile_commands.json、API集合
- **输出**: 调用图数据（节点和边）
- **依赖**: `processor/bin/cgprocessor` 二进制工具

#### 4. build_consumer_callgraph
- **作用**: 构建库消费者代码的调用图
- **输入**: 消费者源代码路径、compile_commands.json
- **输出**: 消费者调用图数据

#### 5. extract_incidentals
- **作用**: 提取函数之间的附带关系
- **输入**: 调用图数据
- **输出**: 附带关系映射

#### 6. calculate_type_relevance
- **作用**: 基于函数参数/返回值的类型计算 API 函数之间的相关性
- **输入**: API集合、meta路径
- **输出**: 类型相关性分数

#### 7. calculate_class_relevance
- **作用**: 基于类成员关系计算函数之间的相关性
- **输入**: API集合、信息库路径
- **输出**: 类相关性分数

#### 8. calculate_call_relevance
- **作用**: 基于调用关系计算函数之间的相关性
- **输入**: API集合、调用图
- **输出**: 调用相关性分数

#### 9. calculate_complexity
- **作用**: 计算 API 函数的复杂度
- **输入**: API集合、信息库、调用图
- **输出**: 复杂度分数

#### 10. get_function_info
- **作用**: 获取特定函数的详细信息
- **输入**: 函数位置标识符、信息库路径
- **输出**: 函数名、签名、位置等

#### 11. calculate_complexity
- **作用**: 计算 API 函数的复杂度指标
- **输入**: API集合、信息库路径、调用图
- **输出**: 复杂度分析结果

### Comprehender

The Comprehender module uses LLMs to understand library semantics through RAG.

#### 1. init_knowledge_base
- **作用**: 从文档（文件/目录/URL）构建 RAG 知识库
- **输入**: `document_paths`: 文档路径列表；`output_path`: 输出路径
- **输出**: 知识库信息（文档数量等）
- **依赖**: 嵌入模型（如 nomic-embed-text）

#### 2. retrieve_documents
- **作用**: 使用 RAG 从知识库中检索相关文档
- **输入**: `query`: 查询字符串；`knowledge_base_id`: 知识库ID；`top_k`: 返回结果数
- **输出**: 相关文档摘要列表

#### 3. comprehend_library_purpose
- **作用**: 使用 LLM 理解库的整体目的和功能
- **输入**: 知识库ID
- **输出**: 库用途描述
- **工作流**: 检索文档 → LLM分析 → 生成描述

#### 4. comprehend_function_usage
- **作用**: 使用 LLM 理解特定函数的用法
- **输入**: 函数名、知识库ID
- **输出**: 函数使用说明
- **工作流**: 检索函数相关文档 → LLM分析 → 生成用法说明

#### 5. comprehend_all_functions
- **作用**: 批量理解 API 集合中所有函数的用法
- **输入**: API集合、知识库ID
- **输出**: 进度更新 + 所有函数的理解结果

#### 6. comprehend_function_relevance
- **作用**: 基于语义计算函数之间的相关性
- **输入**: API集合、库目的描述、函数用法映射
- **输出**: 函数相关性分析结果

## Workflow

### Overall Pipeline

```
+------------------------------------------------------------------+
|                        Preprocessor Stage                         |
+------------------------------------------------------------------+
|  Source Code  -->  run_ast_preprocessor  -->  meta.json          |
|                                                                  |
|  Header Files -->  extract_api_functions  -->  APICollection    |
|                                                                  |
|  Source Code  -->  build_library_callgraph -->  CallGraph        |
|                                                                  |
|  APICollection + Meta --> calculate_type_relevance --> Scores    |
+------------------------------------------------------------------+
                             |
                             v
+------------------------------------------------------------------+
|                       Comprehender Stage                         |
+------------------------------------------------------------------+
|  Documents  -->  init_knowledge_base  -->  RAG Knowledge Base     |
|                                                                  |
|  Knowledge Base  -->  retrieve_documents  -->  Relevant Docs      |
|                                                                  |
|  Knowledge Base  -->  comprehend_library_purpose --> Purpose      |
|                                                                  |
|  Knowledge Base  -->  comprehend_function_usage --> Usage Info    |
|                                                                  |
|  API + Purpose  -->  comprehend_function_relevance --> Relevance  |
+------------------------------------------------------------------+
```

### Preprocessor Workflow

1. **AST Preprocessing**: Use Clang AST to extract metadata from source files
2. **API Extraction**: Identify public API functions from headers
3. **Call Graph Building**: Build function call relationships
4. **Relevance Calculation**: Calculate various relevance scores (type, class, call)

### Comprehender Workflow

1. **Knowledge Base Initialization**: Build RAG index from documentation
2. **Document Retrieval**: Retrieve relevant docs for queries
3. **Purpose Comprehension**: Understand library purpose via LLM
4. **Function Usage**: Understand specific function usage via LLM
5. **Relevance Analysis**: Calculate semantic relevance between functions

## License

MIT
