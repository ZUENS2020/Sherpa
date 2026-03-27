"""
PromeFuzz MCP Prompts for OpenCode

This module contains prompts for each stage of the PromeFuzz MCP workflow.
Use these prompts to guide OpenCode in calling the MCP tools effectively.
"""

from typing import Optional


# ============================================================================
# Stage 1: Preprocessor - Code Analysis
# ============================================================================

PROMPT_AST_PREPROCESSOR = """
## 任务：AST 预处理

使用 `run_ast_preprocessor` 工具对源代码进行 AST 分析。

### 输入参数
- `source_paths`: 源代码文件或目录的路径列表
- `compile_commands_path` (可选): compile_commands.json 的路径
- `output_dir` (可选): 输出目录，默认 "./output/meta"

### 操作步骤
1. 确定要分析的源代码路径
2. 如果有 compile_commands.json，提供其路径以获得更准确的编译选项
3. 调用工具执行 AST 预处理

### 输出解析
返回的字典包含：
- `source_files`: 处理的源文件数量
- `classes`: 发现的类数量
- `functions`: 发现的函数数量
- `output_file`: meta.json 文件路径

### 生成文件
- `./output/meta/meta.json`: 持久化的元数据文件

### 示例
```
请分析 /path/to/libcurl 目录中的源代码，提取元数据，输出到 ./output/libcurl/meta
```
"""

PROMPT_EXTRACT_API_FUNCTIONS = """
## 任务：提取 API 函数

使用 `extract_api_functions` 工具从头文件中提取公共 API 函数。

### 输入参数
- `header_paths`: 头文件或目录的路径列表
- `meta_path`: AST 预处理生成的 meta.json 文件路径
- `output_path` (可选): 输出路径，默认 "./output/api/api_functions.json"

### 前置条件
必须先运行 `run_ast_preprocessor` 获取 meta.json

### 操作步骤
1. 提供头文件路径（通常是 include 目录）
2. 提供 AST 预处理生成的 meta.json 路径
3. 调用工具提取 API 函数

### 输出解析
返回的字典包含：
- `count`: API 函数总数
- `functions`: 函数列表，每个包含 name, header, loc, decl_loc
- `output_file`: api_functions.json 文件路径

### 生成文件
- `./output/api/api_functions.json`: API 函数列表

### 示例
```
从头文件 /path/to/libcurl/include 提取 API 函数，元数据来自 ./output/libcurl/meta/meta.json
```
"""

PROMPT_BUILD_CALLGRAPH = """
## 任务：构建调用图

使用 `build_library_callgraph` 工具构建库的函数调用图。

### 输入参数
- `source_paths`: 源代码文件或目录的路径列表
- `compile_commands_path` (可选): compile_commands.json 的路径
- `api_collection` (可选): API 集合
- `output_path` (可选): 输出路径，默认 "./output/callgraph/callgraph.json"

### 操作步骤
1. 提供源代码路径
2. 调用工具构建调用图

### 输出解析
返回的字典包含：
- `nodes`: 函数节点列表
- `edges`: 调用关系边列表
- `output_file`: callgraph.json 文件路径

### 生成文件
- `./output/callgraph/callgraph.json`: 调用图数据

### 示例
```
为 libpng 库构建函数调用图，输出到 ./output/libpng/callgraph/callgraph.json
```
"""

PROMPT_CALCULATE_TYPE_RELEVANCE = """
## 任务：计算类型相关性

使用 `calculate_type_relevance` 工具基于函数参数和返回类型计算 API 函数之间的相关性。

### 输入参数
- `api_collection`: API 函数集合
- `meta_path`: meta.json 文件路径
- `output_path` (可选): 输出路径，默认 "./output/relevance/type_relevance.json"

### 操作步骤
1. 确保已有 API 集合和元数据
2. 调用工具计算类型相关性

### 输出解析
返回的字典包含：
- `relevance`: 函数对之间的相关性分数
- `output_file`: type_relevance.json 文件路径

### 生成文件
- `./output/relevance/type_relevance.json`: 类型相关性数据

### 示例
```
计算 libjpeg API 函数之间的类型相关性
```
"""

PROMPT_GET_FUNCTION_INFO = """
## 任务：获取函数信息

使用 `get_function_info` 工具获取特定函数的详细信息。

### 输入参数
- `function_location`: 函数位置标识符（通常是文件:行号格式）
- `info_repo_path`: 信息库路径

### 操作步骤
1. 确定要查询的函数位置
2. 调用工具获取函数信息

### 输出解析
返回的字典包含：
- `name`: 函数名
- `signature`: 函数签名
- `location`: 函数位置

### 示例
```
获取函数 png_read_image 的详细信息
```
"


# ============================================================================
# Stage 2: Comprehender - Semantic Understanding
# ============================================================================

PROMPT_INIT_KNOWLEDGE_BASE = """
## 任务：初始化 RAG 知识库

使用 `init_knowledge_base` 工具从文档构建 RAG 知识库。

### 输入参数
- `document_paths`: 文档路径列表（文件、目录或 URL）
- `output_path` (可选): 知识库输出路径，默认 "knowledge_db"

### 操作步骤
1. 收集库的文档（README、API 文档、示例代码等）
2. 提供文档路径列表
3. 调用工具初始化知识库

### 输出解析
返回的字典包含：
- `output_path`: 知识库路径
- `document_count`: 处理的文档数量

### 示例
```
为 zlib 库初始化知识库，文档路径包括 /docs, /README.md, /api.md
```
"""

PROMPT_RETRIEVE_DOCUMENTS = """
## 任务：检索相关文档

使用 `retrieve_documents` 工具从知识库中检索相关文档。

### 输入参数
- `query`: 查询字符串
- `knowledge_base_id`: 知识库标识符
- `top_k` (可选): 返回结果数量，默认 3

### 前置条件
必须先运行 `init_knowledge_base` 初始化知识库

### 操作步骤
1. 构造查询问题
2. 提供知识库 ID
3. 调用工具检索文档

### 输出解析
返回的字典包含：
- `query`: 原始查询
- `results`: 相关文档片段列表

### 示例
```
检索与 "如何解压 gzip 数据" 相关的文档
```
"""

PROMPT_COMPREHEND_LIBRARY_PURPOSE = """
## 任务：理解库目的

使用 `comprehend_library_purpose` 工具使用 LLM 理解库的整体目的。

### 输入参数
- `knowledge_base_id`: 知识库标识符

### 前置条件
必须先初始化知识库

### 操作步骤
1. 确保知识库已初始化
2. 调用工具分析库目的

### 输出解析
返回的字典包含：
- `purpose`: 库的用途描述

### 示例
```
理解 libpng 库的主要功能和用途
```
"""

PROMPT_COMPREHEND_FUNCTION_USAGE = """
## 任务：理解函数用法

使用 `comprehend_function_usage` 工具使用 LLM 理解特定函数的用法。

### 输入参数
- `function_name`: 函数名
- `knowledge_base_id`: 知识库标识符

### 前置条件
必须先初始化知识库

### 操作步骤
1. 指定要查询的函数名
2. 调用工具获取函数用法说明

### 输出解析
返回的字典包含：
- `function`: 函数名
- `usage`: 函数使用说明

### 示例
```
解释 png_create_read_struct 函数的用法
```
"""

PROMPT_COMPREHEND_ALL_FUNCTIONS = """
## 任务：批量理解函数

使用 `comprehend_all_functions` 工具批量理解 API 集合中所有函数的用法。

### 输入参数
- `api_collection`: API 函数集合
- `knowledge_base_id`: 知识库标识符

### 前置条件
- 需要先提取 API 函数
- 需要先初始化知识库

### 操作步骤
1. 确保已有 API 集合
2. 确保知识库已初始化
3. 调用工具批量处理所有函数

### 输出解析
返回的字典包含：
- 每个函数的用法说明

### 示例
```
理解 libcurl 所有 API 函数的用法
```
"""

PROMPT_COMPREHEND_FUNCTION_RELEVANCE = """
## 任务：理解函数相关性

使用 `comprehend_function_relevance` 工具计算函数之间的语义相关性。

### 输入参数
- `api_collection`: API 函数集合
- `library_purpose`: 库目的描述
- `function_usages`: 函数用法映射

### 前置条件
- 需要先提取 API 函数
- 需要先理解库目的和函数用法

### 操作步骤
1. 确保已获取库目的
2. 确保已获取函数用法
3. 调用工具计算语义相关性

### 输出解析
返回的字典包含：
- 函数对之间的语义相关性分数

### 示例
```
计算 libpng API 函数之间的语义相关性
```
"


# ============================================================================
# Complete Workflow Prompts
# ============================================================================

PROMPT_FULL_ANALYSIS_WORKFLOW = """
## 完整分析工作流

按顺序执行以下步骤来完整分析一个 C/C++ 库：

### 阶段 1：预处理
1. **AST 预处理** - 使用 `run_ast_preprocessor` 提取代码元数据
2. **API 提取** - 使用 `extract_api_functions` 识别公共 API
3. **调用图构建** - 使用 `build_library_callgraph` 建立函数调用关系
4. **类型相关性** - 使用 `calculate_type_relevance` 计算类型相关性

### 阶段 2：语义理解
1. **初始化知识库** - 使用 `init_knowledge_base` 从文档构建 RAG
2. **理解库目的** - 使用 `comprehend_library_purpose` 理解库的整体功能
3. **理解函数用法** - 使用 `comprehend_function_usage` 或 `comprehend_all_functions` 理解具体用法
4. **计算相关性** - 使用 `comprehend_function_relevance` 计算语义相关性

### 示例提示词
```
请完整分析 libjpeg 库：
1. 先进行 AST 预处理
2. 提取 API 函数
3. 构建调用图
4. 初始化知识库
5. 理解库目的
6. 理解关键函数的用法
```
"""

PROMPT_FUZZING_GUIDANCE = """
## 模糊测试引导工作流

使用 MCP 工具的输出来引导模糊测试生成：

### 目标
根据代码分析和语义理解，生成针对性的模糊测试

### 步骤
1. 使用 `run_ast_preprocessor` 分析目标库
2. 使用 `extract_api_functions` 找到暴露的 API
3. 使用 `build_callgraph` 理解 API 之间的调用关系
4. 使用 `comprehend_function_usage` 理解关键函数的使用方式
5. 基于分析结果生成模糊测试用例

### 示例提示词
```
基于对 libpng 的分析，生成模糊测试用例：
- 找到主要的 API 函数
- 理解它们的调用顺序
- 生成能触发潜在漏洞的输入
```
"""


# ============================================================================
# Helper Functions
# ============================================================================

def get_prompt_for_stage(stage: str) -> str:
    """
    Get the prompt for a specific workflow stage.
    
    Args:
        stage: The stage name (e.g., 'preprocessor', 'comprehender', 'full_analysis')
    
    Returns:
        The corresponding prompt string
    """
    prompts = {
        'preprocessor': PROMPT_AST_PREPROCESSOR,
        'api_extraction': PROMPT_EXTRACT_API_FUNCTIONS,
        'callgraph': PROMPT_BUILD_CALLGRAPH,
        'type_relevance': PROMPT_CALCULATE_TYPE_RELEVANCE,
        'function_info': PROMPT_GET_FUNCTION_INFO,
        'knowledge_base': PROMPT_INIT_KNOWLEDGE_BASE,
        'document_retrieval': PROMPT_RETRIEVE_DOCUMENTS,
        'library_purpose': PROMPT_COMPREHEND_LIBRARY_PURPOSE,
        'function_usage': PROMPT_COMPREHEND_FUNCTION_USAGE,
        'all_functions': PROMPT_COMPREHEND_ALL_FUNCTIONS,
        'function_relevance': PROMPT_COMPREHEND_FUNCTION_RELEVANCE,
        'full_analysis': PROMPT_FULL_ANALYSIS_WORKFLOW,
        'fuzzing': PROMPT_FUZZING_GUIDANCE,
    }
    return prompts.get(stage, "Unknown stage")


def get_tool_call_prompt(tool_name: str, params: dict) -> str:
    """
    Generate a prompt for calling a specific tool with parameters.
    
    Args:
        tool_name: The name of the MCP tool
        params: Dictionary of parameters for the tool
    
    Returns:
        A formatted prompt string
    """
    param_str = "\n".join([f"- `{k}`: {v}" for k, v in params.items()])
    return f"""
## 调用工具: {tool_name}

### 参数
{param_str}

### 执行
请调用 {tool_name} 工具并提供结果。
"""


def generate_analysis_prompt(library_path: str, library_docs: list[str]) -> str:
    """
    Generate a complete analysis prompt for a library.
    
    Args:
        library_path: Path to the library source code
        library_docs: List of documentation paths
    
    Returns:
        A comprehensive prompt for library analysis
    """
    docs_str = "\n".join([f"- {doc}" for doc in library_docs])
    return f"""
## 库分析任务

请完整分析以下库：

### 源代码路径
`{library_path}`

### 文档路径
{docs_str}

### 请按以下步骤执行：

1. **预处理阶段**
   - 运行 AST 预处理提取元数据
   - 提取公共 API 函数
   - 构建调用图
   - 计算类型相关性

2. **理解阶段**
   - 初始化知识库
   - 理解库的整体目的
   - 理解关键函数的用法
   - 计算函数之间的语义相关性

请开始分析并报告结果。
"""
