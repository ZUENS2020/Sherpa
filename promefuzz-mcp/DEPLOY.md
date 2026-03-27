# PromeFuzz MCP 部署指南

本文档详细介绍如何在全新环境中独立部署 PromeFuzz MCP 服务、配置到 OpenCode，以及如何使用。

> **注意**：删除 PromeFuzz 主目录后，Tools 目录可独立部署运行。本 MCP 服务不依赖 PromeFuzz 主框架。

## 目录

1. [环境要求](#1-环境要求)
2. [部署 MCP 服务](#2-部署-mcp-服务)
3. [配置 OpenCode](#3-配置-opencode)
4. [使用 MCP 工具](#4-使用-mcp-工具)
5. [故障排除](#5-故障排除)
6. [附录](#6-附录)

---

## 1. 环境要求

### 1.1 操作系统

| 系统 | 支持版本 | 备注 |
|------|----------|------|
| Ubuntu | 20.04 LTS / 22.04 LTS / 24.04 LTS | 推荐 Ubuntu 22.04 LTS |
| Debian | 11 (Bullseye) / 12 (Bookworm) | |
| CentOS | 8 / 9 | 需启用 EPEL 仓库 |
| RHEL | 8 / 9 | |
| macOS | 12+ (Monterey 及以上) | 需安装 Homebrew |

> **注意**：以下部署说明以 Ubuntu/Debian 为例。其他发行版请参考对应包管理器命令。

### 1.2 系统依赖

#### 1.2.1 Ubuntu / Debian

```bash
# 更新软件源
sudo apt-get update

# 安装基础构建工具
sudo apt-get install -y \
    build-essential \
    cmake \
    make \
    curl \
    wget \
    git

# 安装 Clang/LLVM 18
sudo apt-get install -y \
    clang-18 \
    clang++-18 \
    llvm-18 \
    llvm-18-dev \
    libclang-18-dev \
    libclang-common-18-dev \
    libclang-cpp18-dev \
    libllvm18 \
    libllvm18tinfo-dev \
    libz-dev \
    libtinfo-dev \
    libncurses-dev \
    zlib1g-dev

# 设置默认 Clang 版本（可选）
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-18 100
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-18 100
```

#### 1.2.2 CentOS / RHEL / Fedora

```bash
# 启用 EPEL 和 CodeReady Builder 仓库 (RHEL/CentOS 8+)
sudo dnf install -y epel-release
sudo dnf config-manager --set-enabled powertools  # CentOS 8
sudo dnf config-manager --set-enabled codeready-builder-for-rhel-9-rhui-rpms  # RHEL 9

# 安装基础工具
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y \
    cmake \
    make \
    clang \
    clang-devel \
    llvm \
    llvm-devel \
    llvm-static \
    zlib-devel \
    libzstd-devel \
    ncurses-devel \
    libffi-devel

# 如果仓库中无 LLVM 18，可使用 LLVM 官方仓库
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18
```

#### 1.2.3 macOS

```bash
# 使用 Homebrew 安装
brew install \
    cmake \
    llvm \
    clang-format \
    autoconf \
    automake \
    libtool

# 设置环境变量
export LLVM_PREFIX=$(brew --prefix llvm)
export PATH="$LLVM_PREFIX/bin:$PATH"
```

### 1.3 Python 环境

#### 1.3.1 系统 Python（推荐）

```bash
# 检查 Python 版本
python3 --version  # 需要 >= 3.10

# 安装 pip
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3

# 安装 pyenv（可选，推荐用于管理多个 Python 版本）
curl -sL https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# 安装 Python 3.12
pyenv install 3.12.0
pyenv global 3.12.0

# 验证
python --version  # 应显示 3.12.x
```

#### 1.3.2 虚拟环境（可选）

```bash
# 创建虚拟环境
python3 -m venv ~/.venv/promefuzz-mcp

# 激活虚拟环境
source ~/.venv/promefuzz-mcp/bin/activate

# 激活后提示符会变为
# (promefuzz-mcp) user@host:~$
```

### 1.4 环境验证

构建前，验证所有依赖已正确安装：

```bash
# 1. 检查 Python 版本
python3 --version
# 预期输出: Python 3.10.x / 3.11.x / 3.12.x

# 2. 检查 CMake
cmake --version
# 预期输出: cmake version 3.15+

# 3. 检查 Clang 编译器
clang --version
# 预期输出: clang version 18.x

# 4. 检查 LLVM 配置
llvm-config --version
# 预期输出: 18.x

# 5. 检查 libclang
ls -la /usr/lib/llvm-18/lib/libclang-cpp.so*
# 预期输出: libclang-cpp.so.18.x 相关文件

# 6. 检查 clang 头文件
ls -la /usr/lib/llvm-18/include/clang/AST/AST.h
# 预期输出: AST.h 文件存在
```

### 1.5 Python 依赖

```bash
cd Tools/promefuzz-mcp

# 方式一：安装完整依赖（推荐）
pip3 install --break-system-packages -e .

# 方式二：使用虚拟环境（无 sudo 权限时推荐）
source ~/.venv/promefuzz-mcp/bin/activate
pip install -e .

# 方式三：仅安装核心依赖（离线或最小安装）
pip3 install --break-system-packages \
    loguru \
    tomli \
    click \
    fastmcp \
    tqdm \
    chromadb \
    sentence-transformers \
    openai \
    ollama \
    tiktoken \
    pydantic
```

---

## 2. 部署 MCP 服务

### 2.1 完整部署流程（全新环境）

```bash
# ============================================
# 步骤 1: 克隆或解压项目
# ============================================
# 如果从压缩包解压
unzip Tools.zip -d /path/to/workspace
cd /path/to/workspace/Tools/promefuzz-mcp

# ============================================
# 步骤 2: 安装系统依赖（需要 sudo）
# ============================================
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake clang-18 llvm-18 libclang-18-dev \
    libllvm18tinfo-dev libz-dev libtinfo-dev

# ============================================
# 步骤 3: 安装 Python 依赖
# ============================================
pip3 install --break-system-packages -e .

# ============================================
# 步骤 4: 构建处理器二进制
# ============================================
cd processor/cxx
./setup.sh

# 验证构建产物
ls -la ../build/bin/
# 应看到 preprocessor 和 cgprocessor

# ============================================
# 步骤 5: 配置 config.toml
# ============================================
cd ../..
cp config.template.toml config.toml

# 编辑 config.toml（至少配置 LLM API key）
vim config.toml

# ============================================
# 步骤 6: 测试启动
# ============================================
python3 -m promefuzz_mcp start

# 看到 FastMCP 启动画面即成功
# 按 Ctrl+C 停止

# ============================================
# 步骤 7: 后台运行（可选）
# ============================================
nohup python3 -c "from promefuzz_mcp.server import main; main()" start --skip-build > mcp.log 2>&1 &

# 验证运行
ps aux | grep promefuzz
cat mcp.log
```

### 2.2 安装步骤概览

```
全新环境部署流程：

1. 安装系统依赖（cmake, clang-18, llvm-18, libclang-dev）
2. 克隆或解压项目到本地
3. 安装 Python 依赖（pip install -e .）
4. 构建处理器二进制（./setup.sh 或手动 cmake/make）
5. 配置 config.toml（LLM API key 等）
6. 启动服务
7. 配置 OpenCode（可选）
```

### 2.3 构建处理器二进制

处理器二进制是 Clang AST 分析工具，用于从 C/C++ 代码中提取元数据。

#### 方式一：使用自动化脚本（推荐）

```bash
cd Tools/promefuzz-mcp/processor/cxx
./setup.sh
```

**环境变量说明**：
- `CLANG_INSTALL_DIR`：指定 LLVM/Clang 安装路径，默认为 `llvm-config --prefix` 检测结果

**示例**：
```bash
# 使用系统默认 LLVM
./setup.sh

# 指定自定义 LLVM 路径
CLANG_INSTALL_DIR=/opt/llvm-18 ./setup.sh
```

#### 方式二：手动构建

```bash
cd Tools/promefuzz-mcp/processor/cxx

# 清理并创建 build 目录
rm -rf build
mkdir build
cd build

# 检测 LLVM 安装路径
LLVM_PREFIX=$(llvm-config --prefix)
echo "Using LLVM: $LLVM_PREFIX"

# CMake 配置
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$LLVM_PREFIX" \
    -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"

# 编译
make -j$(nproc)

# 创建 bin 目录
mkdir -p ../build/bin
ln -sf ../build/preprocessor ../build/bin/
ln -sf ../build/cgprocessor ../build/bin/
```

#### 方式三：通过 Python 模块构建

```bash
cd Tools/promefuzz-mcp
python3 -m promefuzz_mcp build
```

### 2.4 配置 config.toml

```bash
cd Tools/promefuzz-mcp
cp config.template.toml config.toml
```

编辑 `config.toml`，主要配置项：

```toml
[bin]
preprocessor = "processor/build/bin/preprocessor"
cgprocessor = "processor/build/bin/cgprocessor"

[llm]
default_llm = "cloud_llm"

[llm.cloud_llm]
llm_type = "openai"
base_url = "https://api.openai.com/v1/"
api_key = ""  # Use environment variable OPENAI_API_KEY if empty
model = "gpt-4o"
temperature = 0.5
timeout = 80
```

#### 配置 LLM API Key（环境变量方式）

API Key 通过环境变量配置，代码会优先读取配置文件中的值，若为空则使用环境变量：

| LLM 类型 | 环境变量 |
|----------|----------|
| Minimax | `MINIMAX_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Ollama | 无需 API Key |

**示例**：

```bash
# 设置 Minimax API Key
export MINIMAX_API_KEY="your-minimax-api-key"

# 设置 OpenAI API Key（如果使用 OpenAI 作为默认 LLM）
export OPENAI_API_KEY="your-openai-api-key"
```

### 2.5 启动服务

#### 方式一：正常启动（需要二进制）

```bash
cd Tools/promefuzz-mcp
PYTHONPATH=. python3 -m promefuzz_mcp start
```

#### 方式二：跳过二进制检查启动

如果无法编译二进制工具，可跳过检查启动服务：

```bash
cd Tools/promefuzz-mcp
PYTHONPATH=. python3 -c "from promefuzz_mcp.server import main; main()" start --skip-build
```

#### 方式三：后台运行

```bash
cd Tools/promefuzz-mcp
PYTHONPATH=. nohup python3 -c "from promefuzz_mcp.server import main; main()" start --skip-build > /tmp/mcp_server.log 2>&1 &
```

### 2.6 验证服务启动

```bash
# 检查日志
cat /tmp/mcp_server.log

# 或检查进程
ps aux | grep promefuzz_mcp
```

**成功输出示例**：
```
2026-03-20 11:00:00 | INFO | Starting MCP server on localhost:8000
╭──────────────────────────────────────────────────────────────────────────────╮
│                              FastMCP 3.1.1                                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 3. 配置 OpenCode

### 3.1 MCP 配置文件

在项目根目录创建 `.opencode` 目录和配置文件：

```bash
mkdir -p .opencode
```

创建 `opencode.jsonc` 文件：

```jsonc
{
  "mcpServers": {
    "promefuzz": {
      "command": "python3",
      "args": [
        "-c",
        "import sys; sys.path.insert(0, '/path/to/Tools/promefuzz-mcp'); from promefuzz_mcp.server import main; main()",
        "start",
        "--skip-build"
      ],
      "env": {
        "PYTHONPATH": "/path/to/Tools/promefuzz-mcp"
      }
    }
  }
}
```

> **注意**：请根据实际路径修改 `sys.path.insert` 和 `PYTHONPATH` 中的路径。

### 3.2 验证配置

```bash
opencode mcp list
```

应该显示已配置的 MCP 服务器。

---

## 4. 使用 MCP 工具

### 4.1 可用工具列表

#### Preprocessor 模块（10个工具）

| 工具名称 | 功能描述 |
|---------|---------|
| `run_ast_preprocessor` | 使用 Clang AST 解析器处理 C/C++ 源代码，提取元数据 |
| `extract_api_functions` | 从头文件中识别并提取公共 API 函数 |
| `build_library_callgraph` | 构建库源代码的函数调用图 |
| `build_consumer_callgraph` | 构建库消费者代码的调用图 |
| `extract_incidentals` | 提取函数之间的附带关系 |
| `calculate_type_relevance` | 基于函数参数/返回值的类型计算 API 函数之间的相关性 |
| `calculate_class_relevance` | 基于类成员关系计算函数之间的相关性 |
| `calculate_call_relevance` | 基于调用关系计算函数之间的相关性 |
| `calculate_complexity` | 计算 API 函数的复杂度 |
| `get_function_info` | 获取特定函数的详细信息 |

#### Comprehender 模块（6个工具）

| 工具名称 | 功能描述 |
|---------|---------|
| `init_knowledge_base` | 从文档构建 RAG 知识库 |
| `retrieve_documents` | 使用 RAG 从知识库中检索相关文档 |
| `comprehend_library_purpose` | 使用 LLM 理解库的整体目的和功能 |
| `comprehend_function_usage` | 使用 LLM 理解特定函数的用法 |
| `comprehend_all_functions` | 批量理解 API 集合中所有函数的用法 |
| `comprehend_function_relevance` | 基于语义计算函数之间的相关性 |

### 4.2 完整工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        Preprocessor 阶段                        │
├─────────────────────────────────────────────────────────────────┤
│  源代码 ──▶ run_ast_preprocessor ──▶ meta.json                │
│                              │                                   │
│                              ▼                                   │
│  头文件 ──▶ extract_api_functions ──▶ APICollection          │
│                              │                                   │
│                              ▼                                   │
│  源代码 ──▶ build_library_callgraph ──▶ CallGraph              │
│                              │                                   │
│                              ▼                                   │
│  APICollection + Meta ──▶ calculate_*( relevance) ──▶ Scores   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Comprehender 阶段                         │
├─────────────────────────────────────────────────────────────────┤
│  文档 ──▶ init_knowledge_base ──▶ RAG 知识库                   │
│                              │                                   │
│                              ▼                                   │
│  知识库 ──▶ retrieve_documents ──▶ 相关文档                     │
│                              │                                   │
│                              ▼                                   │
│  知识库 ──▶ comprehend_library_purpose ──▶ 库目的               │
│                              │                                   │
│                              ▼                                   │
│  知识库 ──▶ comprehend_function_usage ──▶ 函数用法              │
│                              │                                   │
│                              ▼                                   │
│  API + 库目的 ──▶ comprehend_function_relevance ──▶ 相关性       │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 使用示例

#### 示例 1：AST 预处理

```
使用 run_ast_preprocessor 工具处理 /path/to/source 目录
输入: source_paths=["/path/to/source"]
输出: 元数据字典（包含类和函数数量）
```

#### 示例 2：提取 API 函数

```
使用 extract_api_functions 工具从头文件提取 API
输入: header_paths=["/path/to/headers"], meta_path="meta.json"
输出: API 函数集合
```

#### 示例 3：初始化知识库

```
使用 init_knowledge_base 工具构建 RAG 知识库
输入: document_paths=["/path/to/docs"], output_path="knowledge_db"
输出: 知识库信息
```

---

## 5. 故障排除

### 5.1 二进制编译失败

**错误信息**:
```
fatal error: 'clang/AST/AST.h' file not found
```

**解决方案**:
```bash
# 确认 libclang-dev 已安装
sudo apt-get install --reinstall libclang-18-dev

# 检查头文件是否存在
ls /usr/lib/llvm-18/include/clang/AST/AST.h

# 如果不存在，尝试重新安装整个 LLVM 18
sudo apt-get install --reinstall llvm-18 llvm-18-dev clang-18 libclang-18-dev

# 重新构建
cd Tools/promefuzz-mcp/processor/cxx
rm -rf build
./setup.sh
```

### 5.2 LLVM 版本不匹配

**错误信息**:
```
LLVM version mismatch: found X.X, expected 18.x
```

**解决方案**:
```bash
# 检查当前 LLVM 版本
llvm-config --version
clang --version

# 确认使用的是 LLVM 18
which llvm-config
which clang

# 如果不是 18 版本，设置正确的 PATH
export PATH="/usr/lib/llvm-18/bin:$PATH"
llvm-config --version  # 确认现在显示 18.x

# 指定正确的 CLANG_INSTALL_DIR 重新构建
CLANG_INSTALL_DIR=/usr/lib/llvm-18 ./setup.sh
```

### 5.3 CMake 找不到 LLVM

**错误信息**:
```
CMake Error at CMakeLists.txt:10 (project):
  No CMake module for LLVM 18 found
```

**解决方案**:
```bash
# 安装完整的 LLVM CMake 模块
sudo apt-get install llvm-18-dev

# 检查 CMake 模块路径
ls /usr/lib/llvm-18/lib/cmake/llvm/

# 如果仍然找不到，手动指定路径
cd processor/cxx
rm -rf build && mkdir build && cd build
cmake .. \
    -DCMAKE_PREFIX_PATH=/usr/lib/llvm-18 \
    -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm
make
```

### 5.4 模块导入错误

**错误信息**:
```
ModuleNotFoundError: No module named 'loguru'
```

**解决方案**:
```bash
pip3 install --break-system-packages loguru tomli click fastmcp
```

### 5.5 服务启动失败

**错误信息**:
```
ERROR | Failed to build processor binaries
```

**解决方案**:
使用 `--skip-build` 选项跳过二进制检查：
```bash
python3 -c "from promefuzz_mcp.server import main; main()" start --skip-build
```

### 5.6 OpenCode 无法识别 MCP

**检查步骤**:
1. 确认 MCP 服务正在运行：`ps aux | grep promefuzz`
2. 确认配置文件路径正确：检查 `.opencode/opencode.jsonc`
3. 确认 PYTHONPATH 设置正确
4. 尝试重启 OpenCode

### 5.7 macOS 上编译失败

**错误信息**:
```
ld: library 'clang' not found
```

**解决方案**:
```bash
# 使用 Homebrew 安装完整 LLVM
brew install llvm

# 设置环境变量
export LLVM_PREFIX=$(brew --prefix llvm)
export CMAKE_PREFIX_PATH="$LLVM_PREFIX"
export LLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"

# 重新构建
cd processor/cxx
rm -rf build
./setup.sh
```

---

## 6. 附录

### A. 处理器二进制源码结构

```
Tools/promefuzz-mcp/
└── processor/
    ├── cxx/                      # C++ 源码
    │   ├── CMakeLists.txt        # CMake 构建配置
    │   ├── setup.sh              # 自动化构建脚本
    │   ├── preprocessor.cc      # AST 预处理实现
    │   ├── preprocessor.hh      # 预处理头文件
    │   ├── cgprocessor.cc        # 调用图处理实现
    │   ├── cgprocessor.hh        # 调用图头文件
    │   ├── processor.hh          # 共享类型定义
    │   ├── clang-common.hh       # Clang 公共头文件
    │   ├── json.hpp              # JSON 库
    │   └── example/              # 示例代码
    └── build/
        ├── bin/                  # 编译输出
        │   ├── preprocessor      # AST 预处理工具
        │   └── cgprocessor       # 调用图处理工具
        └── ...                    # CMake 构建文件
```

### B. 完整配置模板

```toml
# PromeFuzz MCP Configuration

[preprocessor]
run_dummydriver_test = false
dump_relevance_as_csv = false
dump_call_graph = false

[comprehender]
embedding_llm = "embedding_llm"
comprehension_llm = ""
retrieve_top_k = 3
function_batch_size = 24

[bin]
preprocessor = "processor/build/bin/preprocessor"
cgprocessor = "processor/build/bin/cgprocessor"

[llm]
default_llm = "cloud_llm"
validate_llm = false
enable_log = true

[llm.cloud_llm]
llm_type = "openai"
base_url = "https://api.openai.com/v1/"
api_key = ""  # Use environment variable OPENAI_API_KEY if empty
model = "gpt-4o"
temperature = 0.5
timeout = 80
```

### C. 相关文件

| 文件路径 | 说明 |
|----------|------|
| `promefuzz_mcp/server.py` | MCP 服务器入口 |
| `promefuzz_mcp/server_tools.py` | 工具定义 |
| `promefuzz_mcp/config.py` | 配置管理 |
| `promefuzz_mcp/build.py` | Python 二进制构建模块 |
| `promefuzz_mcp/preprocessor/` | 预处理工具模块 |
| `promefuzz_mcp/comprehender/` | 理解工具模块 |
| `processor/cxx/setup.sh` | 二进制自动化构建脚本 |
| `config.toml` | 运行时配置 |

### D. 快速参考

```bash
# 完整部署流程
cd Tools/promefuzz-mcp
pip install -e .              # 安装 Python 依赖
cd processor/cxx && ./setup.sh  # 构建二进制
cd ../..
cp config.template.toml config.toml
vim config.toml               # 配置 LLM API key
python -m promefuzz_mcp start # 启动服务

# 跳过二进制检查启动
python -c "from promefuzz_mcp.server import main; main()" start --skip-build

# 后台运行
nohup python -c "from promefuzz_mcp.server import main; main()" start --skip-build > mcp.log 2>&1 &
```

### E. 环境检查清单

部署前，使用以下清单验证环境：

```
□ Python >= 3.10
□ pip 可用
□ cmake >= 3.15
□ clang >= 18
□ clang++ >= 18
□ llvm-config >= 18
□ libclang-dev (包含 clang/AST/AST.h)
□ /usr/lib/llvm-18/lib/cmake/llvm/ 存在
□ zlib-dev / zlib1g-dev
□ libtinfo-dev / libtinfo
```

---

**创建日期**: 2026-03-17
**最后更新**: 2026-03-20
**版本**: 1.2.0