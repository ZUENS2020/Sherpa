---
name: "promefuzz-mcp-setup-run"
description: "Sets up and starts PromeFuzz MCP for OpenCode. Invoke when user needs first-time deployment, environment checks, or server startup troubleshooting."
---

# PromeFuzz MCP Setup and Run

## 适用场景

- 首次在新机器部署 `promefuzz-mcp`
- 需要将 PromeFuzz MCP 接入 OpenCode 的 MCP 配置
- MCP 启动失败，需要定位依赖、路径、构建问题

## 输入信息

- 项目路径（默认可用 `e:/agent_workbase/sherpa-prome/Sherpa/promefuzz-mcp`）
- 是否允许跳过二进制构建（`--skip-build`）
- 当前系统类型（Linux/macOS/Windows）

## 执行步骤

1. 检查 Python 与依赖
   - 需要 Python `>=3.10`
   - 在项目目录执行安装
   - `pip install -e .`
2. 检查构建工具
   - `cmake`, `clang`, `clang++`, `llvm-config`
3. 构建二进制
   - 推荐：`python -m promefuzz_mcp build`
   - 或进入 `processor/cxx` 使用 `./setup.sh`
4. 准备配置
   - 复制 `config.template.toml` 为 `config.toml`
   - 配置 `[llm]` 和 API Key（优先配置文件，缺省读环境变量）
5. 启动 MCP
   - 常规：`python -m promefuzz_mcp start`
   - 跳过构建检查：`python -m promefuzz_mcp start --skip-build`

## OpenCode 接入模板

将以下结构写入 OpenCode MCP 配置（路径按实际调整）：

```json
{
  "mcpServers": {
    "promefuzz": {
      "command": "python3",
      "args": [
        "-c",
        "import sys; sys.path.insert(0, '/path/to/promefuzz-mcp'); from promefuzz_mcp.server import main; main()",
        "start",
        "--skip-build"
      ],
      "env": {
        "PYTHONPATH": "/path/to/promefuzz-mcp"
      }
    }
  }
}
```

## 快速排障清单

- 缺少 `clang/AST/AST.h`：安装 `libclang` 开发包并重建
- CMake 找不到 LLVM：设置 `CMAKE_PREFIX_PATH` 与 `LLVM_DIR`
- 启动时报二进制构建失败：先用 `--skip-build` 验证 MCP 主流程
- 导入错误：补齐 `loguru`, `fastmcp`, `tomli`, `click` 等 Python 依赖

## 参考来源

- `DEPLOY.md`
- `run_mcp.sh`
