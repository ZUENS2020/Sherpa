# Sherpa 系统改进计划（5 天）

> **状态: 🟡 主要项已收口，剩余回归阻塞待清理** — 2026-04-02
>
> | Day | 任务 | 状态 |
> |-----|------|------|
> | 1 | 依赖锁定 + Django 清理 | ✅ |
> | 2 | CI 测试 workflow + 自动部署 | ✅ |
> | 3 | 统一日志 (loguru) — 71 print→logger | ✅ |
> | 4 | Provider 配置重构 + secrets 解耦 | ✅ |
> | 5 | 死代码清理 + OSS-Fuzz 基础设施移除 | 🟡 |

## 背景

Sherpa 是基于 LangGraph 的模糊测试编排平台，自动化完成目标选择 → 用例生成 → 构建 → 模糊测试 → 崩溃分流的全流程。

经过对整个代码库的全面探索，识别出 15 项改进方向。本文档聚焦 **5 天内可落地、影响最大** 的项目，按天拆分为可执行任务。

### 核心问题摘要

| 类别 | 发现 | 严重性 |
|------|------|--------|
| 依赖管理 | `requirements.web.txt` 使用 `>=` 宽松约束，LangChain/LangGraph 频繁破坏性升级 | P0 |
| CI/CD | 28 个后端测试文件 + 2 个前端测试文件存在但 CI 从未执行 | P0 |
| 异常处理 | 全代码库 411 个 `except Exception:` 块，静默吞掉真实 bug | P0 |
| 日志 | stdlib logging + loguru + print() 三种方式混用 | P1 |
| 代码结构 | `workflow_graph.py` 11,462 行、`main.py` 5,003 行、`FuzzWorkflowState` 195 字段 | P1 |
| 部署 | deploy-dev/prod 均为手动触发 | P2 |
| 健康检查 | 无 K8s liveness/readiness probe | P3 |
| OpenCode 配置 | Provider 白名单硬编码，添加新 provider 必须改代码；环境变量回退链混乱 | P1 |

---

## Day 1: 依赖治理 + 冗余清理

### 任务 1.1: 锁定 `docker/requirements.web.txt` 版本

**问题**: `fastapi>=0.110`, `langchain>=0.2`, `langgraph>=0.2`, `openai>=1.0` 等全部使用 `>=` 下限约束。LangChain 和 LangGraph 在小版本间经常引入破坏性 API 变更，导致今天构建的镜像和上周的行为可能完全不同。

**操作**:
- 将所有 `>=` 替换为 `==` 精确版本（基于当前生产环境可用的稳定版本）
- 重点关注: `langchain`, `langgraph`, `openai`, `fastapi`, `tree_sitter`

**文件**: `docker/requirements.web.txt`

### 任务 1.2: 清理冗余依赖文件

**问题**: `harness_generator/src/requirements.txt` 包含 `Django==5.2.7` 和 `djangorestframework==3.16.1`，但项目使用的是 FastAPI，Django 从未被引用。两个 requirements 文件存在不一致。

**操作**:
- 移除未使用的 Django 相关依赖
- 与 `docker/requirements.web.txt` 对齐版本

**文件**: `harness_generator/src/requirements.txt`

**验证**:
```bash
grep -r "django" harness_generator/src/  # 确认无 Django 引用
pip install -r docker/requirements.web.txt  # 在干净环境中验证
```

---

## Day 2: CI 测试 + 自动化部署

### 任务 2.1: 创建 CI 测试 workflow

**问题**: 项目有 28 个后端测试文件（10,700+ 行测试代码）和 2 个前端测试文件，但 GitHub Actions 中没有任何 workflow 执行这些测试。现有 CI 仅有 `pr-review.yml`（生成 PR 摘要）和手动触发的部署。

**操作**: 创建 `.github/workflows/test.yml`

**配置**:
- 触发条件: `pull_request` + `push` 到 `dev`/`main`
- 后端步骤: `pip install -r docker/requirements.web.txt` + `pytest tests/ --tb=short -q`
- 前端步骤: `cd frontend-next && npm ci && npx vitest run`

**参考**: 现有 `.github/workflows/pr-review.yml` 的 runner 和环境配置

### 任务 2.2: 重构 OpenCode Provider 配置为可扩展架构

**问题**: 当前 `persistent_config.py` 的 OpenCode provider 管理存在严重的扩展性障碍：

1. **硬编码白名单** — `_normalize_provider_entry()` 第 405 行 `if name not in {_MINIMAX_PROVIDER, _DEEPSEEK_PROVIDER}: return None`，任何新 provider（如 OpenRouter、Anthropic、智谱）都会被直接丢弃
2. **环境变量回退链混乱** — `_resolve_minimax_env_values()` 和 `apply_llm_env_source()` 中 `LLM_key` → `DEEPSEEK_API_KEY` → `OPENAI_API_KEY` → `MINIMAX_API_KEY` 四层 fallback，无法确定实际生效的是哪个
3. **配置断裂** — `opencode.json` 已配置 openrouter provider，但 `persistent_config.py` 的白名单不包含它，运行时会被过滤掉
4. **遗留命名** — `apply_minimax_env_source()` 只是 `apply_llm_env_source()` 的别名，容易误导
5. **NPM 包映射硬编码** — `_OPENCODE_PROVIDER_NPM` 仅映射了 minimax 和 deepseek

**操作**:

1. **移除 provider 白名单限制**:
   - 修改 `_normalize_provider_entry()` — 移除 `if name not in {...}: return None` 硬编码检查
   - 改为：只要 provider 有 `name` 和 `base_url`（或可从已知 provider 推断），即视为有效
   - 对已知 provider（deepseek, minimax）保留默认 base_url 回退；未知 provider 要求显式提供 base_url

2. **将 provider 注册改为数据驱动**:
   ```python
   # 替代多个硬编码 dict，使用统一的 provider registry
   @dataclass
   class ProviderDefaults:
       base_url: str
       npm_package: str
       default_models: list[str]
       env_key_names: list[str]  # 用于从环境变量读取 API key

   KNOWN_PROVIDERS: dict[str, ProviderDefaults] = {
       "deepseek": ProviderDefaults(
           base_url="https://api.deepseek.com/v1",
           npm_package="@ai-sdk/openai-compatible",
           default_models=["deepseek-reasoner", "deepseek-chat"],
           env_key_names=["DEEPSEEK_API_KEY"],
       ),
       "minimax": ProviderDefaults(
           base_url="https://api.minimaxi.com/anthropic/v1",
           npm_package="@ai-sdk/anthropic",
           default_models=["MiniMax-M2.7-highspeed"],
           env_key_names=["MINIMAX_API_KEY"],
       ),
       "openrouter": ProviderDefaults(
           base_url="https://openrouter.ai/api/v1",
           npm_package="@ai-sdk/openai-compatible",
           default_models=[],
           env_key_names=["OPENROUTER_API_KEY"],
       ),
       # 新 provider 只需在这里添加一行，无需改任何逻辑代码
   }
   ```

3. **简化环境变量解析**:
   - 移除 `_resolve_minimax_env_values()` 四层 fallback
   - 每个 provider 只读自己的 env key（通过 `env_key_names`）
   - 全局 fallback 仅保留 `LLM_key` 作为通用后备
   - 优先级清晰：`provider.api_key`（DB 配置） > `provider.env_key_names` > `LLM_key`

4. **清理遗留代码**:
   - 删除 `apply_minimax_env_source()` 别名，调用方直接用 `apply_llm_env_source()`
   - 删除 `_resolve_minimax_env_values()`，逻辑合并到新的 registry 查询中
   - 合并 `_OPENCODE_PROVIDER_ALIASES`、`_OPENCODE_PROVIDER_NPM`、`_OPENCODE_PROVIDER_MODEL_CANDIDATES` 三个 dict 到 `KNOWN_PROVIDERS`

**文件**: `harness_generator/src/langchain_agent/persistent_config.py`

**验证**:
```bash
# 确认现有 provider 配置不受影响
pytest tests/test_persistent_config_runtime_paths.py -v
# 确认新 provider 可以通过（之前会被丢弃）
python -c "
from harness_generator.src.langchain_agent.persistent_config import OpencodeProviderConfig, normalize_opencode_providers
p = OpencodeProviderConfig(name='openrouter', base_url='https://openrouter.ai/api/v1', models=['qwen/qwen3-max'])
result = normalize_opencode_providers([p])
assert len(result) == 1, 'openrouter provider should not be filtered out'
print('OK: new providers accepted')
"
```

### 任务 2.3: Secrets 与环境变量 Provider 解耦

**问题**: 当前 secrets 和环境变量的命名都绑定了具体 provider 名称，换 provider 时需要改动大量位置：

1. **K8s Secret 名称绑定 provider** — secret 资源叫 `sherpa-deepseek`，包含 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 等。换 provider 需要重建 secret、改引用
2. **GitHub Actions secret 绑定 provider** — `DEEPSEEK_API_KEY_DEV`、`DEEPSEEK_API_KEY_PROD`、`DEEPSEEK_MODEL_DEV` 等，provider 名直接写在 secret 名里
3. **Secret 名称三层 fallback** — `main.py:803-810` 中 `SHERPA_K8S_LLM_SECRET_NAME` → `SHERPA_K8S_DEEPSEEK_SECRET_NAME` → `SHERPA_K8S_MINIMAX_SECRET_NAME`，遗留了两代 provider 的历史包袱
4. **`.env.example` 过时** — 注释写 "AI provider credentials (MiniMax only)"，但默认已是 DeepSeek
5. **docker-compose 代理地址硬编码** — `HTTP_PROXY: http://host.docker.internal:7897` 作为默认值，是本地开发环境特定配置

**操作**:

1. **K8s Secret 重命名为 provider 无关名称**:
   - `sherpa-deepseek` → `sherpa-llm`
   - Secret 内 key 统一为通用名: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`
   - 更新 `k8s/base/deepseek-secret.example.yaml` → `k8s/base/llm-secret.example.yaml`
   - 更新所有引用: `web-deployment.yaml`、`main.py` 中的 `secretRef`

2. **GitHub Actions secrets 统一命名**:
   - `DEEPSEEK_API_KEY_DEV` → `LLM_API_KEY_DEV`
   - `DEEPSEEK_API_KEY_PROD` → `LLM_API_KEY_PROD`
   - `DEEPSEEK_MODEL_DEV` → `LLM_MODEL_DEV`
   - 更新 `deploy-dev.yml` 和 `deploy-prod.yml` 中的 secrets 引用

3. **简化 main.py 中的 secret 名称解析**:
   ```python
   # 之前: 三层 fallback
   llm_secret = (
       os.environ.get("SHERPA_K8S_LLM_SECRET_NAME", "").strip()
       or os.environ.get("SHERPA_K8S_DEEPSEEK_SECRET_NAME", "").strip()
       or os.environ.get("SHERPA_K8S_MINIMAX_SECRET_NAME", "sherpa-deepseek").strip()
   )
   # 之后: 单一来源
   llm_secret = os.environ.get("SHERPA_K8S_LLM_SECRET_NAME", "sherpa-llm").strip()
   ```

4. **更新 `.env.example`**:
   - 替换过时的 "MiniMax only" 注释
   - 使用通用 LLM 变量名，附注说明如何为不同 provider 配置
   - 移除代理地址的硬编码默认值，改为空（由用户按需填写）

5. **更新 `codex_helper.py` 的 Docker env allowlist**:
   - `_docker_opencode_env_args()` 中的 `allowed_keys` 添加 `LLM_API_KEY`、`LLM_BASE_URL`
   - 保留 `DEEPSEEK_API_KEY`、`OPENAI_API_KEY` 等作为向后兼容（但标注 deprecated）

**涉及文件**:
- `k8s/base/deepseek-secret.example.yaml` → `k8s/base/llm-secret.example.yaml`（重命名）
- `k8s/base/web-deployment.yaml`（更新 secretRef）
- `k8s/base/configmap.yaml`（更新 secret 引用变量名）
- `.github/workflows/deploy-dev.yml`（更新 secrets 引用）
- `.github/workflows/deploy-prod.yml`（更新 secrets 引用）
- `harness_generator/src/langchain_agent/main.py`（简化 secret 名称解析）
- `harness_generator/src/codex_helper.py`（更新 Docker env allowlist）
- `.env.example`（更新注释和变量名）
- `docker-compose.yml`（更新环境变量名和默认值）

**验证**:
```bash
# 确认所有旧 secret 名称引用已替换
grep -r "sherpa-deepseek" k8s/ .github/ harness_generator/  # 目标: 0 结果
grep -r "DEEPSEEK_API_KEY_DEV\|DEEPSEEK_API_KEY_PROD" .github/  # 目标: 0 结果
# 确认新 secret 名称一致
grep -r "sherpa-llm" k8s/ harness_generator/  # 应有引用
```

**注意**: 这一步需要同步更新实际部署环境中的 K8s secret 和 GitHub repo secrets。建议提供迁移脚本或在 PR 描述中列出手动操作步骤。

---

## Day 3: 统一日志框架

### 任务 3.1: 替换 main.py 中的 print() 为 loguru

**问题**: `main.py` 中大量使用 `print(f"[job {job_id}]...")` 模式输出日志。`harness_generator/` 使用 stdlib `logging`，`promefuzz-mcp/` 使用 `loguru`。三种日志方式混用导致无法统一格式化和路由。

**操作**:
1. 添加 `from loguru import logger`
2. 将 `print(f"[job {job_id}]...")` 替换为 `logger.info("...", job_id=job_id, stage=stage)`
3. 添加 loguru interceptor 捕获 stdlib logging 输出:
   ```python
   import logging
   class InterceptHandler(logging.Handler):
       def emit(self, record):
           logger.opt(depth=6, exception=record.exc_info).log(record.levelno, record.getMessage())
   logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
   ```

**文件**: `harness_generator/src/langchain_agent/main.py`

### 任务 3.2: 替换其他后端文件的 logging

**文件**:
- `harness_generator/src/fuzz_unharnessed_repo.py`
- `harness_generator/src/codex_helper.py`

**操作**: `import logging` → `from loguru import logger`，替换 `logging.info/warning/error` 为 `logger.info/warning/error`

### 任务 3.3: 确保依赖就位

**操作**: 在 `docker/requirements.web.txt` 中添加 `loguru==<version>`（如尚未包含）

**验证**:
```bash
grep -c "print(" harness_generator/src/langchain_agent/main.py  # 目标: 接近 0
pytest tests/ --tb=short -q  # 确认无破坏
```

---

## Day 4: 收窄异常捕获（main.py）

### 任务 4.1: 建立 SherpaError 异常层次

**问题**: 全代码库 411 个 `except Exception:` 块（main.py 87 个，workflow_graph.py 195 个，fuzz_unharnessed_repo.py 129 个）。这种过度宽泛的捕获已导致实际问题被掩盖——例如 `fix_plan.md` 中记录的 k8s timeout bug 就是因为 timeout 异常被通用 `except Exception` 吞掉。

**操作**: 创建异常定义文件

**新建文件**: `harness_generator/src/langchain_agent/errors.py`

```python
class SherpaError(RuntimeError):
    """Base exception for all Sherpa-specific errors."""

class BuildError(SherpaError):
    """Harness compilation or build script failure."""

class RunError(SherpaError):
    """Fuzzer execution failure."""

class TriageError(SherpaError):
    """Crash triage or analysis failure."""

class ConfigError(SherpaError):
    """Configuration validation or loading failure."""

class K8sJobError(SherpaError):
    """Kubernetes job submission, execution, or timeout failure."""
```

### 任务 4.2: 收窄 main.py 中最关键的 except Exception

**范围**: 聚焦 main.py 中最关键的 30-40 个 `except Exception`，按优先级：

1. **K8s job 路径**（`_k8s_submit_job`, `_k8s_wait_job`）→ 替换为 `K8sJobError`, `TimeoutError`, `OSError`
2. **API endpoint handlers** → 替换为 `ValueError`, `KeyError`, `FileNotFoundError`
3. **配置加载路径** → 替换为 `ConfigError`, `json.JSONDecodeError`

**原则**:
- 不确定具体异常类型的，先加 `logger.exception("Unexpected error in ...")` 保留完整上下文
- 不盲目替换——确认异常来源后再收窄
- 本轮不处理 `workflow_graph.py` 和 `fuzz_unharnessed_repo.py`（留待后续迭代）

**文件**: `harness_generator/src/langchain_agent/main.py`

**验证**:
```bash
grep -c "except Exception" harness_generator/src/langchain_agent/main.py  # 目标: 从 87 降至 < 50
pytest tests/test_api_stability.py tests/test_k8s_*.py -v  # 关键路径测试
```

---

## Day 5: 全面清理 — 死代码、孤立文件、过时文档、环境变量

> Day 5 专注清理。前 4 天已完成结构性改进，Day 5 把所有遗留垃圾清理干净。

### 任务 5.1: 删除死函数、遗留别名和废弃模块

**A. 死函数和遗留别名**:

| 位置 | 行号 | 内容 | 类型 |
|------|------|------|------|
| `persistent_config.py` | 588-608 | `_resolve_minimax_env_values()` — 定义但从未调用，逻辑已在 `apply_llm_env_source()` 中重复 | 死函数 |
| `persistent_config.py` | 661-663 | `apply_minimax_env_source()` — 仅是 `apply_llm_env_source()` 的别名 | 遗留别名 |
| `codex_helper.py` | 1822-1826 | `CodexPatcher = CodexHelper` — 旧名称别名 | 遗留别名 |
| `workflow_graph.py` | 10807-10809 | `_node_repro_crash()` — 仅调用 `_node_re_build()`，旧步骤名称兼容 | 遗留别名 |
| `fuzz_relative_functions.py` | 74-78 | `coverage_loop_max_rounds`, `max_fix_rounds`, `same_error_max_retries` — 保留仅为向后兼容，从未使用 | 死参数 |
| `main.py` | 33, 3242 | `import apply_minimax_env_source` — 引用已废弃别名 | 过时 import |

**操作**:
1. 删除 `_resolve_minimax_env_values()`
2. 删除 `apply_minimax_env_source()` 别名，全局搜索替换为 `apply_llm_env_source`
3. 删除 `CodexPatcher = CodexHelper`，全局搜索 `CodexPatcher` 替换为 `CodexHelper`
4. 删除 `_node_repro_crash()` 别名，将图中 `repro_crash` 节点直接指向 `_node_re_build`
5. 移除 `fuzz_relative_functions.py` 中未使用的参数

**B. 废弃模块 — 整文件删除**:

| 文件 | 行数 | 原因 |
|------|------|------|
| `langchain_agent/ossfuzz_auto.py` | 605 | **完全孤立**：从未被任何文件 import。定义了 `OssFuzzAutoError` 和 OSS-Fuzz 集成工具函数，但整个模块无人调用。相关 env var（`SHERPA_OSS_FUZZ_DIR`、`SHERPA_OSS_FUZZ_REPO_URL`）也因此成为死配置 |
| `langchain_agent/main_brain.py` | ~130 | **LangChain demo 残留**：包含 `get_weather_for_location()` demo tool 和 `create_agent_outside()` 函数，从未被调用。项目已全面迁移到 LangGraph，此文件是早期 POC 代码 |

**C. 废弃执行模式链路**:

| 位置 | 内容 | 原因 |
|------|------|------|
| `main.py:350-357` | `_executor_mode()` — 接受 `SHERPA_EXECUTOR_MODE` 但只允许 `k8s_job`，其他值直接 `raise RuntimeError` | 已无其他模式，env var 接口可简化 |
| `codex_helper.py:404-406` | `_opencode_container_mode_enabled()` — 当 `SHERPA_EXECUTOR_MODE != "k8s_job"` 时启用 Docker 容器模式 | 永远返回 `False`，因为只有 k8s_job 模式 |
| `docker/Dockerfile.opencode` | OpenCode Docker 容器镜像 | 仅在非 k8s 模式下使用，而该模式已被废弃 |
| `docker-compose.yml` 中 `sherpa-opencode` 服务 | `profiles: [opencode]` | 依赖已废弃的容器模式 |
| `codex_helper.py` 中 `_docker_opencode_env_args()`, `_build_docker_opencode_image()` 等 | Docker 容器模式相关函数 | 在 k8s_job 模式下永远不会执行 |

**操作**:
1. 删除 `ossfuzz_auto.py` 和 `main_brain.py`
2. 简化 `_executor_mode()` 为直接返回 `"k8s_job"`（移除 env var 检查）
3. 在 `codex_helper.py` 中标注 Docker 容器模式相关函数为 `@deprecated`（暂不删除，因为 docker-compose 本地开发可能仍需要。但需在代码中明确标注这是遗留路径）
4. 从 `.env.example` 和 `docker-compose.yml` 中移除 `SHERPA_OSS_FUZZ_REPO_URL`、`SHERPA_OSS_FUZZ_DIR`（已无代码读取）

**D. 死 API 端点和前端死代码**:

| 位置 | 端点/函数 | 原因 |
|------|-----------|------|
| `main.py:3271` | `GET /api/health` | 前端从未调用 |
| `main.py:3266` | `GET /api/metrics` | 前端从未调用 |
| `main.py:4940` | `POST /api/task/{job_id}/resume` | 前端无对应函数 |
| `frontend-next/lib/api/client.ts:103-115` | `getOpencodeProviderModels()` | 导出但从未被任何组件 import |

**操作**:
- `/api/health` 和 `/api/metrics`：保留但标注为内部/运维端点（不是死代码，而是缺少前端 UI。Day 5.6 的 `/healthz` 会提供标准健康检查）
- `/api/task/{job_id}/resume`：确认是否计划实现前端恢复功能。如果短期不做，添加注释说明
- `getOpencodeProviderModels()`：从 `client.ts` 中删除未使用的导出和类型定义

**验证**:
```bash
# 死模块
python -c "import importlib; importlib.import_module('langchain_agent.ossfuzz_auto')" 2>&1 | grep -i "error"  # 应报错（文件已删除）
# 死别名
grep -rn "apply_minimax_env_source\|CodexPatcher\|_resolve_minimax\|_node_repro_crash" harness_generator/  # 目标: 0
# 死 env var
grep -rn "SHERPA_OSS_FUZZ_REPO_URL\|SHERPA_OSS_FUZZ_DIR" .env.example docker-compose.yml  # 目标: 0
pytest tests/ --tb=short -q
```

### 任务 5.2: 清理孤立文件和空目录

**发现清单**:

| 路径 | 类型 | 原因 |
|------|------|------|
| `RUN_FULL_TEST_FLOW.ps1` | 孤立脚本 | 远古 Windows 测试脚本（时间戳 2010 年），项目全面 Linux/Docker 化 |
| `harness_generator/scripts/gather_reports.py` | 孤立脚本 | 无 CI/Makefile/文档引用 |
| `harness_generator/scripts/generate_reports.py` | 孤立脚本 | 同上 |
| `harness_generator/scripts/sort_jobs.py` | 孤立脚本 | 同上 |
| `harness_generator/scripts/summarize.py` | 孤立脚本 | 同上 |
| `/false/` | 空目录 | 疑似工具产生的垃圾目录 |
| `outputs/` | 近空目录 | 仅含 2 个状态机示意图，非运行时产物 |
| `setup-env.sh`（根目录） | 重复文件 | 与 `harness_generator/setup-env.sh` 功能重叠，后者更完整 |

**操作**:
1. 删除 `RUN_FULL_TEST_FLOW.ps1`
2. 删除 `/false/` 空目录
3. 将 `harness_generator/scripts/` 的 4 个文件移入 `harness_generator/scripts/_archived/`（或直接删除，取决于是否需要保留历史参考）
4. `outputs/` 中的图片如有参考价值移入 `docs/diagrams/`，否则删除
5. 合并两个 `setup-env.sh`：保留 `harness_generator/setup-env.sh`（更完整），根目录的改为 symlink 或删除

### 任务 5.3: 清理 promefuzz-mcp 占位实现

**发现清单** — 8 个空壳函数（全部 return 空值或 stub 字符串）:

| 文件 | 函数 | 返回 |
|------|------|------|
| `promefuzz_mcp/llm/client.py:101` | `LLMClient.embed()` | `[]` + "Placeholder" 注释 |
| `promefuzz_mcp/comprehender/purpose.py:17` | `comprehend()` | stub 字符串 |
| `promefuzz_mcp/comprehender/func_usage.py:17,22` | `comprehend()`, `comprehend_all()` | stub |
| `promefuzz_mcp/comprehender/func_relevance.py:17` | `comprehend()` | `{}` |
| `promefuzz_mcp/preprocessor/complexity.py:18` | `calculate()` | placeholder |
| `promefuzz_mcp/preprocessor/incidental.py:16` | `extract()` | placeholder |
| `promefuzz_mcp/preprocessor/relevance.py:17,30,43` | `TypeRelevance/ClassRelevance/CallRelevance.calculate()` | placeholder |
| `promefuzz_mcp/server_tools.py:207` | `calculate_type_relevance()` — TODO 注释 | `{}` |

**操作**:
- 这些函数由 `SHERPA_PROMEFUZZ_ENABLE_COMPREHENDER` 环境变量控制（默认关闭）
- 在每个 placeholder 函数中添加 `raise NotImplementedError("...")` 替代静默返回空值，防止误以为功能正常
- 在 `promefuzz-mcp/README.md` 中明确标注哪些模块已实现、哪些待实现

### 任务 5.4: 清理无用和未文档化的环境变量

**问题**: 代码中读取 199 个 env var，`.env.example` 仅记录 42 个。存在大量定义但从未读取、读取但从未文档化的变量。

**A. 删除死定义（定义但代码从未读取）**:

| 变量 | 定义位置 | 状态 |
|------|----------|------|
| `SHERPA_GITNEXUS_AUTO_ANALYZE` | `.env.example`, `docker-compose.yml` | 代码中无 `os.environ.get("SHERPA_GITNEXUS_AUTO_ANALYZE")` 调用 |
| `SHERPA_GITNEXUS_SKIP_EMBEDDINGS` | `.env.example`, `docker-compose.yml` | 同上 |
| `SHERPA_AUTO_INIT_OSS_FUZZ` | `k8s/base/configmap.yaml` | 同上 |

**操作**: 从 `.env.example`、`docker-compose.yml`、`configmap.yaml` 中移除

**B. 补充文档化（代码读取但无文档的关键变量）**:

在 `.env.example` 中分组补充，至少覆盖以下高频使用变量：

```bash
# --- OpenCode 调优 ---
SHERPA_OPENCODE_CONTEXT_MAX_LINES=       # LLM 上下文最大行数
SHERPA_OPENCODE_CONTEXT_MAX_CHARS=       # LLM 上下文最大字符数
SHERPA_OPENCODE_STAGE_SKILLS_PATH=       # stage skill 模板路径
SHERPA_OPENCODE_DOCKER_IMAGE=            # OpenCode Docker 镜像

# --- 运行调优 ---
SHERPA_RUN_RSS_LIMIT_MB=                 # 进程 RSS 内存上限
SHERPA_RUN_PLATEAU_PULSES=               # 覆盖率瓶颈检测脉冲数
SHERPA_SEED_FILTER_MODE=                 # 种子过滤模式

# --- K8s 分析伴侣 ---
SHERPA_K8S_ANALYSIS_COMPANION_ENABLED=0  # 启用 promefuzz MCP 分析
SHERPA_K8S_ANALYSIS_COMPANION_IMAGE=     # 分析伴侣镜像
SHERPA_K8S_ANALYSIS_COMPANION_PORT=8080  # 分析伴侣端口

# --- 嵌入模型（OpenRouter）---
OPENROUTER_EMBEDDING_API_KEY=            # 嵌入 API key
OPENROUTER_EMBEDDING_MODEL=              # 嵌入模型名
```

**C. 统一代理配置命名**:

当前 3 套代理变量并存：

| 用途 | 当前命名 | 统一为 |
|------|----------|--------|
| 通用 | `HTTP_PROXY`, `HTTPS_PROXY` | 保留（标准） |
| Docker 内 | `SHERPA_DOCKER_HTTP_PROXY` | 保留（Docker 容器特有） |
| Git | `SHERPA_GIT_HTTP_PROXY` | 合并到 `SHERPA_DOCKER_HTTP_PROXY`（Git 运行在 Docker 内） |

**操作**: 在代码中将 `SHERPA_GIT_HTTP_PROXY` 的读取改为读 `SHERPA_DOCKER_HTTP_PROXY`，保留前者作为 fallback 一个版本后删除

### 任务 5.5: 清理 OSS-Fuzz 废弃基础设施

**背景**: `ossfuzz_auto.py`（已在 5.1 中标记删除）是 OSS-Fuzz 集成的核心模块但从未被 import。围绕它还存在大量**仍在运行但无实际用途**的基础设施：

**全量盘点** — OSS-Fuzz 在项目中的 17 个触点：

| 层级 | 位置 | 内容 | 是否活跃 |
|------|------|------|----------|
| 死代码 | `ossfuzz_auto.py` (606 行) | 完整 OSS-Fuzz 自动化模块 | ❌ 从未 import |
| 配置字段 | `persistent_config.py:97` | `oss_fuzz_dir: str = ""` | ⚠️ 存储在 DB 但下游不消费 |
| K8s job | `main.py:4820-4849` | OSS-Fuzz auto-init 逻辑 + `_ensure_oss_fuzz_checkout()` | ⚠️ `SHERPA_AUTO_INIT_OSS_FUZZ` 默认 "0"（关闭） |
| K8s worker | `k8s_job_worker.py:201` | 从 payload 读 `oss_fuzz_dir` | ⚠️ 传递但不使用 |
| Docker | `docker-compose.yml:24-52` | `sherpa-oss-fuzz-init` bootstrap 容器 | ⚠️ 每次 compose up 都克隆 oss-fuzz 仓库 |
| Docker | `docker-compose.yml:72,201,269` | volume mount `sherpa-oss-fuzz` | ⚠️ 挂载 20Gi 卷 |
| K8s PVC | `k8s/base/pvc-shared.yaml:44-49` | `sherpa-oss-fuzz` 20Gi PVC | ⚠️ 占用存储 |
| K8s PV (dev) | `k8s/overlays/dev/pv-dev.yaml:42-51` | `pv-sherpa-dev-oss-fuzz` | ⚠️ 占用存储 |
| K8s PV (prod) | `k8s/overlays/prod/pv-prod.yaml:52-63` | `pv-sherpa-prod-oss-fuzz` | ⚠️ 占用存储 |
| K8s ConfigMap | `configmap.yaml:11,21,36,37` | 4 个 OSS-Fuzz 相关变量 | ⚠️ 定义但核心逻辑不读取 |
| Dockerfile | `docker/Dockerfile.web:160-161` | 创建 `/shared/oss-fuzz` 目录 | ⚠️ 空目录占位 |
| CI/CD | `deploy-dev.yml`, `deploy-prod.yml` | 引用 `${SHARED_ROOT}/oss-fuzz` | ⚠️ 目录操作 |
| .env.example | 3 个变量 | `SHERPA_K8S_PVC_OSS_FUZZ` 等 | ⚠️ 文档 |
| .gitignore | 1 行 | 忽略 `oss-fuzz/` | 低影响 |

**注意**: `NonOssFuzzHarnessGenerator` 类名含 "OssFuzz" 但它是核心引擎，**不应删除**。它引用 `contrib/oss-fuzz/corpus.zip` 作为种子发现约定，这与 OSS-Fuzz 基础设施无关。

**操作**:
1. 删除 `docker-compose.yml` 中的 `sherpa-oss-fuzz-init` 服务（停止自动克隆）
2. 删除 `docker-compose.yml` 中相关 volume mount 和 volume 定义
3. 从 K8s ConfigMap 移除: `SHERPA_DEFAULT_OSS_FUZZ_DIR`, `SHERPA_K8S_PVC_OSS_FUZZ`, `SHERPA_OSS_FUZZ_REPO_URL`, `SHERPA_AUTO_INIT_OSS_FUZZ`
4. 删除 K8s PVC `sherpa-oss-fuzz`（`pvc-shared.yaml`）和对应的 dev/prod PV
5. 删除 `main.py` 中 `_ensure_oss_fuzz_checkout()` 函数和 auto-init 逻辑
6. 从 `persistent_config.py` 的 `WebPersistentConfig` 中移除 `oss_fuzz_dir` 字段
7. 更新 `.env.example` 移除 3 个 OSS-Fuzz 变量
8. 更新 deploy workflow 移除 oss-fuzz 目录引用
9. 更新 `Dockerfile.web` 移除 `/shared/oss-fuzz` 目录创建

**⚠️ 注意**: 部署后需手动清理已有 K8s 集群中的 PVC/PV 资源

### 任务 5.6: 废弃 Fix 循环链路

**背景**: Fix 循环（`fix_build`、`fix_crash`、`fix-harness`）已被 **repair 态 plan 循环**替代。经代码追踪确认：

- **`fix_build` 已是死路径**: `_route_after_build_state()` (line 10812-10819) 只返回 `"plan"` 或 `"run"`，从不返回 `"fix_build"`。
- **`fix_crash` 已是死路径**: `_route_after_run_state()` (line 10822-10841) 只返回 `"crash-triage"` / `"coverage-analysis"` / `"plan"`，从不返回 `"fix_crash"`。
- **`fix-harness` 保留不动**: `_route_after_crash_triage_state()` 在 `harness_bug` 时路由到 `fix-harness`，但这 **不是循环**。它只跑一次针对性修复，成功则 `build`，失败/noop 则 `restart_to_plan=True` 自动回 `plan`（repair 态）。这是合理的优化路径。
- **Repair plan 循环已完善**: 有 6 个 repair skill（`plan_repair_build/crash/coverage` + `synthesize_repair_build/crash/coverage`），含 constraint memory、strategy gate、error digest 等完整机制。

但残留代码仍然很多：

**全量盘点**:

| 层级 | 内容 | 代码量 |
|------|------|--------|
| 节点实现 | `_node_fix_build()` (workflow_graph.py:6327-7938) | ~1600 行 |
| 节点实现 | `_node_fix_crash()` (workflow_graph.py:9549-10032) | ~480 行 |
| 节点实现 | `_node_fix_harness_after_run()` (workflow_graph.py:10035-10125) | ~90 行 |
| 路由函数 | `_route_after_fix_build_state()`, `_route_after_fix_crash_state()`, `_route_after_fix_harness_state()` | ~60 行 |
| 图边定义 | 条件边 fix_build→*, fix_crash→*, fix-harness→* (workflow_graph.py:11198-11228) | ~30 行 |
| 辅助函数 | fix_build 规则引擎、hotfix 逻辑、noop 检测等 (workflow_graph.py:2450-2560) | ~110 行 |
| 状态字段 | `fix_build_attempts`, `fix_build_noop_streak`, `fix_build_attempt_history`, `fix_build_rule_hits`, `fix_build_terminal_reason`, `fix_build_last_diff_paths`, `fix_action_type`, `max_fix_rounds`, `same_build_error_repeats`, `fix_crash_attempts`, `fix_harness_attempts` | 11 个字段 |
| 环境变量 | `SHERPA_FIX_BUILD_MAX_ATTEMPTS`, `SHERPA_FIX_BUILD_MAX_NOOP_STREAK`, `SHERPA_FIX_BUILD_FEEDBACK_HISTORY`, `SHERPA_FIX_BUILD_CONTEXT_MAX_CHARS`, `SHERPA_FIX_BUILD_STDOUT_MAX_CHARS`, `SHERPA_FIX_BUILD_STDERR_MAX_CHARS`, `SHERPA_FIX_BUILD_KEEP_RECENT_ERRORS`, `SHERPA_FIX_BUILD_SAME_SIGNATURE_TO_PLAN` | 8 个 |
| OpenCode Skills | `fix_build/`, `fix_crash_harness_error/`, `fix_crash_upstream_bug/` | 3 个目录 |
| 前端 | `TaskProgressPanel.tsx:30-38` 显示 fix_build 进度 | ~8 行 |
| 测试 | `test_workflow_build_resilience.py` 中 28 个 fix_build 测试 | ~900 行 |
| main.py | fix stage 名称映射、参数传递、终止原因上报 | ~20 行 |

**Repair plan 循环机制**（已实现，是 fix 循环的替代）:
```
[Build/Crash/Coverage 失败]
    ↓  repair_mode=True, repair_origin_stage="build|crash|coverage"
_node_plan()
  → 检测 repair_mode, 选择 skill: plan_repair_build|crash|coverage
  → 注入: error digest, constraint memory, strategy gate
  → 产出: 修改后的 PLAN.md, targets.json, execution_plan.json
    ↓
_node_synthesize()
  → 检测 repair_mode, 选择 skill: synthesize_repair_build|crash|coverage
  → 产出: 修改后的 harness/build 脚本
    ↓
_node_build()
  → 成功: repair_mode 清除，进入正常流程
  → 失败 + 相同签名重复 ≥3 次: repair_strategy_force_change=True, 回到 plan
```

**操作（分两步）**:

**Step 1: 断开剩余 fix 路由，统一走 plan repair 路径（本轮执行）**

实际只需改 **1 处路由**：

- `fix_build`: **已不可达**（路由函数从不返回 `"fix_build"`）。移除图中残留的节点注册和条件边。
- `fix_crash`: **已不可达**（路由函数从不返回 `"fix_crash"`）。移除图中残留的节点注册和条件边。
- `fix-harness`: **唯一活跃路由** — 修改 `_route_after_crash_triage_state()` (line 10937-10938) 把 `fix-harness` 的职责也交给 `plan repair` 路径,新建 `fix-harness` 态的 `plan repair` 路径:
  ```python
  # 之前:
  if label == "harness_bug":
      return "fix-harness"

  # 之后: harness_bug 也走 plan repair 路径
  if label == "harness_bug":
      return "plan"  # repair_mode + repair_origin_stage="crash" 由上游 crash-triage 节点设置
  ```
- 确保 `_node_crash_triage()` 在判定 `harness_bug` 时正确设置 `repair_mode=True` + `repair_origin_stage="crash"`（参考 `_mark_build_repair_state()` 的实现）
- 移除图中 `fix-harness` 节点注册和条件边
- 在 `_node_fix_build`, `_node_fix_crash`, `_node_fix_harness_after_run` 开头添加 deprecation warning log（保留代码但不可达）
- 新建 `fix-harness` 态的 `plan repair` 路径,写好针对修复 harness 优化的提示词和 skill

**Step 2: 后续清理（本轮不含，放入后续迭代）**
- 删除 3 个节点实现函数（~2170 行）
- 删除 3 个路由函数
- 从图中移除 fix_build / fix_crash / fix-harness 节点注册
- 移除 11 个 FuzzWorkflowState 字段
- 移除 8 个 `SHERPA_FIX_BUILD_*` 环境变量
- 删除 3 个 OpenCode skill 目录（`fix_build/`, `fix_crash_harness_error/`, `fix_crash_upstream_bug/`）
- 清理前端 `TaskProgressPanel.tsx` 中 fix 进度显示
- 删除或重写 28 个 fix_build 测试

**为什么分两步**: Fix 循环涉及 ~2500 行代码 + 28 个测试 + 前端 + 3 个 skill 目录，一次性删除风险太大。先断边确认 repair plan 路径工作正常，验证稳定后再彻底清理。

**验证**:
```bash
# 确认 build 失败走 plan（repair 态）— 已是现有行为
python -c "
from harness_generator.src.langchain_agent.workflow_graph import _route_after_build_state
state = {'restart_to_plan': True, 'error': {'code': 'build_failed'}}
result = _route_after_build_state(state)
assert result == 'plan', f'Should route to plan (repair), got: {result}'
print('OK: build failure → plan repair (existing behavior)')
"
# 确认 crash triage 不再路由到 fix-harness
python -c "
from harness_generator.src.langchain_agent.workflow_graph import _route_after_crash_triage_state
state = {'crash_triage_result': 'harness_bug'}
result = _route_after_crash_triage_state(state)
assert result != 'fix-harness', f'Should not route to fix-harness, got: {result}'
print('OK: crash triage → plan repair (new behavior)')
"
# 确认 fix_build 节点不再注册在图中
python -c "
from harness_generator.src.langchain_agent.workflow_graph import build_fuzz_workflow
graph = build_fuzz_workflow()
nodes = [n for n in graph.nodes if 'fix_build' in str(n)]
assert len(nodes) == 0, f'fix_build should be removed from graph, found: {nodes}'
print('OK: fix_build removed from graph')
"
pytest tests/ --tb=short -q
```

### 任务 5.7: 更新过时文档

**A. `.env.example`**（已在 5.4 中覆盖）:
- 替换 "AI provider credentials (MiniMax only)" 为通用描述
- 用 `LLM_API_KEY`/`LLM_BASE_URL`/`LLM_MODEL` 作为主变量名（配合 Day 2 的 secrets 解耦）
- 移除 docker-compose 中代理的硬编码默认值 `http://host.docker.internal:7897`，改为空

**B. `docs/API_REFERENCE.md`**:
- 替换所有 `"MiniMax-M2.7-highspeed"` 示例为 `"deepseek-reasoner"` 或通用占位符

**C. `promefuzz-mcp/DEPLOY.md`**:
- 替换 `MINIMAX_API_KEY` 引用为通用 `LLM_API_KEY`

**D. 向后兼容注释清理**（workflow_graph.py、codex_helper.py、fuzz_unharnessed_repo.py 中 8+ 处）:
- 审查每个 "backward compat" / "legacy" 注释，确认关联代码是否仍需要
- 对已无用的兼容逻辑直接删除（不再保留注释占位）

### 任务 5.8: K8s 健康检查 + dev 自动部署

**A. 添加 `/healthz` 端点**:
- 在 FastAPI 中添加，检查服务存活 + DB 连接
- K8s manifest 中添加 `livenessProbe` / `readinessProbe`

**文件**: `main.py`, `k8s/base/web-deployment.yaml`

**B. dev 自动部署**:
- `deploy-dev.yml` 添加 `on: push: branches: [dev]`

**文件**: `.github/workflows/deploy-dev.yml`

**验证**:
```bash
# 全量测试
pytest tests/ -v
# 死代码验证
grep -rn "apply_minimax_env_source\|CodexPatcher\|_resolve_minimax\|GITNEXUS" harness_generator/ .env.example docker-compose.yml  # 目标: 0
# OSS-Fuzz 基础设施验证
grep -rn "sherpa-oss-fuzz-init\|SHERPA_AUTO_INIT_OSS_FUZZ" docker-compose.yml k8s/  # 目标: 0
# 孤立文件验证
ls RUN_FULL_TEST_FLOW.ps1 false/  # 应不存在
# 文档验证
grep -i "minimax only" .env.example docs/ promefuzz-mcp/  # 目标: 0
```

---

## 5 天产出总结

| Day | 主题 | 涉及文件 | 预期效果 |
|-----|------|----------|----------|
| 1 | 依赖治理 | `requirements.web.txt`, `requirements.txt` | 消除部署不确定性 |
| 2 | CI 测试 + OpenCode 配置 + Secrets 解耦 | `test.yml`(新), `persistent_config.py`, K8s/GHA secrets | 测试自动化，provider 随时可换，secret 不绑 provider |
| 3 | 日志统一 | `main.py`, `fuzz_unharnessed_repo.py`, `codex_helper.py` | 统一 loguru，消除 print() |
| 4 | 异常收窄 | `errors.py`(新), `main.py` | except Exception 减少 40%+ |
| 5 | **全面清理** | 死代码、孤立文件、env var、过时文档、健康检查 | 移除全部遗留垃圾，补齐 env var 文档 |

### Day 5 清理量化目标

| 指标 | 目标 |
|------|------|
| 删除废弃模块 | 2 个（`ossfuzz_auto.py` 605 行, `main_brain.py` ~130 行） |
| 删除死函数/别名 | 6 个 |
| 标注废弃执行模式 | Docker 容器模式链路标注 `@deprecated` |
| 处理死 API 端点 | 3 个端点 + 1 个前端死函数 |
| 删除孤立文件 | 8+ 个 |
| 清理 OSS-Fuzz 基础设施 | 移除 bootstrap 容器、K8s PVC/PV、5+ env var、deploy 引用 |
| 断开 Fix 循环图边 | fix_build 节点移除，fix_crash/fix-harness 路由改为 plan repair |
| promefuzz placeholder 标注 | 8 个函数加 `NotImplementedError` |
| 移除死 env var 定义 | 8+ 个（含 `OSS_FUZZ_*`、`GITNEXUS_*`、`FIX_BUILD_*` 停用后不再需要） |
| 新文档化 env var | 15+ 个（补入 `.env.example`） |
| 更新过时文档 | 3 个文件 |
| 清理向后兼容注释 | 8+ 处 |
| **本轮预计删除/断开代码量** | **~800+ 行**（废弃模块 + 死函数 + fix_build 节点 + OSS-Fuzz 基础设施） |
| **后续清理可删除代码量** | **~2500+ 行**（fix_crash/fix-harness 实现 + 28 个测试 + 3 个 skill 目录） |

---

## 后续迭代（本轮不含）

以下为 5 天之后的改进方向，按优先级排序：

| 优先级 | 项目 | 工作量 |
|--------|------|--------|
| P1 | 分解 FuzzWorkflowState（195 字段 God Object）→ 领域子状态 | XL |
| P1 | 拆分 workflow_graph.py 为 `nodes/` 模块包 | L |
| P1 | 拆分 NonOssFuzzHarnessGenerator（77 方法巨类） | L |
| P1 | 修复覆盖率瓶颈恢复（fix_plan.md 唯一未修复项） | L |
| P1 | 彻底删除 fix 循环代码（~2500 行实现 + 28 个测试 + 3 个 skill 目录 + 前端 fix 显示） | L |
| P1 | 拆分 main.py 路由为 FastAPI Router（jobs, tasks, config, system） | M |
| P2 | 配置优先级统一（env > DB > default）+ Pydantic 验证 | M |
| P2 | 前端测试覆盖（ConfigPanel, TaskProgressPanel） | M |
| P2 | 集中化日志收集（Loki + Grafana） | M |
| P2 | 收窄 workflow_graph.py + fuzz_unharnessed_repo.py 异常 | L |
| P2 | 实现 promefuzz-mcp 的 8 个 placeholder 函数 | M |

---

## 验证清单

> 说明：以下勾选为“按当前代码库本地核对”的结果；未勾选表示未完成或尚未完成可复现验证。
>
> 当前回归阻塞：`pytest tests/ -v` 在本地受 `test_api_stability.py` 的 Postgres 依赖（`127.0.0.1:55432`）影响，同时存在独立用例失败（如 `test_dockerize_path_translation.py::test_dockerize_autoinstall_triggers_for_build_py_from_fuzz_cwd`），需单独处理。

### 依赖 & CI（Day 1-2）
- [ ] `pip install -r docker/requirements.web.txt` 在干净环境中成功安装（未在干净环境复验）
- [x] CI workflow 在 PR/push 上自动触发并执行测试（`test.yml` 已配置 `pull_request` + `push(dev/main)`）
- [x] OpenCode 新 provider（如 openrouter）不再被白名单过滤（`persistent_config.py` 已为 registry 驱动）
- [x] `grep -r "sherpa-deepseek" k8s/ .github/` 返回 0（当前代码范围内未检出）
- [x] `grep -r "DEEPSEEK_API_KEY_DEV\\|DEEPSEEK_API_KEY_PROD" .github/` 返回 0（deploy workflow 已移除 fallback）
- [x] 无 Django 依赖残留（当前代码范围内未检出 `django` 依赖引用）

### 日志 & 异常（Day 3-4）
- [x] `grep -c "print(" main.py` 接近 0（当前为 0）
- [x] `grep -c "except Exception" main.py` < 50（当前为 39）
- [x] `errors.py` 中定义了 `SherpaError` 异常层次（已存在）

### 清理（Day 5）
- [x] `ossfuzz_auto.py` 和 `main_brain.py` 已删除（当前文件已不存在）
- [x] `grep -rn "apply_minimax_env_source\|CodexPatcher\|_resolve_minimax\|_node_repro_crash" harness_generator/` 返回 0
- [x] `ls RUN_FULL_TEST_FLOW.ps1 false/` 不存在（当前均不存在）
- [x] `grep -i "minimax only" .env.example` 返回 0（当前未检出）
- [x] `grep "GITNEXUS\|SHERPA_OSS_FUZZ_REPO_URL\|SHERPA_AUTO_INIT_OSS_FUZZ" .env.example docker-compose.yml k8s` 返回 0
- [x] OSS-Fuzz bootstrap 容器从 docker-compose 移除（当前未检出 `sherpa-oss-fuzz-init`）
- [x] K8s PVC `sherpa-oss-fuzz` 从 manifest 移除（base/overlay 与 deploy workflow 均已清理）
- [x] fix_build / fix_crash 节点从 workflow graph 移除（图中不再注册，也无可达边）
- [x] crash 路由完全切离 fix_crash/fix-harness（`harness_bug -> plan(repair_fix_harness)`）
- [x] `.env.example` 覆盖 `SHERPA_OPENCODE_*`、`SHERPA_K8S_ANALYSIS_COMPANION_*`、embedding 关键变量
- [x] 前端 `getOpencodeProviderModels()` 死代码已清除（当前未检出）
- [x] `/healthz` 端点已提供 200 响应与 DB 状态字段（`/api/health` 继续保留）
- [ ] `pytest tests/ -v` 全量通过（未达成：本地 Postgres 依赖与独立失败用例待清理）
