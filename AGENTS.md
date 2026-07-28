## Skills

当前启用 superpowers 插件 (`superpowers@git+https://github.com/obra/superpowers.git`)。使用 OpenCode 原生 `skill` 工具加载。

## 项目提交流程（Git/CI）

目标：所有变更先在 `dev` 验证，再进入 `main` 发布，避免未验证代码直达生产。

### 分支与发布路径

1. 开发分支：`codex/*`（或个人功能分支）。
2. 验证分支：`dev`（集成测试与部署验证）。
3. 发布分支：`main`（生产发布来源）。

标准路径：

1. `feature branch` -> PR 到 `dev`
2. `dev` 工作流通过（含部署/健康检查）
3. `dev` -> PR 到 `main`
4. `main` 工作流通过并完成生产发布

### 强制约束

1. 禁止直接 `push` 到 `dev` 和 `main`（仅允许 PR 合并）。
2. `main` 只接受来自 `dev` 的 PR（不接受功能分支直提）。
3. 每个 PR 必须包含：
   - 变更摘要（做了什么）
   - 风险与回滚点（失败怎么退回）
   - 验证结果（最少一条可复现验证）
4. 默认禁止使用管理员强制合并：
   - 禁止使用 `gh pr merge --admin` 或等效绕过保护规则的方式。
   - `main` 合并必须先满足 Review 要求与必需检查通过，再由人工执行合并。
   - 仅当用户在当前会话中明确授权"紧急绕过"时，才允许一次性管理员合并，并需在 PR 与 Linear 记录原因。
5. fuzz 验证阶段默认禁止 AI 参与：
   - `run` 与 `repro_crash` 阶段仅允许源码构建与命令执行验证，不允许 AI 改写代码或 AI 生成种子参与验证结果判定。
   - 默认配置：`SHERPA_VERIFY_STAGE_NO_AI=1`。
   - 如需临时回退旧行为，必须显式设置 `SHERPA_VERIFY_STAGE_NO_AI=0` 并在 PR/Linear 说明原因。

### 操作步骤（执行版）

1. 从最新 `dev` 拉分支开发：
   - `git checkout dev && git pull`
   - `git checkout -b codex/<topic>`
2. 本地完成修改并自检（最小语法/配置校验通过）。
3. 推送分支并创建 PR 到 `dev`。
4. 等待 `Deploy Dev` 及相关检查通过；失败先修复后再合并。
5. `dev` 稳定后，创建 `dev -> main` PR。
6. 等待 `Deploy Prod`（或主线发布流）通过后合并。

### 热修复规则

1. 生产故障允许临时热修，但必须：
   - 先在独立分支修复并保留 PR 记录；
   - 修复后尽快回补到 `dev`，保证分支一致；
   - 在 PR 中注明 `hotfix` 与影响范围。

## 关键配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `SHERPA_OPENCODE_IDLE_TIMEOUT_SEC` | 600 | OpenCode agent 空闲超时（秒） |
| `SHERPA_PARALLEL_FUZZERS` | 2 | 并行 fuzzer 数量 |
| `SHERPA_RUN_UNLIMITED_ROUND_BUDGET_SEC` | 7200 | 无限模式总轮次预算 |
| `SHERPA_VERIFY_STAGE_NO_AI` | 1 | fuzz 验证禁用 AI |

## Coverage 插桩

build 阶段通过 `_inject_coverage_instrumentation()` 自动向 `build.py` 注入
`-fsanitize-coverage=trace-pc-guard,inline-8bit-counters` 标志，确保 libFuzzer
覆盖率反馈可用。注入在每次 build 重试前执行，防止被 fix_build agent 覆盖。

相关文件：
- `harness_generator/src/langchain_agent/workflow_graph.py` — 注入逻辑
- `harness_generator/src/langchain_agent/opencode_skills/synthesize/SKILL.md` — AI 契约
- `harness_generator/src/langchain_agent/coverage_replay.py` — 逐种子回放分析

## Cursor Cloud specific instructions

本节面向后续 Cloud Agent（启动脚本已跑过、依赖已装好）。启动脚本只做依赖刷新：Python venv (`.venv/`，装 `docker/requirements.web.txt` + `pytest`) 与 `frontend-next` 的 npm 依赖。系统依赖（PostgreSQL 16）由 VM 快照携带，不在启动脚本内。

### 本地可运行 / 不可运行的范围
- 可本地运行：控制面后端 API（FastAPI）、PostgreSQL、Next.js 前端、pytest、前端 lint/build。
- 不可本地端到端运行的部分：真正的 fuzz 阶段执行。`_executor_mode()` 只支持 `k8s_job`，每个阶段以 Kubernetes Job 执行；本 VM 无 `kubectl`/集群，因此提交任务后子任务会停在 `plan`，日志反复出现 `k8s job submit failed: ... No such file or directory: 'kubectl'`。这是预期边界，不是 bug。完整 fuzz 还需外部 LLM key（`LLM_key`/`OPENAI_API_KEY`）与 Docker/DinD。

### 启动服务（非显而易见项）
- PostgreSQL 需手动启动，且被特意配置为监听 **55432** 端口（与 CI `test.yml` 的 `55432:5432` 一致，后端测试 `tests/test_api_stability.py` 硬编码连 `127.0.0.1:55432`）：
  - `sudo pg_ctlcluster 16 main start`
  - 账号/库：`sherpa` / `sherpa` / `sherpa`。
- 后端（在 `harness_generator/src/langchain_agent/` 下）：
  - `HOST=0.0.0.0 PORT=8001 DATABASE_URL="postgresql://sherpa:sherpa@127.0.0.1:55432/sherpa" ../../../.venv/bin/python main.py`
  - 直接 `python main.py` 默认绑 `127.0.0.1:8000`；缺 `DATABASE_URL` 会直接拒绝启动（Postgres-only）。
  - 健康检查：`GET /healthz` 返回 `{"ok":true,...,"db":{"ok":true}}`。
- 前端（`frontend-next/`）独立跑时需把 API 指向后端（默认是 `/api`，靠 gateway）：
  - `NEXT_PUBLIC_API_BASE="http://localhost:8001/api" npm run dev`（端口 3000）。后端 CORS 允许 `*`。

### 测试 / lint / build
- 后端测试：仓库根目录 `.venv/bin/python -m pytest tests/`（需 Postgres 在 55432）。命令见 `.github/workflows/test.yml`。
- 前端：`cd frontend-next && npx next lint` / `npm run build`。CI 里 `npx vitest run` 目前无测试文件（正常）。

### 已知非环境导致 / 环境导致的测试失败（勿误判为回归）
- `tests/test_opencode_skill_contracts.py::test_fix_build_contract_keeps_vcpkg_and_compiler_rules` 与 `tests/test_run_cmd_streaming.py::test_run_cmd_fails_when_declared_ports_require_missing_vcpkg`：在 `main`/`dev` 的 CI 上同样失败，属既有仓库问题。
- `tests/test_codex_helper_sentinel.py` 的约 15 个用例在本 VM 失败：`done` sentinel 判新旧用 `mtime < attempt_start - 1e-3`，而本机 `/tmp`（overlayfs）文件 mtime 比 wall-clock 落后约 3ms，超过 1ms 容差被误判为 stale。这是本环境 mtime/时钟粒度问题（CI 的 ext4 不触发），非代码回归；请勿为此改代码。
