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
