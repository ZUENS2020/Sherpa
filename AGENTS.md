## Skills

No repository-local skill is currently enabled.

## Major Change Rule (Linear)

For every major project change or improvement, follow this workflow strictly:

1. Before writing code, first clarify the change goal and planned scope (what problem this change solves, and the rough implementation content).
2. Record that goal + scope summary in a new Linear issue via MCP, and auto-apply labels based on change type.
3. Assign the issue to yourself.
4. Set issue status to `In Progress`.
5. Ask the user to confirm whether execution should start.
6. Only after explicit user confirmation, complete implementation and verification.
7. Set issue status to `Done`.
8. Add one final comment summarizing:
   - what changed,
   - validation/test result,
   - impact/risk notes.

Progress sync requirement (ongoing):

1. Do not sync progress to Linear only once in the first round.
2. For any subsequent change related to the same topic, continue updating the same Linear issue with progress comments.
3. Each related progress comment should include:
   - what was changed in this increment,
   - current verification result,
   - remaining work or next step.
4. Keep comments cumulative and chronological so the issue reflects the latest full context at any point in time.

Constraints:

1. Do not start implementation before the change purpose and rough scope are written into Linear and the issue is set to `In Progress`.
2. Do not start implementation without explicit user confirmation after the Linear write is complete.
3. Do not mark `Done` without a completion summary comment.
4. Enforce type-based labels on the issue (create labels first if missing, then apply them).
5. If new work is clearly related to an existing in-progress issue, continue in that issue and add progress comments instead of silently updating outside Linear.
6. All Linear comments must be written in Chinese.
7. Issue descriptions used for progress sync should also be written in Chinese unless the user explicitly requests another language.
8. Linear comments must use real line breaks. Never write literal `\n` or `/n` text into comments.
9. Before posting a Linear comment, verify rendered body uses actual multi-line formatting (single escaping only).

Type-to-label mapping for major changes:

1. Frontend/UI changes -> `frontend`
2. Backend/API/Workflow logic -> `backend`
3. Container/Compose/Deploy/Runtime infra -> `infra`
4. CI/CD and GitHub Actions -> `ci/cd`
5. Docs/README/architecture docs -> `docs`
6. Tests/quality/coverage -> `test`
7. Bug fixes (root-cause correction) -> `bugfix`
8. Refactor without behavior change -> `refactor`
9. Security hardening -> `security`

Labeling rule:

1. Choose at least one primary type label from the mapping above.
2. If multiple types are involved, apply multiple labels (max 3, most impactful first).

Recommended issue body structure for major changes:

1. `Goal`: Why this change is needed.
2. `Planned Scope`: Main modules/files and rough implementation plan.
3. `Out of Scope`: What is intentionally excluded in this round.

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
