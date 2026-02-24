## 同步策略（固定）

- 每天仅保留一个“项目现状”条目。
- 当天有新进展时，直接覆盖正文，不追加评论。

## 同步时间

- {{DATE}}

## 当前代码基线

- 仓库：`{{REPO_URL}}`
- 当前工作分支：`{{CURRENT_BRANCH}}`
- 工作区状态：{{WORKTREE_STATUS}}

## 双分支进度（分别说明）

### 分支 A：`main`

- 最新提交：`{{MAIN_HEAD}}`
- 远端同步状态：`{{MAIN_REMOTE_SYNC}}`
- 相对 `{{CURRENT_BRANCH}}` 的独有提交：
{{MAIN_UNIQUE_COMMITS}}

### 分支 B：`{{CURRENT_BRANCH}}`

- 最新提交：`{{CURRENT_HEAD}}`
- 远端同步状态：`{{CURRENT_REMOTE_SYNC}}`
- 相对 `main` 的独有提交：
{{CURRENT_UNIQUE_COMMITS}}

### 两分支关系结论

- 分叉计数：`main...{{CURRENT_BRANCH}} = {{DIVERGENCE_COUNT}}`
- 合并判断：{{MERGEABILITY_NOTE}}

## 今日关键进展（{{DATE}}，累计到当前时点）

{{MAJOR_CHANGES_CUMULATIVE}}

## 验证结果

- 执行：`{{VALIDATION_COMMAND}}`
- 结果：`{{VALIDATION_RESULT}}`

## 当前风险与待办

{{RISKS_AND_TODOS}}

## 下一步建议

{{NEXT_STEPS}}
