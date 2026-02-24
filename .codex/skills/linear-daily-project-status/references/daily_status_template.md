## 同步策略（固定）

- 每天仅保留一个“项目现状”条目。
- 当天有新进展时，直接覆盖正文，不追加评论。

## 同步时间

- {{DATE}}

## 当前代码基线

- 仓库：`{{REPO_URL}}`
- 当前工作分支：`{{CURRENT_BRANCH}}`
- 当前版本摘要：{{CURRENT_VERSION_SUMMARY}}
- 工作区状态：{{WORKTREE_STATUS}}

## 分支进度与差异（必填）

- `main` 远端同步：`{{MAIN_REMOTE_SYNC}}`
- `{{CURRENT_BRANCH}}` 远端同步：`{{CURRENT_BRANCH_REMOTE_SYNC}}`
- `main` 相对 `{{CURRENT_BRANCH}}` 的变更规模：{{MAIN_UNIQUE_SIZE}}
- `{{CURRENT_BRANCH}}` 相对 `main` 的变更规模：{{CURRENT_BRANCH_UNIQUE_SIZE}}
- 分叉计数：`main...{{CURRENT_BRANCH}} = {{DIVERGENCE_COUNT}}`
- 差异总结（高层）：{{BRANCH_DIFF_SUMMARY}}

## 项目检查结果（当前代码）

- 运行/配置检查：{{RUNTIME_CONFIG_CHECKS}}
- 工作流检查：{{WORKFLOW_CHECKS}}
- 文档/接口检查：{{DOC_API_CHECKS}}

## 当日完成工作清单（全量）

{{ALL_WORK_COMPLETED_SUMMARY}}

## 验证结果

- 执行：`{{VALIDATION_COMMAND}}`
- 结果：`{{VALIDATION_RESULT}}`

## 当前风险与待办

{{RISKS_AND_TODOS}}

## 下一步建议

{{NEXT_STEPS}}
