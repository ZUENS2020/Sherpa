# 标准修改流程

本文档描述 Sherpa 当前推荐的修改、验证、合并与部署流程。

## 分支策略

- 功能开发：`codex/*` 或人工特性分支
- 集成验证：`dev`
- 生产主线：`main`

## 推荐流程

1. 在功能分支开发与验证
2. 提 PR 到 `dev`
3. `dev` 部署后跑真实仓库任务验证
4. 验证通过后再合并或继续补丁
5. 生产发布走 `dev -> main`

## 变更前最低检查

### 代码变更

- `python -m py_compile` 覆盖改动模块
- 对应 pytest 子集通过

### 前端变更

- `npm test`
- `npm run build`

### 工作流变更

至少做一条真实仓库验证，优先：

- `libyaml`
- `fmt`
- `zlib`
- `libarchive`

## 文档要求

任何改变下列行为的修改，都要同步文档：

- stage 路由
- `targets.json` schema
- seed 生成规则
- build / repair / crash-triage 规则
- k8s 运行时行为
- 部署方式
- API 契约，尤其是前端直接消费的 `/api/system`、`/api/tasks`、`/api/task/{id}`

如果这类修改会影响技术学习材料，还需要同步 `docs/TECHNICAL_DEEP_DIVE.md`。

## 不推荐做法

- 只改 README，不改运维文档
- 只看 stage 成功，不检查 `run_summary.json`
- 让 `replan` 无实质变化仍继续空转
- 在 `k8s_job` 路径依赖 Docker CLI
