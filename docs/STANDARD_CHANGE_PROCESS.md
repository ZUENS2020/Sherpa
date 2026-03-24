# 标准变更流程

本文档描述 Sherpa 当前推荐的变更、验证与发布流程。

## 1. 分支模型

- 功能分支：`codex/*` 或明确主题分支
- 集成验证分支：`dev`
- 生产发布分支：`main`

推荐路径：

1. 在功能分支上开发
2. 提交 PR 到 `dev`
3. 在 `dev` 上验证
4. 再创建 `dev -> main` PR
5. 通过 `main` 进入生产发布

## 2. Major Change 约束

对于较大的项目变更：

1. 先定义目标和范围
2. 先写 Linear issue
3. 添加正确类型标签
4. 将 issue 设为 `In Progress`
5. 等待确认后再实施
6. 实施期间持续补充同一 issue 的中文进展
7. 完成后设为 `Done`，并加一条中文总结评论

## 3. 验证要求

### 代码变更

- 对修改过的 Python 模块运行 `python3 -m py_compile`
- 运行相关 pytest 子集

### 前端变更

- 运行前端构建和必要测试
- 确认前端字段和后端 API 真实契约一致

### 工作流变更

优先在 `dev` 上做一次真实仓库验证，常见样例包括：

- `fmt`
- `libyaml`
- `zlib`
- `libarchive`

## 4. 文档同步规则

只要变更影响以下内容，就必须同步更新文档：

- 工作流阶段或路由
- 目标选择或 `execution_plan`
- 种子生成或种子质量策略
- 崩溃分诊 / 复现 / 分析行为
- 部署模型
- 前端依赖的 API 契约

至少应复核：

- [`../README.md`](../README.md)
- [`API_REFERENCE.md`](API_REFERENCE.md)
- [`TECHNICAL_DEEP_DIVE.md`](TECHNICAL_DEEP_DIVE.md)
- [`k8s/`](k8s/)

## 5. 发布保护

- 不要直接 push 到 `dev` 或 `main`
- `main` 只接受来自 `dev` 的变更
- PR 必须包含：
  - 变更摘要
  - 风险与回滚点
  - 可复现验证结果

## 6. 不要这样做

- 只更新 README，却让深层文档过期
- 写理想化 API，而不是当前真实实现
- 只看阶段状态，不看实际产物
- 让历史迁移笔记伪装成当前操作手册
