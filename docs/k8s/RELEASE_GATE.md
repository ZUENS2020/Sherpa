# 发布门禁

## 必过项

### 后端
- `python -m py_compile` 覆盖改动模块
- 关键 pytest 子集通过

### 前端
- `npm test`
- `npm run build`

### 部署前核对
- k8s worker 不依赖 Docker CLI
- `run_summary.json` 字段与当前实现一致
- 文档已同步更新

## 建议 smoke test

至少验证：
- 一条 parser 类仓库：`libyaml` 或 `fmt`
- 一条 build/fix_build 易出问题仓库：`zlib` 或 `libarchive`

## 拒绝发布条件

- `run` 仍然无限空转到 dispatch limit
- `replan` 无实质变化仍被视为成功
- `repo_examples` 继续大量吸收源码文件
- k8s worker 回退到 inner Docker 执行
