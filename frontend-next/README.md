# Sherpa for Competition

Sherpa 的比赛展示版本。前后端一体，使用 Next.js API Routes 模拟 Sherpa 后端 API，数据持久化到本地 JSON 文件，可在前端面板手动调整任意字段。

## 启动

```bash
npm install
npm run dev    # 默认端口 3001
```

首次启动自动从 `data/seed.json` 初始化 `data/runtime.json`。

## 页面

| 路径 | 功能 |
|---|---|
| `/` | 监控台——只显示统计数据与任务总览，隐藏技术细节 |
| `/run` | 发起任务——填写仓库 URL 与预算后提交 |
| `/admin` | 数据控制面板——手动调整任务状态、漏洞候选、覆盖率等所有字段 |

## 调数据流程

1. 访问 `/admin`，左侧选择任务，右侧修改字段
2. 点击"保存此任务"——立即写入 `data/runtime.json`
3. 监控台（`/`）自动轮询，2 秒内反映变化
4. 点击右上角"重置 Seed"可恢复初始演示数据

## 与真实 Sherpa 的差异

- API 路径完全一致（`/api/config`、`/api/system`、`/api/tasks`、`/api/task/:id`、`/api/task/:id/stop`）
- 后端改为 Next.js API Routes + 本地 JSON，无需 Python 环境
- 监控台移除了日志面板和 frontier/replay 技术细节，只保留任务状态与漏洞候选统计
- 新增 `/admin` 数据控制面板
