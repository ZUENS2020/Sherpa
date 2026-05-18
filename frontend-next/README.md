# TianHeng Frontend

TianHeng 前端，基于 Next.js 14 + MUI + TanStack Query。

## 启动

```bash
npm install
npm run dev    # 端口 3000
```

## 页面

| 路径 | 功能 |
|---|---|
| `/` | 监控台——任务列表、状态总览、漏洞候选统计 |
| `/run` | 任务发起——填写仓库 URL 与预算后提交 |

## 架构

- 通过 `/api` 前缀与后端 FastAPI 通信（nginx gateway 代理）
- 使用 TanStack React Query 进行 2-3s 轮询刷新
- Zod schema 校验请求/响应
- Zustand 存储客户端 UI 状态
- `NEXT_PUBLIC_API_TARGET` 环境变量可配置本地开发时的后端代理目标

## 与后端的接口

| 端点 | 用途 |
|------|------|
| `GET /api/config` | 读取配置 |
| `PUT /api/config` | 更新配置 |
| `GET /api/system` | 系统概览 |
| `GET /api/tasks` | 任务列表 |
| `POST /api/task` | 提交新任务 |
| `GET /api/task/:id` | 任务详情 |
| `POST /api/task/:id/stop` | 停止任务 |
