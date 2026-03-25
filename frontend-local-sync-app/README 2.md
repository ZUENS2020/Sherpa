# Sherpa 本地联调前端

这是给 Sherpa 后端做本地联调用的前端应用。它的目标是读取后端动态 API，而不是依赖写死数据。

## 依赖的核心 API

- `GET /api/system`
- `GET /api/tasks`
- `GET /api/task/{job_id}`
- `POST /api/task`
- `PUT /api/config`

## 本地运行

1. 安装依赖
   `npm install`
2. 启动开发服务
   `npm run dev`
3. 在设置里把 API Base URL 指向你的 Sherpa 后端，例如：
   `https://dev.zuens2020.work`

## 联调约定

- 所有任务列表、系统指标和详情页都应来自后端 API
- 不要把任务状态、仪表盘数字或仓库名写死在前端
- 如果后端缺字段，优先补后端契约和文档，而不是继续前端硬编码
