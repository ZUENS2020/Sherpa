# SHERPA — 批量化 Fuzz 自动化平台

SHERPA 是一个面向批量 GitHub 项目的自动化模糊测试平台。启动服务后只需通过 REST API 发起任务并查询状态，所有初始化与构建由服务自动完成。

核心能力：
- OpenCode 生成 harness、build 脚本和种子
- 内置工作流循环自动修复与重试
- 批量提交、并行执行与统一状态监控
- 运行日志落盘，Web 仅展示摘要进度

---

## 项目信息

定位：无需人工执行命令的“批量 fuzz 编排服务”。

适用场景：
- 批量测试多个 GitHub 项目
- 由外部系统统一调度与监控
- 多机/多项目流水线集成

---

## 项目架构

核心组件：
- Web API（FastAPI）：任务提交与状态查询
- 工作流引擎（LangGraph）：plan → synthesize → build → run 循环
- OpenCode CLI：代码分析与生成
- Docker 运行层：容器内构建与 fuzz
- 持久化配置：`config/web_config.json` 与 `config/web_opencode.env`

数据流：
- `/api/task` → 初始化（可选）→ 批量 job 入队 → 工作线程并行执行 → `/api/task/{id}` 轮询

---

## 实现方式

每个仓库的工作流：
1. Plan：生成 `fuzz/PLAN.md` 与 `fuzz/targets.json`
2. Synthesize：生成 harness、`fuzz/build.py` 与 corpus
3. Build：执行 `python fuzz/build.py`，失败自动修复重试
4. Run：执行 fuzz 并产出崩溃样本与复现线索

运行环境：
- C/C++ 使用 libFuzzer
- Java 使用 Jazzer

---

## 部署方法

### Docker Compose（推荐，Ubuntu/Windows 通用）

1. 准备配置
```bash
cp .env.example .env
```

2. 启动服务
```bash
docker compose up -d --build
```

3. 打开 Web UI（可选）
```
http://localhost:8000/
```

说明：
- Web UI 仅用于人工查看与配置
- 批量调用建议使用 REST API

### 本地运行（可选）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r harness_generator/requirements.txt
python ./harness_generator/src/langchain_agent/main.py
```

---

## API 信息（集成化）

### 1) 启动任务
`POST /api/task`

请求示例：
```json
{
  "jobs": [
    { "code_url": "https://github.com/madler/zlib.git", "time_budget": 900, "docker": true, "docker_image": "auto" },
    { "code_url": "https://github.com/libexpat/libexpat.git", "time_budget": 900, "docker": true, "docker_image": "auto" }
  ]
}
```

返回示例：
```json
{ "job_id": "<task_id>", "status": "queued" }
```

自动行为：
- OSS-Fuzz checkout 自动初始化
- fuzz 镜像自动构建
- 批量 job 自动并行执行

### 2) 查询任务状态
`GET /api/task/{job_id}`

返回示例：
```json
{
  "job_id": "<task_id>",
  "status": "running",
  "children_status": {
    "total": 2,
    "queued": 0,
    "running": 1,
    "success": 1,
    "error": 0
  },
  "children": [
    {
      "job_id": "<fuzz_job_id>",
      "status": "running",
      "repo": "https://github.com/madler/zlib.git",
      "log_file": "/app/config/logs/jobs/<id>.log"
    }
  ]
}
```

其他可选接口：
- `GET /api/config` / `PUT /api/config`
- `GET /api/system`

---

## 配置说明

常用环境变量：
- `OPENAI_API_KEY`：OpenCode 使用的 Key
- `OPENAI_BASE_URL`：OpenAI-compatible 端点（可选）
- `SHERPA_WEB_MAX_WORKERS`：并发 worker 数量（默认 5）
- `SHERPA_DEFAULT_OSS_FUZZ_DIR`：OSS-Fuzz 根目录（Docker Compose 默认 `/shared/oss-fuzz`）

运行时配置：
- `config/` 目录仅运行时生成，不应打包
- 模板文件位于 `config.example/`

---

## 日志与产物

日志位置：
- `config/logs/jobs/<job_id>.log`

Docker Compose 默认挂载：
- OSS-Fuzz：`/shared/oss-fuzz`
- 临时工作区：`/shared/tmp`

---

## 打包前清理建议

1. 删除运行时 `config/` 目录
2. 不提交 `.env` 或任何密钥文件
3. 保留 `config.example/` 与 `.env.example` 作为模板
