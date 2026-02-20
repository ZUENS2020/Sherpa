# SHERPA — 批量化 Fuzz 自动化平台

SHERPA 是一个面向批量 GitHub 项目的自动化模糊测试平台。启动服务后只需通过 REST API 发起任务并查询状态，所有初始化、构建、运行与报告产出由服务自动完成。

核心能力：
- OpenCode 生成 harness、build 脚本、种子与分析报告
- LangGraph 工作流自动修复与重试（plan → synthesize → build → run → fix）
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
- 工作流引擎（LangGraph）：plan → synthesize → build → run → fix 循环
- OpenCode CLI：代码分析与生成
- Docker 运行层：容器内构建与 fuzz
- 持久化配置：`config/web_config.json` 与 `config/web_opencode.env`

数据流：
- `/api/task` → 初始化（可选）→ 批量 job 入队 → 工作线程并行执行 → `/api/task/{id}` 轮询

---

## 工作流说明

每个仓库的默认流程：
1. Plan：生成 `fuzz/PLAN.md` 与 `fuzz/targets.json`
2. Synthesize：生成 harness、`fuzz/build.py` 与 corpus
3. Build：执行 `python fuzz/build.py`，失败自动修复重试
4. Run：执行 fuzz 并产出崩溃样本与复现线索
5. Fix（可选）：若 crash 产生，尝试修复并生成补丁
6. Summary：写入 `run_summary.md/json`

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
- Web UI 用于人工查看与配置
- 批量调用建议使用 REST API
- 默认会挂载本机 `./output` 作为产物目录

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
      "log_file": "/app/job-logs/jobs/<id>.log"
    }
  ]
}
```

其他可选接口：
- `GET /api/config` / `PUT /api/config`
- `GET /api/system`

---

## 输出目录与产物

### 默认输出路径
- Docker Compose：本机 `./output` → 容器 `/shared/output`
- 每次任务会创建独立子目录：`<repo>-<8位id>/`

### 典型产物
- `fuzz/`：生成的 harness 与 build 脚本
- `crash_info.md` / `crash_analysis.md`
- `reproduce.py`
- `fix.patch` / `fix_summary.md`（若触发修复）
- `run_summary.md` / `run_summary.json`
- `challenge_bundle*/` 或 `false_positive*/` 或 `unreproducible*/`

### 日志
- 主日志：`/app/job-logs/jobs/<job_id>.log`
- 按等级拆分：`<job_id>.level.info.log` / `level.warn.log` / `level.error.log`
- 按类别拆分：`<job_id>.cat.workflow.log` / `cat.build.log` / `cat.opencode.log` / `cat.docker.log` / ...

### 自定义输出路径
- 设置环境变量 `SHERPA_OUTPUT_DIR`（Web 与 CLI 都会生效）

---

## 配置说明

常用环境变量：
- `OPENAI_API_KEY`：OpenCode 使用的 Key
- `OPENAI_BASE_URL`：OpenAI-compatible 端点（可选）
- `SHERPA_WEB_MAX_WORKERS`：并发 worker 数量（默认 5）
- `SHERPA_DEFAULT_OSS_FUZZ_DIR`：OSS-Fuzz 根目录（Docker Compose 默认 `/shared/oss-fuzz`）
- `SHERPA_OSS_FUZZ_REPO_URL`：OSS-Fuzz 仓库地址（默认 `https://gitclone.com/github.com/google/oss-fuzz`）
- `SHERPA_OUTPUT_DIR`：产物输出目录（默认 `/shared/output`）
- `SHERPA_GIT_MIRRORS`：Git 镜像列表（默认 `https://gitclone.com/github.com/`）

运行时配置：
- `config/` 目录仅运行时生成，不应打包
- 模板文件位于 `config.example/`

---

## 国内网络加速（APT 源）

所有 Dockerfile 默认使用国内镜像源（清华 TUNA）：
- Ubuntu：`https://mirrors.tuna.tsinghua.edu.cn/ubuntu`
- Debian：`https://mirrors.tuna.tsinghua.edu.cn/debian`

可通过 build args 覆盖：
```bash
docker build --build-arg APT_MIRROR=... -f docker/Dockerfile.fuzz-cpp .
```

### Node / npm / pip 国内源
- Node.js：默认从 `https://npmmirror.com/mirrors/node` 下载
- npm registry：`https://registry.npmmirror.com`
- pip index：`https://pypi.tuna.tsinghua.edu.cn/simple`

如需覆盖，可在 Docker 构建时传入：
```bash
docker build \
  --build-arg NODE_MIRROR=... \
  --build-arg NODE_VERSION=... \
  -f docker/Dockerfile.web .
```

### OpenCode 禁止执行命令（只改文件）
默认启用：`SHERPA_OPENCODE_NO_EXEC=1`  
只允许只读命令（rg/ls/cat/find/sed），禁止构建/运行/fuzz。

### Web UI 与 OpenCode 分离
默认使用独立的 OpenCode 容器：
- Web 容器不再安装 OpenCode CLI
- OpenCode 在 `sherpa-opencode` 镜像中运行
- 通过 `SHERPA_OPENCODE_DOCKER_IMAGE=sherpa-opencode:latest` 指定镜像
 - 若镜像不存在，默认会自动构建（`SHERPA_OPENCODE_AUTO_BUILD=1`）

---

## 打包前清理建议

---

## 运维改进参考

- OpenCode 产出稳定性提升计划：`harness_generator/docs/opencode_stability_plan.md`

1. 删除运行时 `config/` 目录
2. 不提交 `.env` 或任何密钥文件
3. 保留 `config.example/` 与 `.env.example` 作为模板
