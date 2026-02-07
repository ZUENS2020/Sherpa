# Windows 一键使用（Codex CLI）

## 0) 前置条件
- 已安装 **Python 3.10+**
- 已安装并启动 **Docker Desktop**（推荐：把 fuzz 放进 Docker 跑）
- 已安装并可在终端运行 **Codex CLI**：`codex`
- 已准备一个用于存放 **oss-fuzz checkout** 的目录（用于 `infra/helper.py` 构建/运行；可为空，系统会自动 clone）

> 说明：当你启用 Docker 模式（`docker_image=auto`）时，工具会把 `git clone` / `git diff` 也放进 Docker 跑，
> 因此 Windows 上通常**不需要额外安装 Git**。

> 说明：本项目会用 Codex 在本机工作区生成/修改 `fuzz/` 目录下的脚手架；
> 构建与运行 fuzz 建议放进 Linux Docker，降低 Windows 兼容性门槛。

## 1) 安装依赖
在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果你只用 `harness_generator/requirements.txt` 也可以：

```powershell
pip install -r harness_generator\requirements.txt
```

## 2) fuzz Docker 镜像（按语言自动选择/自动构建）

当前 `docker_image=auto` 仅支持两类项目：
- **C/C++**（libFuzzer）
- **Java**（Jazzer）

如果目标仓库不是 C/C++/Java（例如纯 Python/JS 项目），会在 clone 完成后报错提示“无法自动选择工具链”。

当你在 Web 页面勾选“在 Docker 里构建/运行 fuzz”并将镜像名填写为 `auto` 时，工具会：
- clone 目标仓库后自动判断语言（Java / C/C++）
- 选择对应镜像（默认：`sherpa-fuzz-java:latest` 或 `sherpa-fuzz-cpp:latest`）
- 镜像不存在则自动触发 `docker build`

如需手动构建（可选）：

```powershell
docker build -t sherpa-fuzz-cpp:latest -f .\docker\Dockerfile.fuzz-cpp .
docker build -t sherpa-fuzz-java:latest -f .\docker\Dockerfile.fuzz-java .
```

也可以用脚本（会自动 build + run）：

```powershell
.\docker\run_non_oss_fuzz.ps1 -Repo https://github.com/syoyo/tinyexr.git -Image auto
```

## 3) 启动 Web（前后端一体）

```powershell
.\.venv\Scripts\Activate.ps1
python .\harness_generator\src\langchain_agent\main.py
```

然后打开：
- http://127.0.0.1:8000/

页面里包含：
- 一键提交 fuzz 任务（`/fuzz_code`）
- 轮询显示任务状态与日志（`/api/fuzz/{job_id}`）

说明：当前 Web 流程会对你输入的任意 Git 仓库 URL **自动生成一个 OSS-Fuzz project**（写入你的 oss-fuzz checkout 下 `projects/<auto_name>/`），然后调用 `infra/helper.py build_image/build_fuzzers/run_fuzzer` 跑起来。

## 4) 配置（可选）
- 推荐方式：复制 `.env.example` 为 `.env` 并填写 `OPENAI_API_KEY`。
- Web 配置会持久化到 `config/`（运行时生成）。发布前请使用 `config.example/` 作为模板，不要提交真实配置/密钥。
- OSS-Fuzz（必配）：
  - 在 Web 配置里填写 `oss_fuzz_dir` 指向你的 oss-fuzz 根目录（必须包含 `infra/helper.py`）
  - 如果该目录不存在或为空，系统会自动从 `https://github.com/google/oss-fuzz` clone（可用 `SHERPA_OSS_FUZZ_REPO_URL` 覆盖）。
- OpenRouter（推荐）：
  - `OPENROUTER_API_KEY=...`
  - （可选）`OPENROUTER_MODEL=anthropic/claude-3.5-sonnet`
  - （可选）`OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- GitHub 克隆镜像（可选，国内网络常用）：
  - `SHERPA_GITHUB_MIRROR=https://gitclone.com/github.com/`
  - 或更通用的列表：`SHERPA_GIT_MIRRORS=https://gitclone.com/github.com/,https://ghproxy.com/{url}`
    - 说明：`{url}` 会被替换为原始仓库 URL（例如 `https://github.com/owner/repo.git`）
  - 说明：如果你不配置任何镜像，本项目也会在 `https://github.com/...` 失败后自动尝试一些常见镜像。

> 额外说明：如果你本机 `git config --global http.proxy/https.proxy` 指向 `127.0.0.1/localhost` 但代理程序没开，会导致 host git clone 直接失败。
> 工具会尽量自动检测并在 host 克隆时临时禁用该代理；你也可以手动强制禁用：`SHERPA_GIT_DISABLE_PROXY=1`。

- Docker 内透传代理（可选，适用于“本机开了代理能访问，但容器里 git clone 失败”）：
  - 直接设置（对本机与工具都生效）：
    - `HTTP_PROXY=http://127.0.0.1:7891`
    - `HTTPS_PROXY=http://127.0.0.1:7891`
    - （可选）`NO_PROXY=127.0.0.1,localhost,::1`
  - 仅用于本项目（优先级更高）：
    - `SHERPA_DOCKER_HTTP_PROXY=http://127.0.0.1:7891`
    - `SHERPA_DOCKER_HTTPS_PROXY=http://127.0.0.1:7891`
    - （可选）`SHERPA_DOCKER_NO_PROXY=127.0.0.1,localhost,::1`
  - 说明：若代理地址包含 `127.0.0.1` / `localhost`，会自动改写为 `host.docker.internal` 供容器访问。
    - 如需自定义该 host 名称：`SHERPA_DOCKER_PROXY_HOST=host.docker.internal`
- Codex/OpenAI Key 文件（可选）：
  - `.env` 中写 `OPENAI_API_KEY=...`
  - 或设置 `AI_KEY_PATH` 指向你的 `.env`
