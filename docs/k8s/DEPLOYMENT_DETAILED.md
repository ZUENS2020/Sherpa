# Sherpa K8s 详细部署手册

本文档用于生产/预发/本地 K8s 的统一部署，面向熟悉 Docker 但不一定熟悉 K8s 的同学。

## 1. 目标与范围

目标：在 K8s 上稳定运行 Sherpa，具备以下能力：

1. 前端与 API 正常访问（Ingress + 可选 Cloudflare Tunnel）。
2. 任务可从 `plan -> synthesize -> build -> run` 全流程执行。
3. 状态持久化使用 Postgres（唯一持久化后端）。
4. 任务排障可通过 `k8s_job_names` 精确定位 stage 日志。

不在本手册范围：

1. 业务功能设计与 Prompt 策略调优。
2. 云厂商专有负载均衡/安全组细节。

---

## 2. 架构总览

```mermaid
flowchart LR
  U["User/Client"] --> IN["K8s Ingress"]
  IN --> FE["sherpa-frontend Deployment"]
  IN --> API["sherpa-web Deployment (/api/*)"]
  API --> DB[("Postgres StatefulSet")]
  API --> SH[("PVC: /shared/output")]
  API --> OF[("PVC: /shared/oss-fuzz")]
  API --> J1["Stage Job: plan"]
  API --> J2["Stage Job: synthesize"]
  API --> J3["Stage Job: build"]
  API --> J4["Stage Job: run"]
  J1 --> SH
  J2 --> SH
  J3 --> SH
  J4 --> SH
```

说明：

1. 主控服务是 `sherpa-web`，负责创建/追踪 stage jobs。
2. 每个 stage 是独立 Job，减少长流程单容器污染。
3. `shared-output` PVC 保存任务产物和 stage 结果文件。
4. `oss-fuzz` PVC 提供固定工具链目录（校验 `infra/helper.py`）。

---

## 3. 前置条件

### 3.1 基础环境

1. 可用 Kubernetes 集群（Docker Desktop K8s / kind / minikube / 云上 K8s 均可）。
2. `kubectl` 可访问目标集群。
3. 可拉取基础镜像（若网络受限，请提前导入镜像）。
4. DNS 可用（若使用公网域名接入）。

### 3.2 必填 Secret

至少准备以下 Secret 值：

1. `MINIMAX_API_KEY`
2. `DATABASE_URL`（Postgres 连接串）

可用如下方式创建（示例）：

```bash
kubectl -n sherpa create secret generic sherpa-secrets \
  --from-literal=MINIMAX_API_KEY='your_key' \
  --from-literal=DATABASE_URL='postgresql://sherpa:sherpa@postgres:5432/sherpa' \
  --dry-run=client -o yaml | kubectl apply -f -
```

---

## 4. 部署步骤（标准流程）

### Step 1: 部署基础资源

```bash
kubectl apply -k k8s/base
```

### Step 2: 等待核心工作负载就绪

```bash
kubectl -n sherpa rollout status deploy/sherpa-web
kubectl -n sherpa rollout status deploy/sherpa-frontend
kubectl -n sherpa rollout status statefulset/postgres
```

### Step 3: 检查 Ingress 与路由

```bash
kubectl -n sherpa get ingress
kubectl -n sherpa get svc
```

路由期望：

1. `/` -> `sherpa-frontend`
2. `/api/*` -> `sherpa-web`

### Step 4: 验证 API 健康

```bash
kubectl -n sherpa port-forward deploy/sherpa-web 8001:8001
curl -sS http://127.0.0.1:8001/api/system
```

---

## 5. Cloudflare Tunnel（可选公网入口）

```mermaid
flowchart LR
  I["Internet"] --> CF["Cloudflare Edge"]
  CF --> T["cloudflared Pod"]
  T --> IN["K8s Ingress"]
  IN --> FE["sherpa-frontend"]
  IN --> API["sherpa-web"]
```

执行：

1. 在 Cloudflare 控制台创建 Tunnel 与 Public Hostname（如 `sherpa.zuens2020.work`）。
2. 将 Tunnel token 写入 K8s Secret。
3. 应用 overlay：

```bash
kubectl apply -k k8s/overlays/cloudflare
kubectl -n sherpa get pods | grep cloudflared
```

常见错误：

1. `1033`: Tunnel 连接器不在线。
2. `1016`: DNS 或 Tunnel route 目标不可解析。
3. `404 (nginx)`: Host/Path 与 Ingress 规则不匹配。

---

## 6. 任务执行链路与产物

### 6.1 阶段流

```mermaid
sequenceDiagram
  participant C as Client
  participant W as sherpa-web
  participant P as plan Job
  participant S as synthesize Job
  participant B as build Job
  participant R as run Job
  participant O as shared-output PVC

  C->>W: POST /api/task
  W->>P: create job
  P->>O: stage-plan.json
  W->>S: create job
  S->>O: stage-synthesize.json
  W->>B: create job
  B->>O: stage-build.json
  W->>R: create job
  R->>O: stage-run.json
  W-->>C: task status/result
```

### 6.2 关键产物路径

1. `/shared/output/_k8s_jobs/<job_id>/stage-*.json`
2. `/shared/output/_k8s_jobs/<job_id>/stage-*.error.txt`
3. `/shared/output/<repo-*>/fuzz/PLAN.md`
4. `/shared/output/<repo-*>/fuzz/out/*`

---

## 7. 本轮部署中遇到的问题与修复项

### 7.1 oss-fuzz 拉取源不稳定

现象：初始化阶段可能在镜像源超时，导致任务卡在准备期。

修复：

1. 统一默认 `SHERPA_OSS_FUZZ_REPO_URL` 为 `https://github.com/google/oss-fuzz.git`。
2. 清理旧默认回退为第三方镜像源的逻辑。

### 7.2 节点 metrics 缺失日志误导

现象：日志出现 `node_ready_no_metrics:error:*`，但实际是可降级场景。

修复：

1. 语义改为 `node_ready_no_metrics_warn:*`。
2. 保持可调度（只要 Node Ready），避免误报阻断。

### 7.3 架构镜像不匹配

现象：amd64/arm64 混用导致组件异常。

修复建议：

1. 明确目标架构，统一导入对应镜像。
2. 对关键组件（Postgres/Ingress）执行 `image inspect` 校验。

---

## 8. 一键巡检清单（上线前）

```bash
# 命名空间资源
kubectl -n sherpa get pods,svc,ingress

# 核心日志
kubectl -n sherpa logs deploy/sherpa-web --tail=200
kubectl -n sherpa logs deploy/sherpa-frontend --tail=200
kubectl -n sherpa logs statefulset/postgres --tail=200

# PVC 状态
kubectl -n sherpa get pvc

# Secret 是否存在
kubectl -n sherpa get secret sherpa-secrets

# API 健康
curl -sS http://127.0.0.1:8001/api/system
```

判定标准：

1. 核心 Pod 全部 `Running/Ready`。
2. API 返回 200 且含系统配置字段。
3. 提交任务后 `k8s_job_names` 可追踪每个 stage。

---

## 9. 故障分层排查（建议顺序）

```mermaid
flowchart TD
  A["任务失败"] --> B["看 /api/task/{id}: phase + error + k8s_job_names"]
  B --> C{"问题层级"}
  C -->|"入口不可达"| D["Ingress/Cloudflare/DNS"]
  C -->|"作业没启动"| E["K8s Job/SA/RBAC/Node"]
  C -->|"初始化失败"| F["oss-fuzz checkout 与共享盘"]
  C -->|"构建失败"| G["编译链与依赖"]
  C -->|"运行卡住"| H["run_time_budget 与 fuzzer 行为"]
```

### 9.1 入口层（DNS/Ingress/Tunnel）

1. `kubectl -n sherpa get ingress`
2. `kubectl -n sherpa logs deploy/cloudflared --tail=200`（若启用）
3. Cloudflare 控制台检查 Tunnel connector 与 route。

### 9.2 调度层（Job/Node）

1. `kubectl -n sherpa get jobs,pods`
2. `kubectl -n sherpa describe job <stage-job-name>`
3. `kubectl -n sherpa logs job/<stage-job-name> --tail=300`

### 9.3 数据层（Postgres/PVC）

1. `kubectl -n sherpa get pvc`
2. `kubectl -n sherpa logs statefulset/postgres --tail=200`
3. 在 web 日志中查数据库连接错误关键词。

### 9.4 业务层（plan/synthesize/build/run）

1. 查看 `stage-*.error.txt`。
2. 对应 `k8s_job_names` 拉完整日志。
3. 对照 repo 产物目录判断是否“无产出/产出不合法/运行超时”。

---

## 10. 回滚与恢复

### 10.1 Deployment 回滚

```bash
kubectl -n sherpa rollout undo deploy/sherpa-web
kubectl -n sherpa rollout undo deploy/sherpa-frontend
```

### 10.2 Job 清理

```bash
kubectl -n sherpa delete job -l app=sherpa-fuzz-worker --ignore-not-found
```

### 10.3 共享目录修复（oss-fuzz）

当 `/shared/oss-fuzz` 内容损坏：

1. 删除无效目录内容。
2. 重新 clone 官方 `oss-fuzz`。
3. 校验 `/shared/oss-fuzz/infra/helper.py` 必须存在。

---

## 11. 运维建议（长期）

1. 保留一份可离线导入的关键镜像清单（按架构区分）。
2. 将 `k8s_job_names` + `stage error.txt` 纳入告警关联信息。
3. 对 Cloudflare Tunnel 增加“连接器离线”告警。
4. 每次升级后固定跑一次 zlib E2E 作为基线。

---

## 12. 相关文档

1. `/Users/zuens2020/Documents/Sherpa/README.md`
2. `/Users/zuens2020/Documents/Sherpa/docs/k8s/DEPLOY.md`
3. `/Users/zuens2020/Documents/Sherpa/docs/k8s/RUNBOOK.md`
4. `/Users/zuens2020/Documents/Sherpa/docs/k8s/CLOUDFLARE_TUNNEL.md`
5. `/Users/zuens2020/Documents/Sherpa/docs/k8s/E2E_ZLIB_REPORT.md`
