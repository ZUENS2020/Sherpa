# Kubernetes 部署说明（SHE-52）

## 1. 前置条件
- 已安装并配置 `kubectl`
- 集群已安装 Ingress Controller（示例按 `ingressClassName: nginx`）
- 已构建并推送镜像：
  - `sherpa-web:latest`
  - `sherpa-frontend:latest`

## 2. 准备 Secret
从示例复制并填值：

```bash
cp k8s/base/minimax-secret.example.yaml k8s/base/minimax-secret.yaml
cp k8s/base/postgres-secret.example.yaml k8s/base/postgres-secret.yaml
```

然后把 `k8s/base/kustomization.yaml` 中的 `*.example.yaml` 改为真实 `*.yaml`。

## 3. 部署

```bash
kubectl apply -k k8s/base
```

## 4. 验证

```bash
kubectl -n sherpa get pods
kubectl -n sherpa get svc
kubectl -n sherpa get ingress
```

健康检查：

```bash
kubectl -n sherpa port-forward svc/sherpa-web 8001:8001
curl -sS http://127.0.0.1:8001/api/health
curl -sS http://127.0.0.1:8001/api/metrics | head
```

## 5. 回滚

```bash
kubectl -n sherpa rollout undo deploy/sherpa-web
kubectl -n sherpa rollout undo deploy/sherpa-frontend
```

## 6. 运维注意事项
- `DATABASE_URL` 缺失时，`sherpa-web` 将启动失败（Postgres-only 设计）。
- 执行器当前仅支持 `k8s_job`，并要求 `sherpa-web` Pod 具备创建/查询 Job 的 RBAC 权限。
- 建议给 `postgres` 配置备份策略（逻辑备份 + PV 快照）。
- `sherpa-shared-output` 建议配合 TTL 清理策略（CronJob）避免无限增长。

## 7. CI/CD（GitHub Actions）

- 工作流文件：`.github/workflows/k8s-build-deploy.yml`
- `push main` 自动构建并推送镜像到 GHCR：
  - `sherpa-web`
  - `sherpa-frontend`
  - `sherpa-fuzz-cpp`
- 手动触发 `workflow_dispatch` 且 `deploy=true` 时执行 `kubectl apply -k k8s/base`。
- 部署前需在仓库 Secrets 配置 `KUBE_CONFIG_DATA`（base64 kubeconfig）。

## 8. 本机测试（Python 3.14）
建议统一使用 Python 3.14 虚拟环境运行测试：

```bash
python3.14 -m venv .venv314
source .venv314/bin/activate
python -m pip install -U pip
python -m pip install -r harness_generator/requirements.txt -r docker/requirements.web.txt pytest
```

`tests/test_api_stability.py` 依赖本地 Postgres（`127.0.0.1:55432`），可临时启动：

```bash
docker run -d --name sherpa-test-pg \
  -e POSTGRES_DB=sherpa \
  -e POSTGRES_USER=sherpa \
  -e POSTGRES_PASSWORD=sherpa \
  -p 55432:5432 \
  m.daocloud.io/docker.io/library/postgres:16
```

执行回归：

```bash
python -m pytest -q tests/test_job_store_persistence.py tests/test_api_stability.py
```

## 9. 运维与收口文档

- 运行手册：`docs/k8s/RUNBOOK.md`
- 发布回滚门禁：`docs/k8s/RELEASE_GATE.md`
- E2E 验收报告：`docs/k8s/E2E_ZLIB_REPORT.md`
