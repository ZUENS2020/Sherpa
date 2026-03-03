# Kubernetes 部署说明（SHE-52）

## 1. 前置条件
- 已安装并配置 `kubectl`
- 集群已安装 Ingress Controller（示例按 `ingressClassName: nginx`）
- 已构建并推送镜像：
  - `sherpa-web:latest`
  - `sherpa-frontend:latest`
- 集群节点可提供 Docker socket（默认 `/var/run/docker.sock`，用于 `k8s_job` worker 执行现有 Docker 构建链路）

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
```

## 5. 回滚

```bash
kubectl -n sherpa rollout undo deploy/sherpa-web
kubectl -n sherpa rollout undo deploy/sherpa-frontend
kubectl -n sherpa rollout undo deploy/sherpa-gateway
```

## 6. 运维注意事项
- `DATABASE_URL` 缺失时，`sherpa-web` 将启动失败（Postgres-only 设计）。
- 启用去 DinD 执行链路时，设置 `SHERPA_EXECUTOR_MODE=k8s_job`，并保证 `sherpa-web` Pod 有创建/查询 Job 的 RBAC 权限。
- 若节点 Docker socket 路径不是 `/var/run/docker.sock`，请在 ConfigMap 中同步修改 `SHERPA_K8S_DOCKER_SOCKET_PATH`。
- 建议给 `postgres` 配置备份策略（逻辑备份 + PV 快照）。
- `sherpa-shared-output` 建议配合 TTL 清理策略（CronJob）避免无限增长。

## 7. 本机测试（Python 3.14）
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
