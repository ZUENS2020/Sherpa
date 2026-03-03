# 本地 Kubernetes 快速启动

本指南用于“本地也走 k8s”的最小闭环。

## 1. 前置条件

1. 本机已安装并可用 `kubectl`
2. 本机有可用 Kubernetes 集群（Docker Desktop K8s / kind / minikube）
3. 已配置可用镜像与密钥 Secret

## 2. 准备 Secret

```bash
cp k8s/base/minimax-secret.example.yaml k8s/base/minimax-secret.yaml
cp k8s/base/postgres-secret.example.yaml k8s/base/postgres-secret.yaml
```

将 `k8s/base/kustomization.yaml` 中 `*.example.yaml` 切换为真实 `*.yaml`。

## 3. 一键部署

```bash
kubectl apply -k k8s/base
kubectl -n sherpa get pods
```

## 4. 健康检查

```bash
kubectl -n sherpa port-forward svc/sherpa-web 8001:8001
curl -sS http://127.0.0.1:8001/api/health
curl -sS http://127.0.0.1:8001/api/system | jq
curl -sS http://127.0.0.1:8001/api/metrics | head
```

## 5. 提交任务（示例）

```bash
curl -sS -X POST http://127.0.0.1:8001/api/task \
  -H 'Content-Type: application/json' \
  -d '{
    "jobs": [{
      "code_url": "https://github.com/madler/zlib.git",
      "docker": true,
      "docker_image": "auto",
      "total_time_budget": 900,
      "run_time_budget": 900,
      "max_tokens": 1000
    }]
  }'
```

## 6. 任务查询

```bash
curl -sS http://127.0.0.1:8001/api/tasks?limit=20
curl -sS http://127.0.0.1:8001/api/task/<job_id>
```

## 7. 恢复与停止

```bash
curl -sS -X POST http://127.0.0.1:8001/api/task/<job_id>/resume
curl -sS -X POST http://127.0.0.1:8001/api/task/<job_id>/stop
```

## 8. 注意事项

1. 当前执行器仅支持 `k8s_job`。
2. `DATABASE_URL` 缺失时，`sherpa-web` 启动会失败。
3. 如需回滚，请参考 `docs/k8s/RUNBOOK.md` 与 `docs/k8s/RELEASE_GATE.md`。
