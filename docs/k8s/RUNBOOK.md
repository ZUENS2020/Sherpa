# Sherpa K8s 运行手册（SHE-59）

## 1. 目标

用于迁移后日常运维，覆盖：

1. 故障排查
2. 日志定位
3. 任务重试/恢复
4. 回滚操作

## 2. 快速诊断

```bash
kubectl -n sherpa get pods
kubectl -n sherpa get svc
kubectl -n sherpa get ingress
```

健康检查：

```bash
kubectl -n sherpa port-forward svc/sherpa-web 8001:8001
curl -sS http://127.0.0.1:8001/api/health
curl -sS http://127.0.0.1:8001/api/system | jq
curl -sS http://127.0.0.1:8001/api/metrics | head
```

## 3. 日志定位

Web 日志：

```bash
kubectl -n sherpa logs deploy/sherpa-web --tail=200
```

Frontend 日志：

```bash
kubectl -n sherpa logs deploy/sherpa-frontend --tail=200
```

Postgres 日志：

```bash
kubectl -n sherpa logs statefulset/postgres --tail=200
```

任务日志（按 job id）：

```bash
curl -sS http://127.0.0.1:8001/api/task/<job_id> | jq '.log'
```

## 4. 常见问题排查

1. `DATABASE_URL is required`
- 原因：Postgres-only 模式下缺失数据库连接串。
- 处理：检查 `Secret/sherpa-postgres` 与 `ConfigMap/sherpa-config` 注入是否完整。

2. `job` 长时间 `running`
- 检查：
  - `/api/task/<job_id>` 的 `phase`、`error_code`、`children_status`
  - `sherpa-web` 日志中对应 job 流
- 处理：必要时调用 stop 或 resume（见下一节）。

3. API/页面不可达
- 检查 Ingress 与 Service：
```bash
kubectl -n sherpa get ingress sherpa -o wide
kubectl -n sherpa get svc sherpa-web sherpa-frontend
```

## 5. 任务重试与恢复

手动恢复：

```bash
curl -sS -X POST http://127.0.0.1:8001/api/task/<job_id>/resume
```

手动停止：

```bash
curl -sS -X POST http://127.0.0.1:8001/api/task/<job_id>/stop
```

说明：

1. 默认关闭自动恢复（`SHERPA_WEB_AUTO_RESUME_ON_START=0`）。
2. 若缺失恢复上下文，任务会进入 `resume_failed` 并给出结构化错误码。

## 6. 回滚流程

应用回滚（Deployment）：

```bash
kubectl -n sherpa rollout undo deploy/sherpa-web
kubectl -n sherpa rollout undo deploy/sherpa-frontend
```

数据库回滚：

1. 建议策略：逻辑备份 + PV 快照。
2. 逻辑备份示例：
```bash
kubectl -n sherpa exec -it statefulset/postgres -- \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > sherpa_backup.sql
```
3. 恢复示例：
```bash
kubectl -n sherpa exec -i statefulset/postgres -- \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < sherpa_backup.sql
```

## 7. 告警骨架（当前）

当前先以 `/api/metrics` 为基础，建议最少接入以下告警：

1. 失败率告警：`sherpa_jobs_failure_rate_window`
2. 积压告警：`sherpa_jobs_status{status="queued"}`
3. 运行堆积告警：`sherpa_jobs_status{status="running"}`

## 8. 值班建议

1. 先看 `/api/system` 和 `/api/metrics` 判断范围。
2. 再定位 `sherpa-web` 日志与具体 job 日志。
3. 若单任务问题，优先 stop/resume，而非重启整个服务。
