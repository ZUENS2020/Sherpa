# Docs 总览

本目录为 Sherpa 的对接与运维文档入口，口径统一为：

- Kubernetes-only
- Postgres-only
- Native Runtime（无 inner Docker）
- 多阶段多 Job 执行（`plan -> synthesize -> build -> run`）

## 文档索引

1. `/Users/zuens2020/Documents/Sherpa/docs/PROJECT_HANDOFF_STATUS.md`：当前进度与对接现状
2. `/Users/zuens2020/Documents/Sherpa/docs/DOCKER_TO_K8S_HANDOFF.md`：Docker 背景团队迁移对接
3. `/Users/zuens2020/Documents/Sherpa/docs/K8S_MIGRATION_CHECKLIST.md`：迁移里程碑与验收清单
4. `/Users/zuens2020/Documents/Sherpa/docs/k8s/LOCAL_K8S_QUICKSTART.md`：本地最小启动
5. `/Users/zuens2020/Documents/Sherpa/docs/k8s/DEPLOY.md`：部署说明（简版）
6. `/Users/zuens2020/Documents/Sherpa/docs/k8s/DEPLOYMENT_DETAILED.md`：部署说明（详细版，含故障树）
7. `/Users/zuens2020/Documents/Sherpa/docs/k8s/RUNBOOK.md`：运行手册
8. `/Users/zuens2020/Documents/Sherpa/docs/k8s/RELEASE_GATE.md`：发布门禁
9. `/Users/zuens2020/Documents/Sherpa/docs/k8s/CLOUDFLARE_TUNNEL.md`：Cloudflare Tunnel 接入
10. `/Users/zuens2020/Documents/Sherpa/docs/k8s/MAPPING.md`：Compose 到 K8s 映射
11. `/Users/zuens2020/Documents/Sherpa/docs/k8s/E2E_ZLIB_REPORT.md`：E2E 报告模板与样例

## 核心图谱

```mermaid
flowchart LR
  U["User"] --> G["Ingress/Gateway"]
  G --> FE["Frontend"]
  G --> API["sherpa-web"]
  API --> DB[("Postgres")]
  API --> J1["Job(plan)"]
  API --> J2["Job(synthesize)"]
  API --> J3["Job(build)"]
  API --> J4["Job(run)"]
```

## 字段口径

任务展示与排障固定关注：

1. `job_id`
2. `status`
3. `runtime_mode`
4. `phase`
5. `error_code`
6. `error_kind`
7. `error_signature`
8. `k8s_job_name` / `k8s_job_names`
9. `children_status`
