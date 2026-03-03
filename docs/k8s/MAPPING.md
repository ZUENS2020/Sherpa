# Compose 到 Kubernetes 映射（SHE-52 / M1）

## 服务映射
- `sherpa-web` -> `Deployment/sherpa-web` + `Service/sherpa-web`
- `sherpa-frontend` -> `Deployment/sherpa-frontend` + `Service/sherpa-frontend`
- `sherpa-job-store`（SQLite sidecar）-> 移除，改为 `StatefulSet/postgres` + `Service/postgres`
- `sherpa-oss-fuzz-init` -> 由 `sherpa-web` 启动流程按需初始化（后续可拆为 Job）
- `sherpa-docker`（DinD）-> 已从 k8s 运行链路移除（统一使用 `k8s_job`）

## 网络映射
- compose 默认网络 -> Kubernetes Service + Ingress
- 路由规则：
  - `/` -> `sherpa-frontend:3000`
  - `/api/*` -> `sherpa-web:8001`

## 存储映射
- `./output` -> `PVC/sherpa-shared-output`
- `sherpa-tmp` -> `PVC/sherpa-shared-tmp`
- `sherpa-config` -> `PVC/sherpa-config`
- `sherpa-job-logs` -> `PVC/sherpa-job-logs`
- `sherpa-oss-fuzz` -> `PVC/sherpa-oss-fuzz`
- SQLite 文件卷 -> `StatefulSet/postgres` 自有数据卷

## 配置映射
- 非敏感环境变量 -> `ConfigMap/sherpa-config`
- 敏感项（模型密钥、数据库凭据）-> `Secret/sherpa-minimax` + `Secret/sherpa-postgres`

## 目录约定（K8s）
- `/shared/tmp`
- `/shared/output`
- `/app/job-logs/jobs`

## 说明
- 当前代码已切换为 Postgres-only job store，启动时必须提供 `DATABASE_URL`。
- 生产建议启用 cert-manager 管理 Ingress TLS；当前 base 清单默认 HTTP，便于先联调。
