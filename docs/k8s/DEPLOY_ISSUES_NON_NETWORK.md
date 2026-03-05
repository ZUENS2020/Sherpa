# 部署问题总结（非网络因素）

> 日期：2026-03-05  
> 范围：仅包含可通过工程与流程改造解决的问题，不包含链路抖动/源站可达性问题。

## 1. 问题清单与根因

1. `sherpa-web` 启动早于 `postgres`，导致应用启动即连库失败并 CrashLoop。
2. `sherpa-minimax` Secret 缺失或创建时机不稳定，导致 `CreateContainerConfigError`。
3. PVC 在无默认 `StorageClass` 集群中容易长期 `Pending`，触发连锁失败。
4. Cloudflare 接入曾依赖主机级临时代理（`localhost:80`），与“仓库即部署”原则冲突。
5. Overlay 对外部组件存在隐式依赖（历史上依赖 `ingress` 命名空间的服务），环境迁移后易失效。
6. 发布流程缺少 PR 级部署前校验，清单错误在运行期才暴露。

## 2. 已落地改进

1. 工作流强制等待顺序：`postgres -> sherpa-web -> sherpa-frontend`。
2. 部署工作流中统一创建运行时 Secret（`sherpa-minimax`、`sherpa-postgres`）。
3. 新增 `PR Deploy Check` 工作流：对 `base/dev/prod/cloudflare` 执行 `kustomize build` 与 guard 校验。
4. Cloudflare overlay 改为自包含：
   - `cloudflared` + `loopback-proxy(nginx)` sidecar
   - `/` 反代到 `sherpa-frontend`，`/api/` 反代到 `sherpa-web`
5. 移除“主机临时代理”依赖，部署与回滚都走 K8s 声明式资源。
6. `prod` 工作流新增失败回滚步骤（deployment 级）。

## 3. 新的自动化发布策略

1. `PR Deploy Check`：PR 到 `dev/main` 时自动执行（清单检查 + actionlint）。
2. `Deploy Dev`：
   - 触发：`PR -> dev`（非 draft，同仓库）与 `push dev`
   - 自动构建并发布镜像
   - 自动部署到 `sherpa-dev`，并执行健康检查
3. `Deploy Prod`：
   - 触发：`push main`（即 PR 合并后）
   - 结合 GitHub Environment `prod` 审批门禁
   - 自动部署到 `sherpa-prod`，失败自动回滚

## 4. 运维与配置要求（非网络）

1. GitHub Environments:
   - `dev`：`KUBE_CONFIG_B64_DEV`、`MINIMAX_API_KEY_DEV`、`POSTGRES_PASSWORD_DEV`
   - `prod`：`KUBE_CONFIG_B64_PROD`、`MINIMAX_API_KEY_PROD`、`POSTGRES_PASSWORD_PROD`
2. 集群要求：
   - 可用 PVC 方案（默认 StorageClass 或手动 PV/PVC 绑定策略）
   - 具备执行 `rollout status` 的基础资源配额
3. Cloudflare Token 通过 K8s Secret 注入，不写入仓库文件。

## 5. 后续建议

1. 在 `dev/prod` workflow 增加“PVC 全部 Bound”显式检查步骤（避免运行期才发现）。
2. 在 `prod` workflow 增加 smoke test 扩展（例如任务提交 + `/api/task/{id}` 查询）。
3. 将 `k8s/overlays/cloudflare` 纳入每次发布后的自动探活。
