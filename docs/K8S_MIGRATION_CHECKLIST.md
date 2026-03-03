# Sherpa 迁移至 Kubernetes 清单（SHE-52）

> 目标：以“先可用、再增强”为原则完成从 docker-compose 到 k8s 的迁移，优先消除 DinD 风险并建立可持续运维能力。

## 0. 里程碑总览

- [x] M1：基础设施与路由可用（Ingress + Config + PVC）
- [x] M2：核心执行链路完成去 DinD（K8s Job Executor）
- [x] M3：状态持久化迁移到 Postgres 并增强幂等/锁
- [x] M4：可观测性 + CI/CD + E2E 验收闭环

---

## 1. M1 基础设施与路由（建议先完成）

### 1.1 Compose → K8s 映射（Spike）
- [ ] 输出服务映射表：frontend/web/worker（如有）对应 Deployment
- [ ] 输出网络映射：compose network 对应 Service + Ingress
- [ ] 输出存储映射：volume 对应 PVC（含 accessMode 与容量建议）
- [ ] 输出配置映射：env 拆分为 ConfigMap/Secret
- [ ] 决策记录：共享目录先用 PVC，后续可演进对象存储

**验收证据**
- [ ] `docs/` 内有映射文档与目录规范

### 1.2 Ingress 路由替代 gateway
- [ ] `/` 路由到 frontend Service
- [ ] `/api/*` 路由到 sherpa-web Service
- [ ] 记录 TLS 策略（是否启用 cert-manager、证书来源）

**验收证据**
- [ ] `kubectl get ingress` 可见路由规则
- [ ] 浏览器访问 `/` 正常
- [ ] `curl /api/health`（或等价健康检查）返回 200

### 1.3 配置与密钥外置化
- [ ] 敏感配置迁移到 Secret（例如 `MINIMAX_API_KEY`）
- [ ] 非敏感配置迁移到 ConfigMap
- [ ] 更新部署模板，确保 Pod 通过 env/envFrom 注入配置
- [ ] 文档化“配置更新策略”（滚动重启/热更新约束）

**验收证据**
- [ ] 仓库中不再提交明文敏感值
- [ ] Pod 环境变量与预期一致

### 1.4 共享目录与产物（PVC 第一阶段）
- [ ] 定义 `/shared/tmp`、`/shared/output` 目录约定
- [ ] web 与 job 挂载同一逻辑共享卷（或可互通卷）
- [ ] 输出清理策略（TTL/CronJob）初版

**验收证据**
- [ ] Job 可写入 artifacts 与 `run_summary`
- [ ] web 可读取对应 job_id 的输出

---

## 2. M2 去 DinD（核心改造）

### 2.1 执行器切换
- [ ] 移除对 `DOCKER_HOST=tcp://...` 的强依赖
- [ ] 新增 Kubernetes Job Executor（由 web 创建 Job）
- [ ] 规范 Job 命名（包含 `job_id` / trace 信息）

### 2.2 状态跟踪与结果回收
- [ ] web 能轮询/监听 Job 状态（Pending/Running/Succeeded/Failed）
- [ ] Job 结束后 web 能回收输出路径与摘要
- [ ] 超时、失败状态可落库并可查询

### 2.3 幂等保护
- [ ] 同 `job_id` 创建前先做存在性检查
- [ ] 避免重复创建多个 Job（唯一键或 CAS/锁）
- [ ] 对重复触发返回可解释结果（已在执行/已完成）

**验收证据**
- [ ] 单 job 全流程从提交到产物可闭环
- [ ] 重复触发不产生重复执行

---

## 3. M3 数据层升级（SQLite → Postgres）

### 3.1 数据库接入
- [ ] 新增 `DATABASE_URL`
- [ ] job_store 改为 Postgres

### 3.2 索引与查询性能
- [ ] 建立索引：`job_id`、`status`、`updated_at`
- [ ] 关键查询路径（任务列表、任务详情、状态轮询）可接受

### 3.3 一致性与并发控制
- [ ] 状态机转换规则清晰（含失败重试）
- [ ] 避免 resume/trigger 并发重复执行
- [ ] 关键状态变更有结构化日志

**验收证据**
- [ ] 所有新任务状态持久化在 Postgres
- [ ] 并发压测下无重复执行与脏状态

---

## 4. M4 可观测性、CI/CD 与 E2E

### 4.1 可观测性基线
- [ ] web 与 job 日志统一关键字段（`job_id`、phase、error_code）
- [ ] 集群可检索 stdout 日志
- [ ] 指标骨架：Job failure rate、queue backlog、OOM、CPU throttling

### 4.2 CI/CD 固化
- [ ] main 分支构建并推送镜像（web/frontend/fuzz-runtime）
- [ ] 固定部署方式（Helm 或 Kustomize 二选一）
- [ ] 回滚流程文档化（tag + rollback 命令）

### 4.3 E2E 验收（zlib）
- [ ] 成功场景：提交任务 → plan/synthesize → build/run → 产物生成
- [ ] 失败场景：构建失败/超时，状态与日志正确
- [ ] 输出验收报告（成功率、失败归因、后续改进）

**验收证据**
- [ ] E2E 报告与命令记录已归档

---

## 5. 发布与回滚前检查

- [x] 变更清单已评审（配置、部署、数据、运行时）
- [x] Postgres 备份与回滚路径已验证
- [x] 关键告警规则已生效（至少骨架）
- [x] 运行手册已更新（故障排查、日志定位、任务重试）

---

## 6. 推荐执行顺序（简版）

1. [x] 先做 M1：Ingress + Config/Secret + PVC
2. [x] 再做 M2：K8s Job Executor 去 DinD
3. [x] 然后 M3：Postgres 与幂等/锁
4. [x] 最后 M4：Observability + CI/CD + E2E
