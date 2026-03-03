# Kubernetes 发布与回滚门禁（SHE-59）

- 日期：2026-03-03
- 关联任务：`SHE-59`、`SHE-52`
- 基线清单：`docs/K8S_MIGRATION_CHECKLIST.md` 第 5 节

## 1. 门禁项核对

| 门禁项 | 结论 | 证据 |
|---|---|---|
| 变更清单评审（配置/部署/数据/运行时） | 通过 | `docs/k8s/MAPPING.md`、`docs/k8s/DEPLOY.md`、`docs/k8s/E2E_ZLIB_REPORT.md` |
| Postgres 备份与回滚路径验证 | 通过（路径已固化） | `docs/k8s/RUNBOOK.md` 第 6 节 |
| 关键告警规则骨架 | 通过（骨架已固化） | `docs/k8s/RUNBOOK.md` 第 7 节（基于 `/api/metrics`） |
| 运行手册（故障排查/日志定位/任务重试） | 通过 | `docs/k8s/RUNBOOK.md` |

## 2. 可执行验证记录

1. K8s 清单语法与对象完整性（客户端 dry-run）：

```bash
kubectl apply --dry-run=client -k k8s/base
```

2. 核心测试回归（API + worker）：

```bash
./.venv314/bin/python -m pytest -q \
  tests/test_api_stability.py \
  tests/test_k8s_job_worker.py
```

3. 关键工作流文件可解析：

```bash
python3 - <<'PY'
import yaml
with open('.github/workflows/k8s-build-deploy.yml','r',encoding='utf-8') as f:
    data=yaml.safe_load(f)
assert isinstance(data, dict) and 'jobs' in data
print('workflow_yaml_ok')
PY
```

## 3. 回滚说明

1. 应用层回滚：`kubectl rollout undo deploy/sherpa-web|sherpa-frontend`
2. 数据层回滚：按 `RUNBOOK` 中的 `pg_dump/psql` 路径执行
3. 回滚后验证：`/api/health`、`/api/system`、`/api/metrics` + 任务提交流程抽测

## 4. 收口结论

1. `SHE-59` 门禁项已满足收口条件。
2. `SHE-52` 的 M1~M4 子任务已完成，父任务可进入 Done。

## 5. 已知风险（保留）

1. 当前 metrics 为进程内窗口统计，服务重启后窗口归零。
2. 告警仍为骨架层，后续建议接入 Prometheus + Alertmanager + Grafana 完整链路。
