# Cloudflare Tunnel（内网接入）

目标：把内网 Kubernetes 中的 Sherpa 入口绑定到 `https://sherpa.zuens2020.work`，不暴露公网端口。

## 1. 前提

1. Cloudflare 已托管 `zuens2020.work`。
2. 已在 Zero Trust 创建 Tunnel，并拿到 `TUNNEL_TOKEN`。
3. Kubernetes 已部署 Sherpa Ingress（`k8s/base/ingress.yaml`）。

## 2. Ingress Host

当前已支持：

1. `sherpa.local`
2. `sherpa.zuens2020.work`

检查命令：

```bash
kubectl -n sherpa get ingress sherpa -o jsonpath='{.spec.rules[*].host}{"\n"}'
```

## 3. 部署 cloudflared（K8s overlay）

### 3.1 准备 Tunnel Secret

```bash
cp k8s/overlays/cloudflare/cloudflare-tunnel-secret.example.yaml \
   k8s/overlays/cloudflare/cloudflare-tunnel-secret.yaml
```

把 `token` 改成真实值后，修改 `k8s/overlays/cloudflare/kustomization.yaml`：

- 把 `cloudflare-tunnel-secret.example.yaml` 替换为 `cloudflare-tunnel-secret.yaml`

### 3.2 应用

```bash
kubectl apply -k k8s/overlays/cloudflare
```

### 3.3 校验

```bash
kubectl -n sherpa get pods -l app=cloudflared
kubectl -n sherpa logs deploy/cloudflared --tail=200
```

## 4. Cloudflare Zero Trust 配置

在 Tunnel 的 Public Hostname 中新增：

1. Hostname: `sherpa.zuens2020.work`
2. Service Type: `HTTP`
3. URL: `http://sherpa-web.sherpa.svc.cluster.local:8001`（仅 `/api`）

推荐更简单方案：

- 直接把 Service URL 设为 Ingress Controller 的集群内地址（或节点地址），并让 Ingress 处理 `/` 与 `/api` 分流。

## 5. 验证链路

```bash
curl -I https://sherpa.zuens2020.work/
curl -sS https://sherpa.zuens2020.work/api/health
```

## 6. 常见问题

1. 访问 404（nginx）
- 原因：Host 不匹配。
- 检查：`Ingress.rules.host` 是否包含 `sherpa.zuens2020.work`。

2. Tunnel 健康但 API 不通
- 检查 cloudflared 日志中的 upstream 地址。
- 检查 `sherpa-web` Service 与 Pod 端口 8001。

3. 前端空白
- 检查 Ingress `/` 是否路由到 `sherpa-frontend:3000`。
