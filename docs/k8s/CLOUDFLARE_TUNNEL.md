# Cloudflare Tunnel 说明

本文档仅记录当前外网接入方式，不涉及工作流内部执行逻辑。

## 作用

- 为 dev / prod 暴露固定域名
- 将入口流量转发到 Kubernetes Ingress / Service

## 当前原则

- Tunnel 只负责入口访问
- 不参与 stage job 调度
- 不影响 `run`、`build`、`coverage-analysis` 等内部工作流语义

## 验证点

- 域名可以访问前端
- 前端能够请求 `sherpa-web`
- 提交任务后日志可以持续刷新
