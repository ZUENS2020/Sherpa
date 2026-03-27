---
name: "promefuzz-fuzzing-guidance"
description: "Guides fuzzing target and seed planning using PromeFuzz analysis outputs. Invoke when user asks for fuzzing strategy based on API/callgraph/semantics."
---

# PromeFuzz Fuzzing Guidance

## 适用场景

- 用户希望基于 MCP 分析结果制定 fuzz 策略
- 需要从 API、调用图、函数语义中挑选高价值目标
- 需要生成“可执行”的 fuzz 任务清单

## 前置数据

- API 函数列表（`extract_api_functions` 产物）
- 调用图（`build_library_callgraph` 产物）
- 函数用法/库目的（Comprehender 产物，可选）

## 分析步骤

1. 目标函数分层
   - 入口型 API（外部可直接调用）
   - 桥接型函数（连接多模块）
   - 深层高扇入/高扇出函数（高传播风险）
2. 输入面识别
   - 字节流、文件、协议包、结构体参数
   - 长度字段、编码字段、边界值触发点
3. 序列关系识别
   - 初始化 -> 处理 -> 释放
   - 成对调用（open/close, init/free）
4. 生成 fuzz 任务
   - Harness 入口建议
   - Seed 类型建议
   - 变异重点建议

## 输出模板

- 候选目标 Top N（附理由）
- 每个目标的输入模型
- 调用顺序约束
- 初始 seed 规划
- 风险点与预期崩溃类型

## 约束说明

- 若语义能力返回占位结果，优先依赖 AST/API/调用图事实数据
- 建议先给“小而确定”的高价值入口，再扩展复杂链路

## 参考来源

- `prompts.py` 中 fuzzing guidance 相关提示词
