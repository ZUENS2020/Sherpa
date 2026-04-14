# Sherpa 吸收 AIxCC AFC 思路迁移计划

## 目标
在不照搬 Theori AFC 实现细节的前提下，迁移其可落地的方法论能力，提升 Sherpa 的稳定性、可解释性和长期 coverage 收益。

当前主流程保持不变：
`analysis -> plan -> synthesize -> build -> run -> coverage-analysis`

## 迁移边界

### 仅迁移能力，不照搬实现
- 迁移：策略、数据契约、可观测性、调度与反馈机制。
- 不迁移：AFC 的全局状态管理风格、私有镜像依赖、重型云部署拓扑。

### 约束
- 不新增主状态机阶段。
- 继续保持 fail-open 主语义（非硬失败不自动停机）。
- 优先增量改造，避免大重构。

## M1（优先，低风险高收益）

### 1. Agent/Stage 可观测性收口
- 增加统一 trace 结构：`stage/tool/model/latency/token/error_kind/error_code/retry_count`。
- 固化单任务决策链快照（每轮 choose-target / choose-seed / choose-repair 的输入输出摘要）。
- 输出到任务可见字段和归档工件，便于复盘。

### 2. Crash/Vuln 去重
- 对 crash 签名做统一归并（stack top + sanitizer type + key frame hash）。
- repair 链路消费去重结果，避免重复处理同类问题。

### 3. 目标/策略可解释打分
- 引入 score breakdown：
  - `coverage_gap`
  - `complexity/depth`
  - `api_relevance`
  - `recent_yield_penalty`
- 将分解结果落盘到 `fuzz/selected_targets.json`。

### M1 验收
- 能解释“为什么回 plan/为什么没 improve”。
- 同签名 crash 不再反复触发同类修复动作。
- target 选择具备可追踪评分依据。

## M2（中风险，中高收益）

### 1. Coverage Frontier 调度
- 建立目标边际收益指标：单位时间新增覆盖率。
- coverage-analysis 优先依据 frontier 做 target 切换建议。

### 2. 策略翻转器（plateau 触发）
- 连续 plateau 后在三类策略中切换：
  - `seed-first`
  - `harness-first`
  - `target-switch`
- 每次翻转必须记录 `strategy_delta`，禁止重复 no-op。

### 3. 任务级失败记忆
- 增加 `constraint_memory`（短期记忆）：
  - 失败签名
  - 已尝试策略
  - 无效补丁摘要
- plan/synthesize 强制读取并规避已证伪路径。

### M2 验收
- plateau 场景不再长期 run<->coverage 空转。
- 低收益目标会被阶段性降权并轮换。
- repair 输出包含“相对上一轮的变化点”。

## M3（高收益，成本较高）

### 1. 标准化评测集 + Nightly
- 建立固定仓库集（如 zlib/libpng/yaml-cpp 等）每日回归。
- 指标：
  - 首次有效构建成功率
  - 首崩时间
  - 覆盖增量
  - 循环率

### 2. 任务回放模拟器
- 支持 schedule 回放和阶段复现（聚焦线上疑难任务）。
- 将线上问题可重放到本地/测试环境。

### 3. 工件标准化打包
- 统一导出 `analysis/plan/build/run/crash/coverage` 工件。
- 支持自动评分与横向对比。

### M3 验收
- 每次改动都能量化收益或回退风险。
- 线上疑难任务可稳定复现与定位。

## 不建议直接照搬的项
- AFC 私有依赖链（镜像、密钥、权限数据）。
- 高预算下的激进多模型并发策略。
- 全量复制其部署编排。

## 推荐实施顺序
1. M1 全量完成并稳定运行一轮。
2. M2 按 `frontier -> strategy flipper -> memory` 顺序上线。
3. M3 建评测与回放，作为长期演进基座。

## 风险与回滚
- 风险：策略调整引入短期波动，可能影响单任务成功率。
- 回滚点：
  - 评分权重/阈值全部可配置化。
  - 可一键关闭 frontier/strategy 翻转/constraint memory 的新逻辑，回退到旧决策路径。

