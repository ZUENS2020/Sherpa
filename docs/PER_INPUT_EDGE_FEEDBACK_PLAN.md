# 档 4 改造计划 v2：Per-Input Coverage Frontier

## 目标

把 fuzz 反馈从“语料库整体的 cov/ft 标量”细化到“每个 corpus input 独立 replay 后得到的源码覆盖摘要”，让 AI 在 `improve-harness` 和 seed 改进环节能基于“这条 seed 已经走到哪些函数/行区间、距离目标还差哪一段源码覆盖”做决策，而不是继续只看 `ft_delta == 0` 这类聚合信号。

**v2 范围收缩**：

- 一期采集机制**锁死**为：`profraw + llvm-cov export` 的 per-input replay 路径
- `sancov`
- `trace-pc-guard`
- `pc-table`
- `trace-cmp`
- clang AST `missing_predicates`

以上全部从一期移出，统一放到二期。

## 与档 1–3 的关系

档 1–3 仍然是这项工作的上游，其中：

- 档 1：结构化 coverage 字段进入 state / prompt
- 档 2：llvm-cov 行/分支级导出
- 档 3：cgprocessor 反推 entry -> target 路径

档 4 v2 的一期只依赖：

1. 档 1 已落地 `coverage_uncovered_functions_detailed`
2. llvm/clang 工具链可在 replay binary 上产出 `profraw`
3. 现有 `coverage-analysis -> improve-harness` 链路可接收新的结构化反馈

与档 3 的联动收益仍然很大，但不是一期前置依赖。  
一期允许 `distance_to_target = 0` 或缺省，等档 3 落地后二期再补真实距离。

## 现状基线

| 模块 | 文件:行 | 当前能力 |
|---|---|---|
| 编译 flags | [fuzz_unharnessed_repo.py:172](../harness_generator/src/fuzz_unharnessed_repo.py:172) | 现有 sanitizer fuzz binary，未区分 replay profile |
| 运行 | [fuzz_unharnessed_repo.py](../harness_generator/src/fuzz_unharnessed_repo.py) | 直接 `binary corpus_dir`，libFuzzer 聚合运行 |
| 覆盖采集 | `collect_source_coverage()` ([fuzz_unharnessed_repo.py:1172](../harness_generator/src/fuzz_unharnessed_repo.py:1172)) | 仅对整轮 corpus 跑一次 llvm-cov，拿累计覆盖 |
| seed feedback | `_build_seed_feedback()` ([workflow_graph.py:3078](../harness_generator/src/langchain_agent/workflow_graph.py:3078)) | 只有 `cov_delta / ft_delta / early_new_units_30s` 等聚合值 |
| improve prompt | [workflow_graph.py:11112](../harness_generator/src/langchain_agent/workflow_graph.py:11112) | source coverage 仍然偏散文，没有 per-input 结构化前沿 |
| coverage 决策 | [workflow_coverage_decision.py:119](../harness_generator/src/langchain_agent/workflow_coverage_decision.py:119) | 已有 `seed_limited / target_limited / harness_limited` 分流，但没有 replay 质量门控 |

**关键缺口**：

- 每个 input 的独立覆盖画像不存在
- 无法知道“哪个 seed 最接近目标路径”
- plateau 时无法区分“seed 不行”还是“harness 真已经榨干”

## 一期主输出

一期不再追求“真实 edge bitmap”，改为“per-input source coverage frontier”。

目录结构：

```text
fuzz/coverage/
├── per_input/
│   ├── <sha1>.profraw
│   ├── <sha1>.llvmcov.json
│   └── manifest.json
├── frontier.json
└── llvm_cov_export.json
```

`manifest.json` 是一期核心索引，`binary_hash` 进入**主 schema**，不是附注：

```json
{
  "schema_version": 2,
  "generated_at": 1777150795,
  "binary_hash": "sha256:...",
  "build_id": "2026-04-26T09:30Z-libpng-replay",
  "replay_mode": "profraw-llvm-cov",
  "inputs": {
    "37ab...": {
      "path": "fuzz/corpus/seed-37ab",
      "size_bytes": 412,
      "mtime_ns": 1777150700123456789,
      "exec_time_us": 1820,
      "function_hits": 19,
      "region_hits": 84,
      "export_path": "fuzz/coverage/per_input/37ab....llvmcov.json",
      "replay_status": "ok",
      "replay_round_id": 44,
      "replay_error": null
    }
  }
}
```

`replay_status` 取值：

- `ok`：成功 replay 并生成 `.llvmcov.json`
- `pending`：本轮预算耗尽未处理，下轮优先重排
- `failed`：replay 进程失败、profraw 损坏或 timeout，`replay_error` 记录原因

`profraw_path` 不再出现在 manifest 中——成功后 profraw 已被清理（详见 4.2 清理策略）。

`manifest` 还必须支持**删除清理**：

- 若某个 corpus input 在本轮扫描中已不存在，则从 `manifest.inputs` 中移除
- 同时从 `frontier.json` 中移除该 input 的历史摘要
- 不允许保留“文件已删但 frontier 仍引用”的脏条目

`frontier.json` 作为一期直接喂给 LLM 的结构化输入：

```json
{
  "schema_version": 2,
  "generated_at": 1777150795,
  "binary_hash": "sha256:...",
  "frontier_inputs": [
    {
      "input_hash": "37ab...",
      "size_bytes": 412,
      "exec_time_us": 1820,
      "function_hits": 19,
      "region_hits": 84,
      "depth_score": 0.62,
      "frontier_functions": [
        {
          "name": "png_handle_iCCP",
          "file": "pngrutil.c",
          "line": 1843,
          "distance_to_target": 0,
          "uncovered_regions_nearby": 12
        }
      ],
      "rationale": "触达 png_chunk_unknown_handler 附近，但 png_handle_iCCP 仍未覆盖"
    }
  ]
}
```

这里的 `frontier_functions` 一期只承诺：

- 函数名
- 文件
- 行号
- 周边未覆盖 region 数

不承诺 AST 级 `missing_predicates`。

## 技术方案

### 4.1 编译期：双 binary，但一期只做 replay coverage profile

保留现有 fuzz binary 继续承担真实 fuzz。

新增 sibling replay binary：

- profile 名：`coverage_replay`
- 编译目标：`fuzz/out/<harness>_replay`
- flags：
  - `-fsanitize=address,undefined,fuzzer`
  - `-fprofile-instr-generate`
  - `-fcoverage-mapping`

**一期明确不加**：

- `-fsanitize-coverage=trace-pc-guard`
- `trace-cmp`
- `pc-table`

插入点：

- [fuzz_unharnessed_repo.py:172](../harness_generator/src/fuzz_unharnessed_repo.py:172) 的 profile 表新增 `coverage_replay`
- build 阶段串行 build 两份 binary

这样做的好处是机制单一：

- replay 只产出 `profraw`
- 分析只消费 `llvm-profdata + llvm-cov export`
- 不引入第二套 edge runtime 语义

### 4.2 采集：增量 replay，不修改主 fuzz 行为

graph 中新增：

```text
fuzz run -> per_input_replay -> coverage-analysis -> improve-harness
```

`per_input_replay` 只做增量：

1. 扫描 `fuzz/corpus/`
2. 对比 `manifest.json.inputs[*].mtime_ns` 和 `binary_hash`
3. 找出：
   - 新增 input
   - 修改过的 input
   - 已删除 input（从 manifest/frontier 清理）
   - `binary_hash` 失配导致需要重放的 input
   - **上轮残留 pending**：`manifest.inputs[*].replay_status == "pending"` 的条目（见预算耗尽语义）
4. 拼成本轮 replay queue：`(上轮 pending) ∪ (本轮新增/修改)`，按 mtime 从新到旧排序，新 input 优先
5. 在预算/并行约束内逐个 replay；未做完的回写 `replay_status = "pending"` 留给下轮

#### Replay 预算与并行度

| 配置 | 默认值 | 含义 |
|---|---|---|
| `SHERPA_REPLAY_BUDGET_SEC` | `60` | 单轮 replay 总预算；超时停止排队中的 input，不影响已完成的入 frontier |
| `SHERPA_REPLAY_PARALLELISM` | `4` | 并发 replay worker 数（每个 input 独立，天然可并行） |
| `SHERPA_REPLAY_PER_INPUT_TIMEOUT_SEC` | `5` | 单 input replay 超时；超时该 input 标 `replay_failed_per_input`，不进 frontier，不阻塞队列 |

**预算耗尽 ≠ replay 失败**：
- 预算耗尽：剩余 input 标 `replay_status="pending"`，`coverage_replay_pending_inputs > 0`，`coverage_replay_error` 留空
- 真失败（replay binary 不存在、profraw 损坏、llvm-profdata merge 失败等）：写 `coverage_replay_error`，单 input 标 `replay_status="failed"` 但仍计入 manifest（带错误原因）

#### Profraw 清理

每个 input 的 `.profraw → .profdata → .llvmcov.json` 链生成完毕后立即删除 `.profraw` 和 `.profdata`，**仅保留 `.llvmcov.json`**。否则 1000 input 的 corpus 几百 MB 中间产物会累积到磁盘。`replay_failed_per_input` 的 profraw 保留 1 份用于排障，过 7 天清理。

#### `exec_time_us` 一期取值约束

一期**不引入**统一 harness timing shim，也不要求在 `LLVMFuzzerTestOneInput` 内埋点。

因此 `exec_time_us` 一期定义为：

- 单 input replay 子进程从启动到退出的 wall-clock 时间
- **包含**进程启动、ASan 初始化、profile runtime 开销

这意味着它不是纯净的“业务逻辑执行时间”，但实现简单、口径稳定，足够用于一期的相对排序。二期若引入统一 replay shim，再替换成更细粒度的 in-process timing。

执行形式：

```bash
LLVM_PROFILE_FILE=fuzz/coverage/per_input/<sha1>.profraw \
  fuzz/out/<harness>_replay fuzz/corpus/<input>

llvm-profdata merge -sparse fuzz/coverage/per_input/<sha1>.profraw \
  -o fuzz/coverage/per_input/<sha1>.profdata

llvm-cov export \
  fuzz/out/<harness>_replay \
  -instr-profile=fuzz/coverage/per_input/<sha1>.profdata \
  > fuzz/coverage/per_input/<sha1>.llvmcov.json
```

一期不引入 `.sancov`，不做 edge dump。

#### `llvm-cov` 结果过滤约束

一期 frontier 只允许消费 **repo source root** 下的源码文件覆盖，不允许把运行时和第三方噪声混进去。

过滤规则写死到 replay/export 实现中：

- 保留：当前仓库 checkout root 下的源码文件
- 排除：
  - libFuzzer runtime
  - sanitizer runtime
  - `/usr/include`
  - system library source
  - toolchain/LLVM 自身路径
  - vendored third-party 目录（若后续需要，单独白名单）

也就是说，`frontier_functions` 必须来自“对 harness/目标仓库真正有意义”的源码，而不是 coverage runtime 噪声。

### 4.3 frontier 选择：基于 function / region，而不是 edge

一期 frontier score 改成源码覆盖导向，**一期不引入 `depth_score`**（依赖档 3 callgraph，留二期）：

```text
frontier_score =
    α * unique_frontier_functions
  + β * nearby_uncovered_regions
  - δ * exec_time_us
  - ε * size_bytes
```

#### 默认权重

```
α = 1.0      # 函数命中
β = 0.5      # 邻近未覆盖 region
δ = 0.001    # exec_time penalty (us → 1ms ≈ 1.0 减分)
ε = 0.0001   # size penalty (byte → 1KB ≈ 0.1 减分)
```

可通过 `SHERPA_FRONTIER_WEIGHTS_JSON='{"alpha":1.0,"beta":0.5,...}'` 覆盖；二期接入档 3 时再加 `gamma * depth_score`。

#### 字段定义

- **`unique_frontier_functions`**：该 input 命中的函数中，**满足"region 覆盖率 < 50%"** 的函数数。50% 阈值通过 `SHERPA_FRONTIER_PARTIAL_THRESHOLD=0.5` 可调。"region 覆盖率"取自 llvm-cov export JSON 中 `functions[*].regions` 的 `count > 0` 比例
- **`nearby_uncovered_regions`**：对该 input 命中的每个 frontier function，统计其内部 `count == 0` 的 region 数，求和
- **`exec_time_us`**：replay 子进程 wall-clock 时间（**包含**进程启动和 sanitizer/profile 开销）；一期不要求 harness 内埋点，取不到则置 0 不参与排序

#### `binary_hash` 计算时机

`binary_hash = sha256(fuzz/out/<harness>_replay)`，build 完成后立刻计算并写入 manifest。**不**与 fuzz binary 混用——fuzz binary 不带 `-fcoverage-mapping`，hash 不同也是预期。

输出 top-K `frontier_inputs` 给后续节点。

### 4.4 喂给 LLM

改三个消费点：

| 节点 | 文件 | 新增上下文 |
|---|---|---|
| `improve-harness` | [workflow_graph.py:11112](../harness_generator/src/langchain_agent/workflow_graph.py:11112) | `frontier_inputs` top-3 + 每个 input 的 top frontier functions |
| seed 改进 | 同 improve-harness 调用链 | 完整 `frontier_summary` |
| `coverage-analysis` | [workflow_coverage_decision.py:119](../harness_generator/src/langchain_agent/workflow_coverage_decision.py:119) | `frontier_summary` + replay freshness / success 状态 |

上下文控制：

- `frontier_inputs`: top-3
- 每个 input 最多 5 个 `frontier_functions`
- 每个函数只保留 `name/file/line/uncovered_regions_nearby`

### 4.5 `harness_limited` 判定：新增 3 条前置门控

一期必须防止“replay 没跑好却误判 harness_limited”。

因此 `plateau + frontier empty` 不能直接下结论，必须同时满足下面 3 条门控：

1. **replay_stage_success**
   - 本轮 `per_input_replay` 作为一个阶段成功完成
   - `coverage_replay_runtime_sec` 已写入
   - 没有 `coverage_replay_error`
   - 允许存在少量 `replay_status="failed"` 的单 input，只要不是基础设施级失败

2. **manifest_fresh_for_current_binary**
   - `manifest.binary_hash == current_replay_binary_hash`
   - 不允许拿旧 binary 生成的 per-input 数据做当前轮决策

3. **replay_queue_drained**
   - 本轮新增/修改 corpus input 已全部 replay 完成
   - `pending_replay_inputs == 0`

只有在：

- `plateau_no_gain == true`
- `frontier_inputs == []`
- 上述 3 条门控全部满足

时，才允许把 `coverage_bottleneck_kind` 升级为 `harness_limited`。

否则：

- replay 阶段失败 -> `coverage_replay_degraded`
- manifest 不新鲜 -> `coverage_replay_stale`
- 还有待 replay input -> `coverage_replay_pending`

这些都不应折叠成 `harness_limited`。

## 状态字段

新增 state keys：

- `coverage_per_input_manifest_path`
- `coverage_frontier_path`
- `coverage_frontier_summary`
- `coverage_replay_runtime_sec`
- `coverage_replay_binary_hash`
- `coverage_replay_stage_success`
- `coverage_replay_error`
- `coverage_replay_pending_inputs`
- `coverage_replay_failed_inputs`

其中 `coverage_frontier_summary` 是给 DB/API/决策快速查询的扁平摘要。

## 文件改动清单

新增：

- [`harness_generator/src/coverage_replay.py`](../harness_generator/src/)  
  增量 replay、profraw 合并、llvm-cov export、manifest/frontier 构建
- [`harness_generator/src/coverage_replay_schema.py`](../harness_generator/src/)  
  manifest/frontier schema
- [`tests/test_coverage_replay.py`](../tests/)  
  replay + manifest + frontier 端到端
- [`tests/test_coverage_frontier.py`](../tests/)  
  frontier 排序、binary_hash 失配、pending replay 判定

修改：

- [`harness_generator/src/fuzz_unharnessed_repo.py`](../harness_generator/src/fuzz_unharnessed_repo.py)
  - 新增 `coverage_replay` build profile
  - 新增 replay binary build 调用
- [`harness_generator/src/langchain_agent/workflow_graph.py`](../harness_generator/src/langchain_agent/workflow_graph.py)
  - 注册 `per_input_replay` node
  - `_build_seed_feedback()` 增 `frontier_summary`
  - improve-harness prompt 注入 `frontier.json`
  - state schema 增 replay 相关字段
- [`harness_generator/src/langchain_agent/workflow_coverage_decision.py`](../harness_generator/src/langchain_agent/workflow_coverage_decision.py)
  - `evaluate_coverage_decision()` 接 `frontier_summary`
  - 新增 `harness_limited` 三条门控
- [`docker/Dockerfile.web`](../docker/Dockerfile.web)
  - 确认 `llvm-profdata` / `llvm-cov` 可用

前端：

- [`frontend-next/components/LogPanel.tsx`](../frontend-next/components/LogPanel.tsx)  
  可选二阶段接入，展示 top frontier inputs

## 工作拆分

### 一期（必须先交付）

1. replay binary
2. per-input `profraw -> llvmcov.json`
3. manifest with `binary_hash`
4. frontier summary
5. workflow 接入
6. `harness_limited` 三门控

### 二期（明确后移）

以下全部不在一期范围：

- `sancov`
- `trace-pc-guard`
- `trace-cmp`
- edge bitmap
- edge inverted index
- `functions_to_inputs` 的 edge 精细映射
- clang AST `missing_predicates`
- 真实 `distance_to_seed`

二期可以在一期稳定后单独开文档。

## 工作量估计（v2）

| 阶段 | 工时 | 风险 |
|---|---|---|
| A. replay binary + profile pipeline | 2 天 | 低 |
| B. manifest / binary_hash / 增量 diff | 1.5 天 | 低 |
| C. frontier summary 生成 | 1.5 天 | 低 |
| D. workflow node + state schema + prompt 接入 | 1 天 | 低 |
| E. `harness_limited` 三门控 | 0.5 天 | 低 |
| F. 测试（unit + libpng smoke） | 2 天 | 中 |
| **合计** | **~8.5 工作日** | — |

比 v1 少掉一整块 AST/edge 复杂度，路径更稳。

## 验证

### 单元测试

- `tests/test_coverage_replay.py`
  - 3 个 fixture input
  - 验证 per-input `profraw` / `llvmcov.json` 生成
  - 验证 `manifest.binary_hash`
- `tests/test_coverage_frontier.py`
  - 空 corpus
  - 单 input
  - binary_hash 失配触发全量 replay
  - pending replay 未清空时，不得判 `harness_limited`

### 集成测试

跑 libpng 一轮完整 workflow：

| 指标 | 改造前 | v2 期望 |
|---|---|---|
| build + fuzz + analysis 总时长 | 基线 X | X + 可控 replay 开销 |
| improve-harness 额外上下文 | 基线 Y | Y + ~2k 到 3k tokens |
| plateau 时误判 harness_limited | 无法定义 | 明显下降 |
| improve-harness 具体性 | 低 | 能指到具体 input / 函数 |

### 黑盒验证

看 libpng / libxml2 这类目标：

1. 改造前是否只会反复调整 dict
2. 改造后是否能基于 frontier inputs 给出更具体的 seed 变换方向

## 回滚策略

- env flag：`SHERPA_PER_INPUT_REPLAY_ENABLED=0`
- 关闭时跳过 `per_input_replay` node，原行为保持不变
- replay binary 独立，不影响主 fuzz binary
- replay 失败只降级反馈，不阻断 fuzz 主链路

## 已知限制 / 二期计划

- Java / Jazzer 不在本期范围
- 大 corpus 首次冷启动仍可能慢
- `distance_to_target` 一期允许粗粒度或缺省
- 二期再单独引入：
  - edge runtime
  - sancov / trace-pc-guard
  - AST predicate 抽取
  - 真正的 edge inverted index
