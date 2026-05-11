# TianHeng 全面改造计划：持续漏洞挖掘主引擎 + 异步 Fuzz 验证引擎

> 参考项目：[FuzzingBrain (o2lab/afc-crs)](https://github.com/o2lab/afc-crs-all-you-need-is-a-fuzzing-brain)
> 本地 clone：`references/afc-crs-all-you-need-is-a-fuzzing-brain` (commit `0fba44e`)

## 1. 目标与范围

### 1.1 目标
- 把 TianHeng 从单线性 `analysis → plan → synthesize → build → run → coverage-analysis` 模式，升级为"持续漏洞挖掘优先"的双引擎系统。
- 模仿 FuzzingBrain 的 **可疑点生命周期** 和 **攻击策略菜单**，让 TianHeng 具备"发现可疑点 → 生成攻击输入 → 验证 → 反馈 → 继续发现"的完整闭环。AI 自主选择攻击策略组合，系统只校验输出质量。
- 主引擎持续发现漏洞候选，验证引擎异步消费并验证，主引擎不中断继续发现下一个候选。
- 复用现有 fuzz 执行与修复能力，避免推倒重写。

### 1.2 非目标
- 本次不引入全新外部服务依赖（可优先使用现有 DB/队列/k8s 基础设施）。
- 不改变现有 API 路径语义（只做兼容扩展字段）。

---

## 2. 参考架构：FuzzingBrain 核心模式

> 以下是从参考项目提取的关键设计模式，TianHeng 的改造将模仿这些模式。

### 2.1 FuzzingBrain 的漏洞发现循环

```mermaid
flowchart TD
    SA["静态分析<br/>LLVM/Joern 提取调用图<br/>LLM 对函数打分排序"]
    SA --> P0["Phase 0: 静态分析结论直接攻击"]
    SA --> P1["Phase 1: 漏洞分类攻击<br/>OOB / UAF / 溢出 / 注入 逐类试"]
    SA --> P2["Phase 2: 函数过滤攻击<br/>LLM 筛 top-k 函数精准攻击"]
    SA --> P3["Phase 3: 调用路径攻击<br/>沿调用链精准构造输入"]
    SA --> P4["Phase 4: Claude Agent 深度审计"]

    P0 & P1 & P2 & P3 & P4 --> LOOP

    subgraph LOOP["迭代验证循环（每个可疑点，最多 5 轮）"]
        direction TB
        GEN["1. LLM 生成 Python 代码"] --> BLOB["2. 执行代码 → 产出 blob"]
        BLOB --> FUZZ["3. blob 喂给 fuzzer"]
        FUZZ --> CHECK{"4. 崩溃？"}
        CHECK -- "ASAN/MSAN/UBSAN 崩溃" --> POV["保存 PoV，提交<br/>继续下一个可疑点"]
        CHECK -- "无崩溃" --> COV["提取覆盖率反馈"]
        COV --> GEN
    end
```

### 2.2 FuzzingBrain 的覆盖率反馈机制（TianHeng 重点借鉴）

FuzzingBrain 在每次迭代失败后，会把**具体执行了哪些代码路径**回传给 LLM：

```
"The following shows the executed code path of the fuzzer with input x.bin.
You should generate a new x.bin to execute a different code path:
  function_a() at file.c:100-110
  function_b() at file.c:150-180
  ...
  function_x() at file.c:999-1050"
```

- C/C++：LLVM profdata → LCOV → 提取函数级控制流
- 压缩到 200 行（前 100 + 后 100）
- 直接追加到 LLM 对话上下文中

**TianHeng 现状差距**：目前 coverage loop 只看覆盖率数字（cov/ft），不告诉 LLM 具体走了哪些路径、哪些分支没覆盖。

### 2.3 FuzzingBrain 的 Security Analyzer Agent

- 基于 Claude Agent SDK
- 50 轮对话上限
- 工具：Read / Write / Bash / Glob / Grep
- 输出结构化漏洞发现：`VULNERABILITY_FOUND: { type, location, function, root_cause, trigger_condition, seed_input_path, verified, severity }`
- 发现排序：verified 优先，severity 高优先
- 可以自己跑 fuzzer 验证发现

---

## 3. 目标架构

### 3.1 双引擎架构

```mermaid
flowchart TB
    subgraph HUNT["Vuln-Hunt Engine（侦察兵 — 广度优先）"]
        direction LR
        SA2["静态分析"] --> DISC["可疑点发现<br/>（浅层快扫）"]
        DISC --> RANK["候选排序 & 派发"]
        POLL["异步轮询 DB<br/>读结论 + 新发现候选"] -.->|调整优先级<br/>合并新候选| RANK
    end

    RANK -->|"候选卡片<br/>写入 DB / 文件"| DB[("DB / 文件<br/>（异步边界）")]

    DB -->|读取候选| DEEP

    subgraph VALID["Fuzz-Validation Engine（特种兵 — 深度优先）"]

        subgraph DEEP["深度挖掘阶段"]
            direction TB
            DEEP_IN["围绕候选做深度分析<br/>读周边代码 / 调用链 / 数据流"]
            DEEP_IN --> SPLIT["拆分子候选<br/>（发现多个触发路径）"]
            SPLIT --> NEW["发现关联新候选<br/>（调用者/兄弟函数也有问题）"]
        end

        subgraph ITER["迭代验证闭环（最多 5 轮 × 每个子候选）"]
            direction TB
            SYN["synthesize<br/>（攻击导向）"] --> BR["build + run"]
            BR -->|"覆盖率路径反馈<br/>（引擎内部闭环）"| SYN
        end

        DEEP --> ITER
        ITER --> TRIAGE["crash triage"]
    end

    TRIAGE -->|"写入 DB：最终结论<br/>confirmed / rejected / inconclusive<br/>+ 覆盖率 delta 摘要"| DB

    NEW -->|"写入 DB：新发现候选<br/>（深度挖掘产出）"| DB

    DB -.-> POLL
```

**双引擎角色定义**：

| 角色 | 主引擎（侦察兵） | 验证引擎（特种兵） |
|------|-----------------|-------------------|
| 策略 | **广度优先**：快速扫描全代码库，产出候选清单 | **深度优先**：围绕单个候选深挖，拆分子路径，发现关联候选 |
| 输入 | analysis_context.json（全局代码事实） | 单个候选卡片 + 周边源码 |
| 输出 | 候选清单（浅层评分 + attack_hint） | 验证结论 + 覆盖率 delta + **新发现候选** |
| 分析深度 | 函数签名级（看函数名、参数类型、调用图） | 代码行级（读实现、跟数据流、分析分支条件） |
| 运行时间 | 快（一次性产出，分钟级） | 慢（多轮迭代，每候选可能几十分钟） |

**异步边界规则**：
- 主引擎**写候选、读结论 + 读新发现候选**，不等验证完成。
- 验证引擎**读候选、写结论 + 写新发现候选**，内部迭代闭环不出引擎。
- 验证引擎深度挖掘发现的新候选，**写回 DB**，主引擎下次轮询时合并到候选池。
- 细粒度覆盖率反馈**只在验证引擎内部流转**，不回传主引擎。

### 3.2 完整工作流状态机（改造后）

```mermaid
stateDiagram-v2
    [*] --> analysis : 任务开始

    analysis --> vuln_hunt : 产出 analysis_context.json

    state vuln_hunt {
        [*] --> ai_hunt : AI 自主选择策略组合
        ai_hunt --> validate_output : 产出候选清单
        validate_output --> ai_hunt : 不合格（带反馈重试，≤2次）
        validate_output --> [*] : 合格 → vuln_candidates.json
    }

    vuln_hunt --> dispatch : 候选写入 DB

    state dispatch {
        [*] --> rank : 按 priority 排序
        rank --> assign : 分配给验证引擎
    }

    dispatch --> validation : 验证引擎拉取候选

    state validation {
        state deep_analysis {
            [*] --> read_context : 读候选周边代码/调用链/数据流
            read_context --> split_paths : 拆分子候选（多触发路径）
            split_paths --> discover_related : 发现关联新候选
            discover_related --> [*]
        }

        deep_analysis --> iter_loop : 子候选逐个验证

        state iter_loop {
            [*] --> synthesize_v : 攻击导向 harness/seed
            synthesize_v --> build_run : 编译 + 运行
            build_run --> check_crash : 检查崩溃
            check_crash --> cov_feedback : 无崩溃
            cov_feedback --> synthesize_v : 覆盖率路径反馈
            check_crash --> pov_save : 崩溃触发
        }

        iter_loop --> triage : 迭代结束
    }

    validation --> feedback : 写入 DB：结论 + delta 摘要
    validation --> new_candidates : 写入 DB：深度挖掘发现的新候选
    new_candidates --> dispatch : 主引擎轮询合并
    feedback --> dispatch : 调整剩余候选优先级

    validation --> [*] : 所有候选处理完毕
```

### 3.3 关键层
- `Orchestrator`：预算、并发、优先级、退避、重试调度。
- `Knowledge/Memory Layer`：候选库、签名库、失败经验、约束记忆。
- `Artifacts & Observability`：decision trace、评分拆解、降级原因、验证证据链全量落盘。

---

## 4. 可疑点生命周期（模仿 FuzzingBrain）

### 4.1 发现阶段：策略自由 + 输出校验

`_node_vuln_hunt` 不强制 AI 按固定阶段执行。AI 接收全部上下文后自主决定使用哪些策略、以什么顺序组合。系统只校验最终输出是否达标。

#### 策略菜单（prompt 中作为建议提供，AI 自由选用）

| 策略 | 说明 | 来源 |
|------|------|------|
| 证据直攻 | 从 `security_evidence[]` 中高置信度条目直接生成候选 | FuzzingBrain Phase 0 |
| 漏洞分类攻击 | 按类型（OOB/UAF/整数溢出/注入…）逐类扫描 | FuzzingBrain Phase 1 |
| 高危函数筛选 | 从 `target_analysis` 中筛选高风险函数精准攻击 | FuzzingBrain Phase 2 |
| 调用路径构造 | 沿调用图追踪数据流，构造端到端攻击路径 | FuzzingBrain Phase 3 |
| 安全审计 | 综合审计（已由 analysis prompt 部分覆盖） | FuzzingBrain Phase 4 |

AI 可以只用一种策略，也可以组合多种，也可以发明新策略。**系统不关心过程，只校验产出。**

#### 输出校验器（硬性，不合格则带反馈重试）

```mermaid
flowchart TD
    AI["AI 自由产出候选"] --> V{"输出校验"}
    V -- "全部通过" --> OK["写入 vuln_candidates.json ✅"]
    V -- "有不合格项" --> FB["生成校验反馈：\n哪些字段缺失/不具体"]
    FB --> RETRY{"已重试 2 次？"}
    RETRY -- "否" --> AI
    RETRY -- "是" --> DEGRADE["降级：保留合格候选\n丢弃不合格候选\n写入 prompt_render_degraded"]
```

**校验规则**（每个候选必须满足）：

| 规则 | 检查内容 | 不合格反馈示例 |
|------|---------|---------------|
| `evidence_ref` | 必须引用至少一个 `evidence_id` | "候选 X 没有引用任何 evidence，请关联具体代码证据" |
| `attack_hint_complete` | `attack_hint` 必须含 `trigger_condition` + `key_code_path` + `boundary_values` | "候选 X 的 attack_hint 缺少 boundary_values" |
| `concrete_path` | `key_code_path` 必须是具体函数名列表（非描述性文字） | "候选 X 的 key_code_path 是描述文字，请改为具体函数名如 ['png_read_row', 'memcpy']" |
| `concrete_values` | `boundary_values` 必须是具体触发值（非占位符） | "候选 X 的 boundary_values 含占位符 'TBD'，请填入具体值" |
| `min_candidates` | 总候选数 ≥ 3 | "只产出了 1 个候选，请补充更多可疑点" |
| `priority_ordered` | 按 `priority` 降序排列 | "候选未按 priority 排序" |

### 4.2 候选状态机

```mermaid
stateDiagram-v2
    [*] --> pending : 主引擎发现 / 深度挖掘发现

    pending --> deep_analysis : 验证引擎拉取

    state deep_analysis {
        [*] --> analyzing : 读周边代码/调用链/数据流
        analyzing --> split : 拆分子候选（多触发路径）
        split --> discover : 发现关联新候选 → 写回 DB
        discover --> [*]
    }

    deep_analysis --> validating : 子候选逐个进入迭代验证

    validating --> confirmed : 崩溃触发（ASAN/MSAN/UBSAN）
    validating --> inconclusive : 单轮无崩溃但覆盖率有增长
    validating --> rejected : 5 轮无崩溃且无增长

    inconclusive --> validating : 下一轮迭代（≤5 轮）
    inconclusive --> cooling : 5 轮耗尽，覆盖率无增长

    confirmed --> [*] : 提取签名，保存 PoV

    rejected --> cooling : priority = 0

    cooling --> pending : 冷却期过后（24h），\n若有新证据触发增量更新
    cooling --> [*] : 永久放弃（同签名重复 / 多次 rejected）

    note right of deep_analysis
        深度挖掘阶段：
        - 不只是验证，还做深度分析
        - 可拆分出子候选
        - 可发现关联新候选写回候选池
    end note

    note right of confirmed
        回传主引擎：
        - 最终结论
        - 覆盖率 delta 摘要
        - 崩溃签名
        - 新发现候选（如有）
    end note
```

### 4.3 候选输出契约：`vuln_candidate`

每个候选不仅有评分，还必须包含**攻击策略提示**（模仿 FuzzingBrain 的 prompt 构造）：

```jsonc
{
  "candidate_id": "mem_oob_001",           // 幂等主键
  "repo": "libpng",
  "target_api": "png_read_row",
  "target_file": "pngread.c",
  "signal_type": "mem_oob_candidate",      // 漏洞分类
  "security_signals": ["unchecked_memcpy", "external_input_length"],

  // 证据链（与 analysis_evidence.security_evidence 对齐）
  "evidence": [
    {
      "evidence_id": "ev_001",
      "signal_id": "mem_oob_candidate",
      "severity": "high",
      "confidence": 0.85,
      "source_path": "pngread.c",
      "line": 342,
      "summary": "memcpy size derived from untrusted IHDR width without bounds check"
    }
  ],

  // 攻击策略提示（给 synthesize/seed 阶段用）
  "attack_hint": {
    "trigger_condition": "width * channels > allocated row buffer size",
    "suggested_input_pattern": "valid PNG header + oversized IHDR width field (0xFFFF+)",
    "key_code_path": ["png_read_info", "png_read_row", "png_read_filter_row", "memcpy"],
    "boundary_values": ["width=0xFFFFFFFF", "height=1", "channels=4", "bit_depth=8"],
    "vuln_category": "heap-buffer-overflow",
    "sanitizer_hint": "address"             // 建议使用的 sanitizer
  },

  // 评分
  "vuln_likelihood": 0.85,
  "exploitability": 0.70,
  "reachability_confidence": 0.90,
  "detectability_confidence": 0.75,         // fuzzer+sanitizer 组合对该候选的可触发信心
  "priority": 0.82,
  "score_breakdown": {
    "vuln_likelihood": 0.85,
    "exploitability": 0.70,
    "reachability_confidence": 0.90,
    "detectability_confidence": 0.75,
    "coverage_gap": 0.60,
    "complexity_depth": 0.45
  },

  // 状态
  "status": "pending",                      // pending → deep_analysis → validating → confirmed/rejected/inconclusive → cooling
  "validation_rounds": 0,
  "max_validation_rounds": 5,               // 模仿 FuzzingBrain 的 MAX_ITERATIONS
  "created_at": "2026-04-14T12:00:00Z",
  "updated_at": "2026-04-14T12:00:00Z"
}
```

### 4.4 验证结果契约：`validation_result`

```jsonc
{
  "candidate_id": "mem_oob_001",
  "status": "confirmed",                   // confirmed / rejected / inconclusive
  "crash_signature": "heap-buffer-overflow-png_read_row-a3f2b1",
  "sanitizer": "address",
  "crash_trace_summary": "ERROR: AddressSanitizer: heap-buffer-overflow at pngread.c:342",
  "repro_artifacts": {
    "blob_path": "fuzz/povs/mem_oob_001/x1.bin",
    "fuzzer_output_path": "fuzz/povs/mem_oob_001/fuzzer_output.txt",
    "harness_path": "fuzz/png_read_row_fuzz.c"
  },
  "coverage_delta": {
    "before_cov": 120,
    "after_cov": 185,
    "new_edges": 65,
    "new_functions_hit": ["png_read_filter_row", "png_check_chunk_length"]
  },
  "iteration_used": 3,                     // 第几轮迭代触发的
  "failure_reason": "",
  "created_at": "2026-04-14T12:30:00Z"
}
```

### 4.5 签名去重：`signature_cluster`

```jsonc
{
  "signature": "address-heap-buffer-overflow-png_read_row-a3f2b1",
  "count": 1,
  "last_seen": "2026-04-14T12:30:00Z",
  "candidate_ids": ["mem_oob_001"],
  "fuzzer_names": ["png_read_row_fuzz"]
}
```

---

## 5. 细粒度覆盖率反馈（核心改进）

### 5.1 现状问题

TianHeng 当前的 coverage loop 只传递数字：
```
cov: 120, ft: 450, plateau_detected: true, plateau_idle_seconds: 180
```
LLM 只知道"覆盖率没涨"，不知道具体卡在哪、哪些分支没走到。

### 5.2 目标：代码路径级反馈

模仿 FuzzingBrain 的 `CoverageAnalyzer`，在每次 fuzzer 运行后提取**函数级执行路径**：

```
已覆盖的代码路径：
  png_read_info() at pngread.c:100-145
  png_read_update_info() at pngread.c:200-220
  png_read_row() at pngread.c:300-342 ← 到这里就停了

未覆盖的关键分支：
  png_read_filter_row() at pngrutil.c:3400-3500  ← 从未进入
  png_check_chunk_length() at pngread.c:150-180   ← 从未进入

建议：生成输入使 width * channels 超过 row_buffer 分配大小，
触发 png_read_filter_row 中的 memcpy 越界路径。
```

### 5.3 实现路径

#### 5.3.1 覆盖率数据提取

```mermaid
flowchart LR
    RUN["fuzzer 执行"] --> PROF["收集 LLVM<br/>coverage profile"]
    PROF --> MERGE["llvm-profdata merge"]
    MERGE --> EXPORT["llvm-cov export<br/>（LCOV 格式）"]
    EXPORT --> PARSE["解析函数级覆盖"]
    PARSE --> CROSS["与 analysis_context<br/>调用图交叉"]
    CROSS --> OUT["输出：已覆盖函数<br/>+ 未覆盖高风险函数"]
```

#### 5.3.2 反馈压缩（模仿 FuzzingBrain 的 200 行压缩）

- 已覆盖路径：前 50 行 + 后 50 行（大文件截断）
- 未覆盖关键函数：按 vuln_likelihood 排序，取 top 10
- 总长度控制在 200 行以内，避免 LLM context 爆炸

#### 5.3.3 反馈边界：验证引擎内部闭环 vs. 回传主引擎

```mermaid
flowchart TB
    subgraph INTERNAL["验证引擎内部（同步闭环）"]
        direction TB
        SYN2["synthesize"] -->|"attack_hint<br/>+ previous_coverage_paths"| BUILD2["build + run"]
        BUILD2 -->|"coverage_feedback<br/>（函数级路径，≤200行）"| IMP["improve_harness"]
        IMP -->|"uncovered_vuln_paths<br/>code_paths_covered"| SYN2
    end

    subgraph EXTERNAL["回传主引擎（异步，经 DB）"]
        direction TB
        RES["最终结论<br/>confirmed / rejected /<br/>inconclusive"]
        DELTA["覆盖率 delta 摘要<br/>new_edges / new_functions"]
        SIG["崩溃签名"]
    end

    BUILD2 -->|"迭代结束"| RES
    BUILD2 -.->|"摘要"| DELTA
    BUILD2 -.->|"签名"| SIG

    subgraph NEVER["❌ 不跨越异步边界"]
        direction TB
        X1["细粒度代码路径（200行 trace）"]
        X2["中间迭代覆盖率快照"]
        X3["LLM 对话历史"]
    end

    style NEVER fill:#fee,stroke:#c00
    style INTERNAL fill:#efe,stroke:#0a0
    style EXTERNAL fill:#eef,stroke:#00a
```

细粒度覆盖率反馈**只在验证引擎内部流转**，不跨越异步边界：

**验证引擎内部（同步，迭代间流转）：**

| 注入点 | 注入方式 | 用途 |
|--------|---------|------|
| 迭代 N→N+1 的 LLM 调用 | 对话上下文追加 `coverage_feedback` | LLM 据此调整 seed/harness |
| `improve_harness` (in_place) | seed feedback 增加 `code_paths_covered` 字段 | LLM 据此调整 seed |
| `improve_harness` (seed_replan) | replan context 增加 `uncovered_vuln_paths` 字段 | LLM 重新设计 seed 策略 |
| `synthesize` (重入) | hint 增加 `previous_coverage_paths` 字段 | LLM 改写 harness 瞄准未覆盖分支 |

**回传主引擎（异步，写入 DB/文件，主引擎轮询读取）：**

| 回传内容 | 形式 | 用途 |
|----------|------|------|
| 最终结论 | `confirmed / rejected / inconclusive` | 候选状态流转 |
| 覆盖率 delta 摘要 | `new_edges: 65, new_functions: ["f1","f2"]` | 主引擎据此调整剩余候选优先级 |
| 崩溃签名 | `address-heap-buffer-overflow-png_read_row-a3f2b1` | 签名去重 |

**不回传的内容**（避免破坏异步边界）：
- 细粒度代码路径（200 行的执行 trace）
- 每轮迭代的中间覆盖率快照
- LLM 对话历史

### 5.4 覆盖率反馈数据契约

```jsonc
{
  "coverage_feedback": {
    "format": "function_level",
    "total_functions_reachable": 45,
    "functions_covered": 28,
    "functions_uncovered_high_risk": [
      {
        "function": "png_read_filter_row",
        "file": "pngrutil.c",
        "lines": "3400-3500",
        "vuln_candidate_id": "mem_oob_001",
        "reason_uncovered": "需要特定 filter_type 值才能进入此分支"
      }
    ],
    "execution_trace_compressed": [
      "png_read_info() at pngread.c:100-145",
      "png_read_update_info() at pngread.c:200-220",
      "png_read_row() at pngread.c:300-342"
    ],
    "branch_coverage_summary": {
      "total_branches": 120,
      "covered_branches": 78,
      "uncovered_critical_branches": 12
    },
    "suggestion": "当前输入未触发 filter_type=PNG_FILTER_SUB 分支，建议构造包含 SUB filter 的 IDAT chunk"
  }
}
```

---

## 6. 迭代验证循环（模仿 FuzzingBrain 的 do_pov 循环）

### 6.1 单候选验证流程

```mermaid
flowchart TD
    CAND["候选卡片<br/>vuln_candidate"] --> PROMPT["构造攻击导向 prompt<br/>（attack_hint + 代码片段）"]

    subgraph ITER["迭代循环（最多 5 轮）"]
        direction TB
        PROMPT --> GEN["LLM 生成 harness / seed"]
        GEN --> BUILD["编译"]
        BUILD --> BUILD_OK{"编译成功？"}
        BUILD_OK -- "失败" --> FIX["修复 harness<br/>（不消耗迭代次数）"]
        FIX --> GEN
        BUILD_OK -- "成功" --> RUN["运行 fuzzer"]
        RUN --> CRASH{"崩溃？"}
        CRASH -- "ASAN/MSAN/UBSAN" --> CONFIRMED["✅ confirmed<br/>提取签名，保存 PoV"]
        CRASH -- "无崩溃" --> COV2["提取覆盖率路径反馈"]
        COV2 --> ROUND{"已达 5 轮？"}
        ROUND -- "否" --> FEEDBACK["覆盖率反馈 +<br/>'尝试不同代码路径'<br/>注入 LLM 对话上下文"]
        FEEDBACK --> GEN
        ROUND -- "是" --> INCON["⚠️ inconclusive<br/>进入 cooling，降权"]
    end
```

### 6.2 与 FuzzingBrain 的对比

| 维度 | FuzzingBrain | TianHeng 改造 |
|------|-------------|------------|
| 攻击输入 | LLM 生成 Python → 产出 blob 二进制 | LLM 写 C harness + 生成 seed 文件 |
| 每轮输入数 | AS0: 5 个 blob (x1~x5)；XS0: 1 个 | 每轮 1 个 harness + N 个 seed |
| 迭代上限 | 5 轮 | 5 轮（`max_validation_rounds`） |
| 覆盖率反馈 | LLVM profdata → 函数级路径 → 200行压缩 | 同，但还需要与候选的 key_code_path 交叉分析 |
| 崩溃检测 | 正则匹配 ASAN/MSAN/UBSAN/TSAN 关键词 | 同（已有） |
| 找到后继续 | 继续找下一个（最多 5 个 unique） | 标记 confirmed，继续验证下一个候选 |
| 模型 fallback | Claude → GPT → Gemini 链式 | 当前单模型，计划支持 fallback |

### 6.3 对话上下文维护

模仿 FuzzingBrain 的 `messages` 列表模式——跨迭代保持对话历史：

```python
messages = [
    {"role": "system", "content": VULN_HUNT_SYSTEM_PROMPT},
    {"role": "user", "content": initial_attack_prompt},   # 含 attack_hint + 代码
]

for iteration in range(1, MAX_ITERATIONS + 1):
    # LLM 生成 harness/seed
    response = llm_call(messages)
    messages.append({"role": "assistant", "content": response})

    # 编译 & 运行
    crash, output = run_fuzzer(...)

    if crash:
        save_pov(...)
        break
    else:
        # 覆盖率反馈追加到对话
        coverage_feedback = extract_coverage_paths(...)
        feedback_msg = f"未触发崩溃。\n\n{coverage_feedback}\n\n请调整输入，尝试覆盖不同代码路径。"
        messages.append({"role": "user", "content": feedback_msg})
```

---

## 7. 评分与选择（风险优先）

### 7.1 目标排序

主排序以风险为主，覆盖率为辅：
- `vuln_likelihood`、`exploitability`、`reachability_confidence` 主导（0.88 权重）。
- `coverage_gap`、`complexity_depth`、`api_relevance` 仅参考（0.12 权重）。

```
score_total = 0.45 * vuln_likelihood
            + 0.25 * exploitability
            + 0.18 * reachability_confidence
            + 0.05 * coverage_gap
            + 0.04 * complexity_depth
            + 0.02 * api_relevance
            + 0.01 * consumer_order_support
            - recent_yield_penalty
```

### 7.2 验证反馈调权

每轮验证后根据结果调整候选优先级：

| 验证结果 | 调整 |
|----------|------|
| confirmed | 标记完成，不再验证 |
| inconclusive + 覆盖率有增长 | priority += 0.1（值得再试） |
| inconclusive + 覆盖率无增长 | priority -= 0.2，进入 cooling |
| rejected（5 轮无崩溃）| priority = 0，cooling 24h |
| 同签名重复 | 直接 rejected，不消耗轮次 |

### 7.3 评分可解释

每个候选必须落盘：
- `score_total` + `score_breakdown`
- `penalty_reason`
- `validation_history[]`（每轮的覆盖率变化和结论）

`decision_trace.jsonl` 必含：
- `choose_candidate`：为什么选这个候选
- `choose_seed`：为什么用这个 seed 策略
- `coverage_feedback`：覆盖率反馈内容
- `strategy_delta`：策略调整说明

---

## 8. 与现有 TianHeng 功能的关系

### 8.1 保留
- 现有 `synthesize/build/run/coverage/crash-triage/repair` 执行能力。
- k8s worker、日志、工件管理、API 主体路径。

### 8.2 升级

| 现有阶段 | 升级内容 |
|----------|---------|
| `analysis` | 安全审计输出 `security_evidence[]`（✅ 已实现） |
| `analysis` 扩展 | analysis 结束后落盘 `vuln_candidates.json` 候选工件（✅ 已实现） |
| `vuln_hunt` 内部子阶段 | plan 开头刷新候选、写 `vuln_hunt_summary.md` / `vuln_hunt_events.jsonl`（✅ 已实现，外部 stage 名暂不变） |
| `plan` | 从 coverage-first 改为 candidate-first，显式读取 `vuln_candidates.json`（✅ 已实现） |
| `synthesize` | hint 注入漏洞路径（✅ 已实现 `vuln_hint_lines` + `attack_hint` 候选透传） |
| `seed_generation` | 注入 `VULN_SEED_GUIDANCE` + `VULN_DICTIONARY_TOKENS`（✅ 已实现） |
| `run` | 运行后提取细粒度覆盖率数据（新增） |
| `coverage_analysis` | 增加代码路径级反馈输出（新增） |
| `improve_harness` | 接收覆盖率路径反馈，针对性调整（新增） |

### 8.3 清理
- 移除冲突旧契约（旧字段透传、旧 seed 键、静默降级）。
- 所有降级必须可观测（禁止 silent fallback）。

---

## 9. 上下文与状态管理

### 9.1 双文件上下文继续沿用
- `fuzz/context/control_context.json`：调度硬参数。
- `fuzz/context/workflow_context.json`：业务状态与决策上下文。

### 9.2 新增文件

| 文件 | 作用 | 产出阶段 |
|------|------|---------|
| `fuzz/vuln_candidates.json` | 候选清单（含 attack_hint） | `analysis` / `crash-triage` / `crash-analysis` |
| `fuzz/validation_results.json` | 验证结果记录 | `_node_coverage_analysis` |
| `fuzz/coverage_feedback.json` | 细粒度覆盖率反馈 | `_node_run` 后处理 |
| `fuzz/signature_clusters.json` | 签名去重库 | `_node_crash_triage` |

### 9.3 业务字段归一
- 新增命名空间：`security_*`、`vuln_*`、`candidate_*`、`validation_*`
- 保证跨阶段保真，不回退旧 payload 透传。

---

## 10. Skill 合同（用 SKILL.md 替代 Prompt 约束）

### 10.0 为什么用 Skill 而不是 Prompt

> 参考：[Agent Skills 规范](https://agentskills.io/specification)、[Anthropic Skill 最佳实践](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)

| 维度 | Prompt 约束 | Skill 约束 |
|------|-------------|------------|
| 加载时机 | 每次请求都在上下文中（浪费 token） | 按需加载：空闲时只占 ~100 token（name + description） |
| 约束方式 | 自然语言描述规则，AI 可能忽略 | 结构化：frontmatter 声明 + 工作流步骤 + 校验脚本 |
| 输出控制 | 靠 prompt 中的模板和示例 | 可以挂 `scripts/validate.py` 做机器校验 |
| 维护 | 修改一个大 prompt 文件 | 每个 skill 独立目录，独立版本 |
| 扩展性 | prompt 越长质量越差 | 100+ skill 共存不互相干扰 |
| 工具限制 | 全局设置 | 每个 skill 可声明 `allowed-tools`（只允许只读命令等） |

**结论**：`vuln-hunt` 的策略菜单和输出校验规则应该写成 SKILL.md，而不是嵌在 prompt 模板里。策略菜单是 skill body 的自然语言部分（AI 自由参考），输出校验是 skill 的 `Required outputs` + `Acceptance checklist`（硬性约束）。

### 10.1 `vuln_hunt` Skill 设计

**文件路径**：`opencode_skills/vuln_hunt/SKILL.md`

**设计原则**：
- **策略自由**：策略菜单写在 skill body 中作为建议，AI 自主选用
- **输出严格**：通过 `Required outputs` + `Acceptance checklist` 做硬性约束
- **渐进加载**：skill body < 500 行，详细参考材料放 `references/` 子目录
- **校验脚本**：`scripts/validate_candidates.py` 做结构化校验，不依赖 AI 自查

#### SKILL.md 草案

```markdown
---
name: vuln-hunt
description: >
  Discover vulnerability candidates from analysis context using AI-chosen
  attack strategies. Produces vuln_candidates.json with attack_hint for
  each candidate. Use in the vuln-hunt stage after analysis.
compatibility: opencode
metadata:
  stage: vuln-hunt
  owner: tianheng
---

## What this skill does
从 analysis_context.json 中发现漏洞候选，为每个候选生成攻击策略提示（attack_hint），
输出结构化候选清单供下游验证引擎消费。

## When to use this skill
Use in the dedicated `vuln-hunt` stage, after `analysis` and before `plan`.

## Required inputs
- `fuzz/analysis_context.json`（含 `security_evidence[]`、调用图、目标分类）
- `fuzz/target_analysis.json`
- repository source tree (read-only)

## Required outputs
- `fuzz/vuln_candidates.json`：候选清单文档（strict JSON object，含 `candidates[]`）
- `fuzz/vuln_hunt_summary.md`：发现摘要（含使用了哪些策略、为什么选这些候选）

## Strategy menu（建议，AI 自由选用组合）

以下策略供参考，你可以只用一种、组合多种、或发明新策略。系统不关心你的过程，只校验产出。

### 证据直攻
从 `security_evidence[]` 中置信度 ≥ 0.6 的条目直接生成候选。
适用于 analysis 阶段已发现明确安全信号的情况。

### 漏洞分类扫描
按漏洞类型（OOB / UAF / 整数溢出 / 格式化字符串 / 命令注入 / 路径遍历）
逐类扫描 target_analysis 中的函数，寻找匹配特征。

### 高危函数筛选
从 target_analysis 中按 vuln_likelihood 排序，取 top-k 函数，
读源码做代码级分析判断是否有可触发漏洞。

### 调用路径构造
沿调用图从外部输入入口追踪到危险函数，
识别数据流中未校验的 trust boundary 跨越。

### 综合审计
结合以上多种视角做全面审计，适用于代码库较小或候选较少时。

## Candidate schema（每个候选必须符合）

每个候选是一个 JSON 对象，必须包含以下字段：

| 字段 | 类型 | 要求 |
|------|------|------|
| `candidate_id` | string | 幂等主键，格式 `<signal_type>_<seq>`（如 `mem_oob_001`） |
| `target_api` | string | 具体函数名（非文件路径、非描述文字） |
| `target_file` | string | 源文件路径 |
| `signal_type` | string | 漏洞分类（如 `mem_oob_candidate`） |
| `evidence` | array | 至少一条，每条含 `evidence_id` 引用 |
| `attack_hint` | object | 见下方 attack_hint schema |
| `vuln_likelihood` | float | 0.0-1.0 |
| `exploitability` | float | 0.0-1.0 |
| `reachability_confidence` | float | 0.0-1.0 |
| `priority` | float | 0.0-1.0，降序排列 |

### attack_hint schema

| 字段 | 类型 | 要求 |
|------|------|------|
| `trigger_condition` | string | 触发条件的自然语言描述 |
| `key_code_path` | array[string] | 具体函数名列表（如 `["png_read_row", "memcpy"]`），不是描述文字 |
| `boundary_values` | array[string] | 具体触发值（如 `["width=0xFFFFFFFF"]`），不是占位符 |
| `vuln_category` | string | 漏洞类型（如 `heap-buffer-overflow`） |
| `sanitizer_hint` | string | 建议 sanitizer（`address` / `memory` / `undefined`） |

## Command policy
- Allowed: read-only commands (`find`, `rg`, `grep`, `cat`, `head`, `tail`, `ls`).
- Forbidden: build / execute / mutation commands.

## Acceptance checklist
- [ ] `fuzz/vuln_candidates.json` is a valid JSON object with non-empty `candidates[]`
- [ ] Every candidate has a unique `candidate_id`
- [ ] Every candidate references at least one `evidence_id` from analysis_context
- [ ] Every `attack_hint.key_code_path` contains concrete function names (not descriptions)
- [ ] Every `attack_hint.boundary_values` contains concrete values (not placeholders like "TBD")
- [ ] Candidates are sorted by `priority` descending
- [ ] At least 3 candidates (or all available if source has fewer viable targets)
- [ ] `fuzz/vuln_hunt_summary.md` exists and lists strategies used

## Degraded mode
- If `security_evidence[]` is empty, continue using target_analysis + source code reading.
- If fewer than 3 viable candidates found, output all found and note in summary.
- Record degraded reason in output, do not silently skip.

## Done contract
- Write `fuzz/vuln_candidates.json` into `./done`.
```

#### 校验脚本：`scripts/validate_candidates.py`

```python
"""Machine-verifiable output validator for vuln_hunt skill.
Called by _node_vuln_hunt after AI produces candidates.
Returns structured feedback for retry if validation fails."""

def validate(candidates: list[dict], evidence_ids: set[str]) -> dict:
    errors = []
    for i, c in enumerate(candidates):
        cid = c.get("candidate_id", f"candidate[{i}]")
        # evidence_ref check
        if not c.get("evidence") or not any(
            e.get("evidence_id") in evidence_ids for e in c["evidence"]
        ):
            errors.append(f"{cid}: 没有引用任何有效的 evidence_id")
        # attack_hint completeness
        hint = c.get("attack_hint", {})
        for field in ("trigger_condition", "key_code_path", "boundary_values"):
            if not hint.get(field):
                errors.append(f"{cid}: attack_hint 缺少 {field}")
        # concrete_path check
        if hint.get("key_code_path") and any(
            len(p.split()) > 3 for p in hint["key_code_path"]
        ):
            errors.append(f"{cid}: key_code_path 应为函数名列表，不是描述文字")
        # concrete_values check
        if hint.get("boundary_values") and any(
            v.upper() in ("TBD", "TODO", "PLACEHOLDER") for v in hint["boundary_values"]
        ):
            errors.append(f"{cid}: boundary_values 含占位符")
    # min_candidates
    if len(candidates) < 3:
        errors.append(f"只有 {len(candidates)} 个候选，至少需要 3 个")
    # priority_ordered
    priorities = [c.get("priority", 0) for c in candidates]
    if priorities != sorted(priorities, reverse=True):
        errors.append("候选未按 priority 降序排列")
    return {"valid": len(errors) == 0, "errors": errors}
```

### 10.2 Skill vs Prompt 在 TianHeng 各阶段的对比

```mermaid
flowchart LR
    subgraph PROMPT["现有 Prompt 驱动阶段"]
        direction TB
        P1["opencode_prompts.md\n（大模板，所有规则混在一起）"]
    end
    subgraph SKILL["Skill 驱动阶段"]
        direction TB
        S1["analysis/SKILL.md ✅ 已有"]
        S2["plan/SKILL.md ✅ 已有"]
        S3["synthesize/SKILL.md ✅ 已有"]
        S4["seed_generation/SKILL.md ✅ 已有"]
        S5["vuln_hunt/SKILL.md 🆕 新增"]
    end

    P1 -.->|"渐进迁移：\n约束从 prompt 搬到 skill"| SKILL

    style S5 fill:#ffd,stroke:#aa0
```

**现有模式**：每个阶段已经有 SKILL.md（analysis、plan、synthesize、seed_generation 等），但 `opencode_prompts.md` 仍然包含大量重复约束。

**改进方向**：
1. **新阶段直接用 skill**：`vuln_hunt` 的约束全部写在 SKILL.md 中，不在 prompt 模板中重复
2. **校验逻辑放 scripts/**：`validate_candidates.py` 做机器校验，不依赖 AI 自查
3. **prompt 只做上下文注入**：`opencode_prompts.md` 中 `vuln_hunt` 相关部分只负责注入 `analysis_context` 数据，不重复约束规则

### 10.3 `plan` 合同（已有 SKILL.md，增量更新）
- 在现有 `plan/SKILL.md` 的 `Required inputs` 中增加 `fuzz/vuln_candidates.json`
- 在 `Workflow` 中增加：读取 `vuln_candidates.json` 时按 `priority` 选取 top-k
- 在 `Constraints` 中增加：风险优先，coverage 仅参考

### 10.4 `synthesize` 合同（已有 SKILL.md，增量更新）
- 在现有 `synthesize/SKILL.md` 的 `Required inputs` 中增加 `attack_hint`（来自候选）
- 在 `Workflow` 中增加：当有 `attack_hint` 时，harness 设计瞄准 `key_code_path`
- seed 设计包含 `boundary_values` 中的边界条件

### 10.5 安全渲染
- 所有节点必须走安全渲染路径。
- 模板异常只降级，不中断；必须写：`prompt_render_degraded`、`prompt_render_issue`。

---

## 11. 实施阶段

```mermaid
flowchart LR
    subgraph done["✅ 已完成"]
        PRE["前置工作<br/>评分切换 0.88 / security_evidence<br/>suggested_families / vuln_hint<br/>seed guidance / dict_parse_error fix"]
    end
    subgraph p1["Phase 1"]
        P1A["1a: 最小闭环<br/>_node_vuln_hunt（策略自由+输出校验）<br/>+ candidate-first plan"]
        P1B["1b: 覆盖率反馈<br/>LLVM profdata<br/>函数级路径提取"]
        P1C["1c: 迭代循环<br/>单候选 5 轮验证<br/>对话上下文维护"]
        P1A --> P1B --> P1C
    end
    subgraph p2["Phase 2"]
        P2A["DB 状态机<br/>候选表 + 状态流转<br/>调度器轮询"]
    end
    subgraph p3["Phase 3"]
        P3A["并发验证<br/>多 worker 消费<br/>模型 fallback"]
    end
    PRE --> P1A
    P1C --> P2A --> P3A
```

### Phase 1a：最小可验证闭环（已完成）

**目标**：跑通"候选生成 → 目标选择 → fuzz 验证"单趟闭环。

- 已按最小闭环落地：
  - `analysis` 基于 `analysis_context.json.analysis_evidence.vuln_candidate_inventory` 生成 `fuzz/vuln_candidates.json`
  - `analysis -> vuln-hunt -> plan` 显式 stage 路由已接入；`plan` 在直接 resume 时仍可按需刷新 hunt 候选
  - `_node_vuln_hunt` 刷新候选并记录 `fuzz/vuln_hunt_summary.md` / `fuzz/vuln_hunt_events.jsonl`
  - `plan` 显式读取 `fuzz/vuln_candidates.json`，执行 candidate-first 排序
  - `selected_targets.json` 保留 `vuln_candidate_id`、`vuln_candidate_priority`、`attack_hint`
  - `synthesize` 继续沿用漏洞 hint 注入
- `vuln-hunt` 已暴露为独立 k8s/resume stage；`vuln_hunt` resume alias 会归一化为 `vuln-hunt`。
- 不引入新队列抽象，不改变 k8s Job 调度模型。
- 保持单任务线性主链。

### Phase 1b：细粒度覆盖率反馈

**目标**：让 LLM 知道"跑了哪些路径、卡在哪里"。

- `_node_run` 后处理增加 LLVM coverage 提取：
  - 编译时带 `-fprofile-instr-generate -fcoverage-mapping`
  - 运行后 `llvm-profdata merge` + `llvm-cov export`
  - 解析为函数级覆盖列表
- 输出 `fuzz/coverage_feedback.json`。
- `improve_harness` 阶段注入覆盖率路径反馈。
- 压缩到 200 行以内。

### Phase 1c：迭代验证循环（已完成基础闭环）

**目标**：单候选多轮迭代验证（模仿 FuzzingBrain 的 do_pov 循环）。

- coverage loop 已增加"候选验证进展"语义。
- coverage plateau / seed degraded / harness feedback 会写入 `vuln_hunt_events.jsonl`。
- 当前 active candidate 会更新 `attempt_count`、`validation_status`、`last_result`。
- 每个候选默认最多 5 轮迭代，超过后从 `cooling` 升级为 `exhausted`。
- `coverage-analysis` 在 replan 场景下优先回到 `vuln-hunt`，再进入 `plan/materialize`。

### Phase 2：状态解耦（DB 驱动）

- 落地 `vuln_candidates` 表：`pending → validating → confirmed/rejected/inconclusive → cooling`。
- 调度器轮询 `pending` 候选（按 priority），驱动验证引擎消费。
- 验证结果回写候选状态和覆盖率 delta。
- 不引入外部 MQ，用 DB 状态机实现队列语义。

### Phase 3：执行落地

- 验证引擎并行消费（DB 轮询式），并发与集群容量绑定：
  - `SHERPA_VULN_MAX_CONCURRENT_VALIDATIONS`（默认 `2`，建议上限 `3`）。
- retry / cooling 策略上线。
- seed/repair 与候选状态联动。
- 多模型 fallback 支持（Claude → GPT → Gemini）。

### Phase 4：可观测与治理

- API 字段扩展（兼容新增）。
- trace、snapshot、评分、降级全量可见。
- 监控告警（空转、重复签名、无增量循环）。
- 验证进度看板（候选完成率、平均迭代次数、崩溃发现率）。

### Phase 5：收口清理

- 删除 legacy 透传与冲突契约。
- 更新文档、测试与运维手册。
- 统一开关与默认策略。

---

## 12. 验收标准

- 主引擎可从代码库中自主发现 ≥3 个 unique 候选（含 attack_hint）。
- 候选验证端到端链路稳定（可追踪、可复现、可回写）。
- 覆盖率反馈达到函数级精度，LLM 据此调整后覆盖率 ≥10% 增长。
- 同签名重复空转显著下降（签名去重命中率 ≥80%）。
- 单候选多轮迭代验证可用（≤5 轮收敛或放弃）。
- 在长任务中可稳定观察"发现 → 验证 → 反馈 → 继续发现"闭环。
- 全量回归测试通过，且无旧契约残留。

---

## 13. 风险与控制

### 13.1 风险
- 状态复杂度提升，跨阶段一致性风险变大。
- 并发提升后资源竞争与成本上升。
- 候选质量不足时可能导致验证资源浪费。
- 覆盖率提取依赖 LLVM 工具链，非 C/C++ 项目需要适配。
- 细粒度覆盖率数据量大，context 管理不当会导致 LLM 输出质量下降。
- 现有基础设施以 k8s Job 为主，不适合直接套重型 MQ 语义。

### 13.2 控制
- 幂等主键 + 签名去重 + 限流冷却。
- 多维预算硬阈值（time/token/cost/round）。
- 覆盖率反馈压缩到 200 行以内，避免 context 爆炸。
- 强制结构化降级与死信隔离。
- 关键阶段回滚开关（策略层可回退，执行层不回退）。
- `vuln-hunt` 执行策略：首轮全量候选生成 + 验证反馈触发增量更新。
- `detectability_confidence` 定义为"当前 fuzzer/sanitizer 组合对该候选的可触发可观测信心"。
- 并发验证上限与集群容量绑定，默认 2。

---

## 14. 推荐默认配置

```env
SHERPA_VULN_HUNTING_ENABLED=1
SHERPA_VULN_SCORE_MODE=risk_first_v1
SHERPA_VULN_INTERNAL_API_MIN_SCORE=0.75
SHERPA_VULN_MIN_EVIDENCE_CONFIDENCE=0.45
SHERPA_VULN_TOPK=24
SHERPA_VULN_MAX_CONCURRENT_VALIDATIONS=2
SHERPA_VULN_MAX_ITERATIONS_PER_CANDIDATE=5
SHERPA_VULN_COVERAGE_FEEDBACK_ENABLED=1
SHERPA_VULN_COVERAGE_FEEDBACK_MAX_LINES=200
SHERPA_VULN_COOLING_HOURS=24
```

---

## 15. 已完成工作（当前进度）

以下改动已合入 `sherpa-improvements` 和 `dev` 分支：

| 改动 | 性质 | 状态 |
|------|------|------|
| 评分权重切换到 0.88 vuln-dominant（`0.45 vuln + 0.25 exploit + 0.18 reach`） | 前置：评分基础 | ✅ 已完成 |
| `required_families` → `suggested_families` 全链路重命名 | 前置：去硬约束 | ✅ 已完成 |
| analysis prompt 增加安全审计输出（`security_evidence[]`） | 前置：数据源 | ✅ 已完成 |
| `_load_security_evidence_list()` 严格合约 | 前置：数据源 | ✅ 已完成 |
| synthesize hint 注入漏洞路径（`vuln_hint_lines`） | 前置：攻击引导 | ✅ 已完成 |
| `VULN_SEED_GUIDANCE` + `VULN_DICTIONARY_TOKENS` | 前置：种子引导 | ✅ 已完成 |
| dict_parse_error 立即 exhaust target | 前置：Bug 修复 | ✅ 已完成 |
| 降级传播到 decision snapshot | 前置：可观测性 | ✅ 已完成 |
| plan prompt candidate-first 排序声明 | 前置：排序策略 | ✅ 已完成 |
| analysis 落盘 `fuzz/vuln_candidates.json` | Phase 1a | ✅ 已完成 |
| 显式 `vuln-hunt` stage（analysis -> vuln-hunt -> plan） | Phase 1a/2 | ✅ 已完成 |
| plan 显式消费 `fuzz/vuln_candidates.json` | Phase 1a | ✅ 已完成 |
| selected target 透出 `vuln_candidate_id` / `attack_hint` | Phase 1a | ✅ 已完成 |
| coverage feedback 回写 `vuln_hunt_events.jsonl` | Phase 1c | ✅ 已完成 |
| active candidate 状态回写（attempt/status/last_result） | Phase 1c | ✅ 已完成 |
| coverage replan 路由回 `vuln-hunt` | Phase 2/3 | ✅ 已完成 |

**当前状态**：前置工作、显式 `vuln-hunt` stage、candidate-first plan、synthesize attack hint、以及 coverage feedback 驱动候选状态回写已完成。当前是“发现 → 候选 → 目标物化 → fuzz 验证 → 反馈回 hunt”的文件工件闭环版本。

**下一步**：DB 驱动候选表与并发验证引擎仍未实现；该部分属于后续重架构，不在当前文件工件闭环范围内。

---

## 16. 参考项目路径

### 16.1 上游参考仓库
- GitHub：[o2lab/afc-crs-all-you-need-is-a-fuzzing-brain](https://github.com/o2lab/afc-crs-all-you-need-is-a-fuzzing-brain)

### 16.2 本机 clone
- 完整 clone：`references/afc-crs-all-you-need-is-a-fuzzing-brain`
- 当前 commit：`0fba44e`

### 16.3 关键参考文件

| 文件 | TianHeng 对照点 |
|------|-------------|
| `crs/strategy/core/pov_strategy.py` | 迭代验证循环模板 |
| `crs/strategy/jeff/as0_full.py` | 多阶段攻击策略 |
| `crs/strategy/code_analysis/coverage_analyzer.py` | 细粒度覆盖率反馈 |
| `crs/strategy/common/security_analyzer/agent.py` | 安全审计 Agent 设计 |
| `crs/strategy/strategies/xs0_delta_new.py` | 简化版迭代循环 |
| `crs/internal/models/models.go` | 数据契约定义 |
| `crs/internal/executor/task_execution.go` | 任务编排 |
