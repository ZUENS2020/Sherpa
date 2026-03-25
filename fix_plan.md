# Sherpa 问题修复计划

## 问题总览

| # | 问题 | 严重程度 | 优先级 | 状态 |
|---|------|----------|--------|------|
| 1 | Seed Generation 未为所有 Fuzzer 生成 AI Seeds | 中等 | P1 | ✅ 已修复 |
| 2 | K8s Job Timeout 导致 build/run 阶段失败 | 高 | P0 | ✅ 已修复 |
| 3 | OpenCodeHelper 僵尸进程 | 中等 | P2 | ✅ 已修复 |
| 4 | Coverage Plateau 检测后无法有效改善 | 中等 | P2 | 未修复 |
| 5 | Re-build 失败：CMakeCache.txt 路径冲突 | 高 | P0 | ✅ 已修复 |
| 6 | Seed Gen Idle Timeout: activity_watch_paths 缺失 | 高 | P0 | ✅ 已修复 |
| 7 | FastAPI 数据暴露不完整 | 中等 | P1 | ✅ 已修复 |

---

## P0-1：K8s Job Timeout 导致 build/run 阶段失败

### 现象

```
[job xxx] stage build failed (stage_dispatch_exception): k8s_job_timeout -> fallback to plan
[job xxx] stage run failed (stage_dispatch_exception): k8s_job_timeout -> fallback to plan
```

### 根因

| 根因 | 位置 | 说明 |
|------|------|------|
| `k8s_job_timeout` 被 `except Exception` 兜底 | `main.py:3447-3462` | 不区分超时和其他异常，一律 `stage_dispatch_exception → fallback to plan` |
| build 阶段无重试机制 | `main.py:3459` | 仅 `stage == "run"` 可重试，build 超时直接放弃 |
| run 的 `wait_timeout` 不考虑 seed gen 内部 9× 重试 | `main.py:1223-1235` | 预算公式只算 `fuzzer_count × round_budget + grace`，没计入 `max_attempts(3) × max_cli_retries(3)` |

### 修复方案

#### Fix 2-A: 扩展 k8s timeout 重试到 build 阶段

**文件**: `main.py:3459`

```python
# 现在:
if stage == "run" and is_k8s_timeout and run_timeout_retry_count < max_timeout_retries:

# 改为:
if stage in ("run", "build") and is_k8s_timeout and run_timeout_retry_count < max_timeout_retries:
```

#### Fix 2-B: run 阶段 wait_timeout 加入 seed gen 重试安全系数

**文件**: `main.py:_k8s_stage_wait_timeout_sec()`

在 `run_base` 计算后增加:

```python
seed_gen_retry_multiplier = int(os.environ.get("SHERPA_SEED_GEN_RETRY_MULTIPLIER", "3"))
run_base = run_base * seed_gen_retry_multiplier
```

- 默认 3×（`max_attempts=3` 的最坏情况）
- 通过 `SHERPA_SEED_GEN_RETRY_MULTIPLIER` 环境变量可调

#### Fix 2-C: 精确异常分类

**文件**: `main.py:3447-3462`

`k8s_job_timeout` 时 `stage_fail_reason` 应为 `"k8s_job_timeout"` 而非 `"stage_dispatch_exception"`。

---

## P0-2：Re-build 失败 — CMakeCache.txt 路径冲突 ✅ 已修复

### 现象

```
CMake Error: The current CMakeCache.txt directory
  /shared/output/zlib-7161af14/.repro_crash/workdir/fuzz/build-work/CMakeCache.txt
is different than the directory
  /shared/output/zlib-7161af14/fuzz/build-work
where CMakeCache.txt was created.
```

### 根因

| 根因 | 位置 | 说明 |
|------|------|------|
| `shutil.copytree` 复制了 `build-work` 目录 | `workflow_graph.py:8374` | 包含 `CMakeCache.txt`，内部硬编码了原始路径 |
| clone 到新路径但沿用旧缓存 | `workflow_graph.py:8347` | `clone_root = repro_workspace / "workdir"`，路径变了但 CMake cache 还指向老路径 |

### 修复内容

**文件**: `workflow_graph.py:8374` 和 `workflow_graph.py:8626`（两处 copytree）

```python
shutil.copytree(
    source_fuzz,
    dest_fuzz,
    ignore=shutil.ignore_patterns(
        "build-work", "CMakeFiles", "out", "__pycache__", "*.o", "*.a",
    ),
)
```

---

## P1-1：Seed Generation 未为所有 Fuzzer 生成 AI Seeds

### 现象

- decode fuzzer: 22 AI seeds, 23 radamsa seeds ✓
- fread_file_func: AI seeds = 0, radamsa seeds = 0 ✗
- fseek64_file_func: AI seeds = 0, radamsa seeds = 0 ✗

### 根因

| 根因 | 位置 | 说明 |
|------|------|------|
| Seed gen 串行执行 | `fuzz_unharnessed_repo.py:2342-2348` | `for bin_path in bins` 逐个调 `_pass_generate_seeds`，第一个耗时过长则后续无时间 |
| idle timeout 不受 `seed_generation_timeout_sec` 控制 | `fuzz_unharnessed_repo.py:4096` | 固定 300s 或环境变量，不跟随动态预算 |
| 默认 idle timeout = 900s | `codex_helper.py:1021` | AI 卡住 900s 才被杀，但 `seed_generation_timeout_sec` 无法覆盖 |
| 异常被静默捕获 | `fuzz_unharnessed_repo.py:2346-2348` | `except HarnessGeneratorError` 只 warning，不中断不报告 |

### 修复方案

#### Fix 1-A: 传递 idle_timeout_override ✅ 已修复（Fix 6 一并解决）

见下方 P1-2。

#### Fix 1-B: 按 fuzzer 数量分配时间预算

**文件**: `fuzz_unharnessed_repo.py:2342` 附近

```python
total_seed_budget = seed_generation_timeout_sec or 1800
per_fuzzer_budget = max(300, total_seed_budget // len(bins))
for bin_path in bins:
    self.seed_generation_timeout_sec = per_fuzzer_budget
    try:
        self._pass_generate_seeds(bin_path.name)
    except HarnessGeneratorError as e:
        print(f"[!] Seed generation failed ({bin_path.name}): {e}")
```

#### Fix 1-C: 改善失败报告

- 记录哪些 fuzzer 的 seed gen 失败
- 在 `run_summary.json` 加 `seed_gen_failed_fuzzers` 字段
- 让 coverage-analysis 知道哪些 fuzzer 缺种子

---

## P0-3：Seed Gen Idle Detection 缺少 activity_watch_paths ✅ 已修复

### 现象

Seed generation 阶段 AI 被 idle timeout 误杀。AI 实际在思考并写入 corpus 文件，但 idle 检测只看 stdout 输出，300s 无 stdout 即判定 idle 并 kill。

### 根因

| 根因 | 位置 | 说明 |
|------|------|------|
| seed gen 的 `run_codex_command` 调用缺少 `activity_watch_paths` | `fuzz_unharnessed_repo.py:4088-4122` | 普通 stage 会传 `activity_watch_paths` 让 idle 检测监控文件系统变化，但 seed gen 阶段遗漏了 |
| idle 检测逻辑依赖三种信号 | `codex_helper.py:1532-1566` | `git diff` + `activity_watch_paths` + stdout。缺 `activity_watch_paths` 时只靠 stdout，AI 思考期间无 stdout 即被判定 idle |

### 修复内容

**文件**: `fuzz_unharnessed_repo.py` seed gen 的 `run_codex_command` 调用

```python
# 添加 activity_watch_paths 监控 corpus 目录变化
stdout = self.patcher.run_codex_command(
    instructions,
    additional_context="\n\n".join(additional_context_parts),
    stage_skill="seed_generation",
    activity_watch_paths=[str(corpus_dir), f"fuzz/corpus/{fuzzer_name}"],
    **patcher_kwargs,
)
```

同时将 idle timeout 恢复为简单的环境变量默认值:
```python
patcher_kwargs["idle_timeout_override"] = max(60, seed_idle_timeout)
```

**效果**: AI 写入 corpus 文件时文件系统活动会被检测到，不再被误杀。idle timeout 仅在 AI 真正停止所有活动时触发。

---

## P2-A：OpenCodeHelper 僵尸进程 ✅ 已修复

### 现象

大量 defunct opencode 进程。

### 根因

`codex_helper.py:1489-1493` 中 `proc.kill()` 后没有调用 `proc.wait()`。

### 修复内容

**文件**: `codex_helper.py:1494-1497`

```python
if proc.poll() is None:
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=4)
    except Exception:
        pass
```

---

## P2-B：Coverage Plateau 检测后无法有效改善

### 现象

- Coverage 在 200 (decode), 182 (fread_file_func) 处停止
- fread_file_func 的 seed novelty 仅 0.35

### 根因

| 根因 | 位置 | 说明 |
|------|------|------|
| improve-harness 的 in_place 模式缺乏结构化种子补种 | `workflow_graph.py:7632-7666` | 只让 AI 改文件，没有定向种子策略 |
| archive-container 类 fuzzer 只靠 repo examples | — | AI 不擅长生成二进制 archive 格式种子 |
| `_filter_seed_corpus` 去重过于激进 | `fuzz_unharnessed_repo.py:3766-3781` | strict mode 同 shape 只留 1 个 |

### 修复方案

#### Fix 4-A: improve-harness 增加 seed-only 改进路径

在 `improve-harness (in_place)` 中：
1. 读 `run_summary.json` 中 `missing_families`
2. 仅对缺失族触发定向 seed gen（不改 harness 代码）
3. 直接 `build → run`

#### Fix 4-B: archive-container 特殊处理

- 优先从 repo 挖掘更多 test fixtures（不限 `corpus/` 目录）
- radamsa 对 repo examples 多轮变异（当前只 1 轮）
- 降低 AI seed gen 权重

#### Fix 4-C: 放宽 strict filter 的 shape 去重

```
strict mode 同 shape: 1 → 2
required family cap: 3 → 5
```

#### Fix 4-D: plateau 检测参数动态化

```
当前: SHERPA_RUN_PLATEAU_IDLE_GROWTH_SEC=180, SHERPA_RUN_PLATEAU_PULSES=3
建议: 覆盖率 > 500 时动态延长检测窗口
```

---

## P1-3：FastAPI API 数据暴露不完整 ✅ 已修复

### 现象

后端存储了丰富的 workflow 状态、恢复状态、per-fuzzer 性能指标，但 API 只暴露了基础字段（status, error, timestamps），前端/监控无法看到运行时细节。

### 修复内容

#### Fix 7-A: 暴露 workflow/resume/cancel 追踪字段

**文件**: `main.py` — 新增 `_enrich_job_view()` 函数

在 `/api/task/{id}` 和 `/api/tasks` 的响应中添加:

| 字段 | 说明 |
|------|------|
| `k8s_phase` | K8s Pod 阶段 |
| `cancel_requested` / `last_cancel_requested_at` | 取消状态 |
| `workflow_active_step` / `workflow_last_step` / `workflow_last_step_ts` | 流水线可见性 |
| `parent_id` | 子任务→父任务导航 |
| `recoverable` / `resume_attempts` / `resume_error_code` / `resume_from_step` | 恢复状态 |
| `last_resume_reason` / `last_resume_requested_at` / `last_resume_started_at` / `last_resume_finished_at` | 恢复历史 |
| `last_interrupted_at` | 中断时间戳 |
| `request` | 原始请求 payload |

#### Fix 7-B: 暴露 per-fuzzer 实时性能指标

**架构**: workflow_graph.py → `[wf-metrics]` JSON log line → main.py 解析 → `_JOBS` → API

**新增 `[wf-metrics]` 结构化日志**（`workflow_graph.py`）:

在 `_node_run` 和 `_node_coverage_analysis` 完成时 emit:

```json
{
  "ts": 1711234567,
  "stage": "run",
  "fuzzers": {
    "decode_fuzzer": {
      "final_cov": 342, "final_ft": 1205,
      "final_execs_per_sec": 8923, "final_corpus_files": 156,
      "plateau_detected": false, "crash_found": false,
      "seed_quality": {...}
    },
    "parse_fuzzer": {
      "final_cov": 187, "final_ft": 623,
      "final_execs_per_sec": 12340, ...
    }
  },
  "max_cov": 342, "max_ft": 1205,
  "total_execs_per_sec": 21263,
  "coverage_history": [...],
  "coverage_source_report": {...}
}
```

**API 新增字段** (在 `/api/task/{id}` 和 `/api/tasks` 中):

| 字段 | 说明 |
|------|------|
| `fuzz_fuzzers` | `{fuzzer_name: {cov, ft, execs/s, corpus, plateau, crash, seed_quality}}` |
| `fuzz_max_cov` / `fuzz_max_ft` | 全部 fuzzer 的最大覆盖率/特征数 |
| `fuzz_total_execs_per_sec` | 所有 fuzzer 的总执行速度 |
| `fuzz_crash_found` | 是否发现 crash |
| `fuzz_coverage_history` | 覆盖率历史（每轮数据） |
| `fuzz_coverage_source_report` | llvm-cov 源码级覆盖报告 |
| `fuzz_coverage_loop_round` / `fuzz_coverage_loop_max_rounds` | 覆盖率循环进度 |
| `fuzz_coverage_plateau_streak` | 平台期连续轮次 |
| `fuzz_coverage_seed_profile` / `fuzz_coverage_quality_flags` | 种子质量信息 |

#### Fix 7-C: 增强 run_summary.json 日志文件

**文件**: `workflow_summary.py`

新增 `fuzz_performance` 块写入 `run_summary.json`:

```json
{
  "fuzz_performance": {
    "fuzzers": {"decode_fuzzer": {...}, "parse_fuzzer": {...}},
    "aggregate": {"max_cov": 342, "max_ft": 1205, "total_execs_per_sec": 21263, "fuzzer_count": 2},
    "coverage_loop_round": 3, "coverage_loop_max_rounds": 5,
    "coverage_plateau_streak": 1, "coverage_quality_flags": ["seed_family_undercovered"]
  }
}
```

Markdown 摘要 (`run_summary.md`) 新增 "Fuzz Performance (Aggregate)" 和增强的 per-fuzzer 表。

---

## 修复优先级与依赖

```
P0-1 (k8s timeout 重试/安全系数)  ← ✅ 已修复
P0-2 (re-build CMakeCache 冲突)   ← ✅ 已修复
P0-3 (seed gen activity_watch)    ← ✅ 已修复
         ↓
P1-1 (seed gen 时间分配)          ← ✅ 已修复
P1-3 (API 数据暴露)              ← ✅ 已修复
         ↓
P2-A (僵尸进程)                   ← ✅ 已修复
P2-B (plateau 改善)               ← 长期，依赖 P1 生效后有更好的种子数据
```

## 涉及文件清单

| 文件 | 涉及 Fix | 状态 |
|------|----------|------|
| `harness_generator/src/langchain_agent/main.py` | P0-1, P1-3 (API enrichment + metrics parsing) | ✅ |
| `harness_generator/src/langchain_agent/workflow_graph.py` | P0-2 (copytree), P2-B (Fix 4-A) | P0-2 ✅ |
| `harness_generator/src/fuzz_unharnessed_repo.py` | P1-1 (Fix 1-B, 1-C), P1-2 (idle timeout) | P1-2 ✅ |
| `harness_generator/src/codex_helper.py` | P2-A (zombie process) | ✅ |
