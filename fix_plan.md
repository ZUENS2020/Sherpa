# Sherpa 问题修复计划

## 问题总览

| # | 问题 | 严重程度 | 现象 |
|---|------|----------|------|
| 1 | Seed Generation 未为所有 Fuzzer 生成 AI Seeds | 中等 | decode 有 22 AI seeds；fread_file_func / fseek64_file_func AI seeds = 0 |
| 2 | K8s Job Timeout 导致 build/run 阶段失败 | 高 | `stage_dispatch_exception: k8s_job_timeout -> fallback to plan` |
| 3 | OpenCodeHelper 进程回收 Bug | 中等 | 大量 defunct opencode 进程 |
| 4 | Coverage Plateau 检测后无法有效改善 | 中等 | Coverage 在 200/182 处停止；seed novelty 仅 0.35 |

---

## P0：K8s Job Timeout 导致 build/run 阶段失败

### 根因分析

| 根因 | 位置 | 说明 |
|------|------|------|
| `k8s_job_timeout` 被 `except Exception` 兜底 | `main.py:3447-3462` | 不区分超时和其他异常，一律 `stage_dispatch_exception → fallback to plan` |
| build 阶段无重试机制 | `main.py:3459` | 条件 `stage == "run"` 排除了 build，build 超时直接放弃 |
| run 的 `wait_timeout` 不考虑 seed gen 的 9× 重试 | `main.py:1223-1235` | run 预算公式只算 `fuzzer_count × round_budget + grace`，没考虑 seed gen 内部重试的时间消耗 |
| `_estimate_run_parallelism` 默认 2 但实际可能是 1 | `main.py:1274-1283` | OOM 重试后强制设为 1，round_count 翻倍但 timeout 没重算 |

### Fix 0-A: 扩展 k8s timeout 重试到 build 阶段

**文件**: `main.py:3459`

```python
# 现在:
if stage == "run" and is_k8s_timeout and run_timeout_retry_count < max_timeout_retries:

# 改为:
if stage in ("run", "build") and is_k8s_timeout and run_timeout_retry_count < max_timeout_retries:
```

- build 超时后同样自动重试 + 延长预算
- 复用现有的 `run_timeout_retry_count` / `run_timeout_budget_sec_override` 机制（改名为 `stage_timeout_retry_count` 更准确）

### Fix 0-B: run 阶段的 wait_timeout 考虑 seed gen 重试倍数

**文件**: `main.py:_k8s_stage_wait_timeout_sec()`

```python
# 在 run_base 计算后增加 seed gen 安全系数:
seed_gen_retry_multiplier = int(os.environ.get("SHERPA_SEED_GEN_RETRY_MULTIPLIER", "3"))
run_base = run_base * seed_gen_retry_multiplier
```

- 默认 3×（即 `max_attempts=3` 的最坏情况），确保 k8s wait 窗口至少能容纳 seed gen 的全部重试
- 通过环境变量可调

### Fix 0-C: 精确的异常分类

**文件**: `main.py:3447-3462`

```python
# 修改 except Exception 块:
# - k8s_job_timeout → stage_fail_reason = "k8s_job_timeout"（不再笼统叫 stage_dispatch_exception）
# - 即使不重试也要日志精确记录
```

---

## P1：Seed Generation 未为所有 Fuzzer 生成 AI Seeds

### 根因分析

| 根因 | 位置 | 说明 |
|------|------|------|
| Seed gen 串行执行 | `fuzz_unharnessed_repo.py:2342-2348` | `for bin_path in bins` 逐个调用 `_pass_generate_seeds`，第一个 fuzzer 耗时过长则后续 fuzzer 无剩余时间 |
| idle timeout 不受 `seed_generation_timeout_sec` 控制 | `fuzz_unharnessed_repo.py:4071-4074` | 只传了 `timeout`（总执行时间），**没传 `idle_timeout_override`** |
| 默认 idle timeout = 900s | `codex_helper.py:1021` | seed gen 如果 AI 在想，900s 不输出就会被杀，但 `seed_generation_timeout_sec` 无法覆盖它 |
| 异常被静默捕获 | `fuzz_unharnessed_repo.py:2346-2348` | `except HarnessGeneratorError` 只打 warning，不中断不报告 |

### Fix 1-A: 传递 `idle_timeout_override` 给 seed generation

**文件**: `fuzz_unharnessed_repo.py:4071-4074`

```python
# 添加:
seed_idle_timeout = int(os.environ.get("SHERPA_SEED_GEN_IDLE_TIMEOUT_SEC", "300"))
patcher_kwargs["idle_timeout_override"] = seed_idle_timeout
```

- 缩短 idle timeout 到 300s（不需要等 900s）
- 让 `seed_generation_timeout_sec` 控制总体时间，idle timeout 控制单次卡死检测

### Fix 1-B: 给每个 fuzzer 分配时间预算

**文件**: `fuzz_unharnessed_repo.py:2342` 附近

```python
# 将 seed gen 循环改为按预算分配:
total_seed_budget = seed_generation_timeout_sec or 1800
per_fuzzer_budget = max(300, total_seed_budget // len(bins))
for bin_path in bins:
    self.seed_generation_timeout_sec = per_fuzzer_budget
    try:
        self._pass_generate_seeds(bin_path.name)
    except HarnessGeneratorError as e:
        print(f"[!] Seed generation failed ({bin_path.name}): {e}")
```

- 保证每个 fuzzer 有公平的时间窗口
- 第一个 fuzzer 不会吃掉所有时间

### Fix 1-C: 改善失败报告

**文件**: `fuzz_unharnessed_repo.py:2346-2348`

- 记录哪些 fuzzer 的 seed gen 失败了
- 在 `run_summary.json` 里加 `seed_gen_failed_fuzzers` 字段
- 让 coverage-analysis 知道哪些 fuzzer 缺种子，驱动 improve-harness 针对性补种

---

## P2-A：OpenCodeHelper 僵尸进程

### 根因分析

| 根因 | 位置 | 说明 |
|------|------|------|
| `proc.kill()` 后未调用 `proc.wait()` | `codex_helper.py:1489-1493` | `_kill_proc()` 中 `proc.terminate()` + `proc.wait(4)` 正确，但 fallback 的 `proc.kill()` 缺 `proc.wait()` |

### 修复方案

**文件**: `codex_helper.py:1489-1493`

```python
# 现在:
if proc.poll() is None:
    try:
        proc.kill()
    except Exception:
        pass

# 改为:
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

一行修复，影响范围极小。

---

## P2-B：Coverage Plateau 检测后无法有效改善

### 根因分析

| 根因 | 位置 | 说明 |
|------|------|------|
| `improve-harness` 的 in_place 模式只靠 AI 改文件 | `workflow_graph.py:7632-7666` | 让 OpenCode 修改 `fuzz/` 下文件，但没有结构化的种子补种策略 |
| 种子多样性不足 | `fuzz_unharnessed_repo.py:1019-1027` | novelty 计算中 early_new_units 占 50%，但如果 seed 全是同一个 pattern，early units 也低 |
| archive-container 类 fuzzer 只靠 repo examples | - | AI seed gen 不擅长生成二进制 archive 格式种子，只有 4 个 repo examples |
| `_filter_seed_corpus` 的去重过于激进 | `fuzz_unharnessed_repo.py:3766-3781` | strict mode 下同一 shape 只留 1 个 seed，可能过度裁剪 |

### Fix 2-A: 种子补种循环

```
improve-harness (in_place mode) 应增加一个 seed-only 改进路径：
1. 读取 run_summary.json 中的 covered_families / missing_families
2. 仅对 missing_families 触发定向 seed gen（不改 harness 代码）
3. 不走完整的 plan→synthesize→build 循环，直接 build→run
```

### Fix 2-B: archive-container 类型的特殊处理

```
对于 seed_profile == "archive-container" 的 fuzzer：
1. 优先从 repo 中挖掘更多 test fixtures（不只是 corpus/ 下的）
2. 用 radamsa 对 repo examples 做更多变异（当前 radamsa 只做 1 轮）
3. 降低 AI seed gen 的权重，因为 AI 不擅长生成有效的二进制 archive
```

### Fix 2-C: 放宽 strict filter 的 shape 去重

**文件**: `fuzz_unharnessed_repo.py:3766-3781`

```
- strict mode 同 shape 从 1 改为 2
- required family 的 cap 从 3 改为 5
- 保留更多高质量变体
```

### Fix 2-D: plateau 检测参数调优

```
当前: SHERPA_RUN_PLATEAU_IDLE_GROWTH_SEC=180, SHERPA_RUN_PLATEAU_PULSES=3
问题: 对 deep target 来说 180s 太短

建议：
- 覆盖率 > 500 时动态延长 plateau 检测窗口（目标越深越需要耐心）
- 或者加一个 SHERPA_RUN_PLATEAU_DEEP_TARGET_MULTIPLIER
```

---

## 修复优先级与依赖关系

```
P0-A (build timeout 重试)  ← 独立，可直接做
P0-B (run timeout 安全系数) ← 独立，可直接做
P0-C (异常分类精确化)      ← 独立，可直接做
      ↓
P1-A (idle_timeout_override) ← 依赖 P0-B 生效后验证
P1-B (per-fuzzer 时间预算)   ← 独立，可直接做
P1-C (失败报告)             ← 依赖 P1-B
      ↓
P2-A (僵尸进程)             ← 独立，一行修复
P2-B (plateau 改善)         ← 长期，依赖 P1 生效后有更好的种子数据
```

---

## 涉及文件清单

| 文件 | 涉及 Fix |
|------|----------|
| `harness_generator/src/langchain_agent/main.py` | P0-A, P0-B, P0-C |
| `harness_generator/src/fuzz_unharnessed_repo.py` | P1-A, P1-B, P1-C, P2-C |
| `harness_generator/src/codex_helper.py` | P2-A |
| `harness_generator/src/langchain_agent/workflow_graph.py` | P2-B (Fix 2-A) |
