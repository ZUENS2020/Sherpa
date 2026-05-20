# Wiki 修正完成报告 (2026-05-20)

## 已修正项

### ✅ CLAUDE.md (4 处)

1. **FastAPI 端口配置**
   - Line 11: `port 8001` → `port 8000, configurable via PORT env var`
   - Line 100: `## API Routes (port 8001)` → `## API Routes (port 8000)`

2. **build_fuzz_workflow 函数行号**
   - Line 53: `line 16271` → `line 16459`

3. **kubectl 调试命令**
   - Line 175-177: `8001` → `8000` (2 处 curl 命令)

### ✅ SHERPA_WIKI/Home.md (3 处)

1. **节点数：15 → 14**
   - Line 3: 主描述中的"15 节点" → "14 节点"
   - Line 10: 导航表中的"15 节点工作流" → "14 节点工作流"
   - Line 55: 关键数字表中 `LangGraph 节点数 | 15` → `| 14`

### ✅ SHERPA_WIKI/工作流状态机.md (5 处)

1. **节点数：15 → 14**
   - Line 5: `15 节点有限状态机` → `14 节点有限状态机`
   - Line 82: `## 15 节点详解` → `## 14 节点详解`

2. **build_fuzz_workflow 行号**
   - Line 9: `workflow_graph.py:16300` → `workflow_graph.py:16459`

3. **代码示例**
   - Line 12-32: 移除不存在的节点
     - ❌ 删除: `graph.add_node("fix_build", _node_fix_build)`
     - ❌ 删除: `graph.add_node("fix-crash", _node_fix_crash)`
     - ❌ 删除: `graph.add_node("fix-harness", _node_fix_harness)`
   - 调整顺序以匹配实际代码

4. **节点分类表**
   - Line 95: `覆盖率循环节点（5 个）` → `（4 个）`
     - 移除表中的 `fix_build`
   - Line 104: `Crash 处理节点（5 个）` → `（4 个）`
     - 移除表中的 `fix-crash` 和 `fix-harness`

5. **_recommended_next_step 行号**
   - Line 115: `workflow_graph.py:16194` → `workflow_graph.py:16353`

---

## 未修正项（需进一步审视）

### ⚠️ SHERPA_WIKI/工作流状态机.md - Mermaid 状态图问题

**问题位置：** Line 37-80（完整状态图）和 Line 41-54（主线状态）

**问题描述：**
- 图中包含 `fix_build` 节点（Line 49-51）
- 但代码中不存在此节点

**建议：**
选择以下之一：
1. **方案A**：删除图中的 fix_build 节点，只显示 14 个实际节点
   - 风险：可能丢失错误恢复流程的文档化
   - 优点：图形与代码完全对应

2. **方案B**：保留作为概念模型，添加注释说明
   - 注释：`fix_build` 代表构建错误的内部重试逻辑，非独立节点
   - 优点：保留工作流意图文档，避免遗漏错误处理知识

**现状：** 保留原样（待决策）

---

## 验证清单

```bash
# 验证 CLAUDE.md 修正
grep "port 8000" CLAUDE.md           # ✓ 出现 3 次
grep "16459" CLAUDE.md               # ✓ 出现 1 次

# 验证 wiki 节点数修正
grep "14 节点" SHERPA_WIKI/*.md      # ✓ 应出现多次
grep "15 节点" SHERPA_WIKI/*.md      # ✗ 应无结果

# 验证 API 端口修正
grep "port 8001" .                   # ✗ CLAUDE.md 中无结果
```

---

## 差异汇总

| 类别 | 文件 | 修正前 | 修正后 |
|---|---|---|---|
| **端口** | CLAUDE.md | 8001 | 8000 |
| **节点数** | Home.md | 15 | 14 |
| **节点数** | 工作流状态机.md | 15 | 14 |
| **函数行号** | 两个文件 | 16271/16300/16194 | 16459/16353 |
| **代码示例** | 工作流状态机.md | 17 个 add_node | 14 个 add_node |
| **节点分类** | 工作流状态机.md | 6+5+5+1=17 | 6+4+4=14 |

---

## 影响评估

- **严重性：高** → 节点数错误导致用户对系统理解偏差
- **严重性：中** → 行号错误导致用户定位代码困难
- **严重性：中** → 端口错误导致调试命令失败
- **严重性：低** → Mermaid 图形与实现的语义差异（概念 vs 实现）

