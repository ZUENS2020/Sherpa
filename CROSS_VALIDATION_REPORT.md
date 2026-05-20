# 代码-Wiki 交叉验证报告 (2026-05-20)

## 关键发现

### ❌ 1. 工作流节点数不匹配

| 文档 | 声称节点数 | 实际节点数 | 差异 |
|---|---|---|---|
| CLAUDE.md | **14** | 14 ✓ | 正确 |
| SHERPA_WIKI/Home.md | **15** | 14 ✗ | **错误 -1** |
| SHERPA_WIKI/工作流状态机.md 表 | **17** | 14 ✗ | **错误 -3** |

**实际 14 个节点：**
```
init, analysis, vuln-hunt, plan, synthesize, build, 
per-input-replay, coverage-analysis, improve-harness,
re-build, re-run, crash-analysis, run, crash-triage
```

**Wiki 中列出但代码中不存在的节点：**
- `fix_build` (Wiki: 覆盖率循环列表)
- `fix-crash` (Wiki: Crash 处理列表)
- `fix-harness` (Wiki: 辅助节点列表)

---

### ❌ 2. API 路由数量不完整

| 文档 | 声称 | 实际 | 差异 |
|---|---|---|---|
| CLAUDE.md | **8 个端点** | **13+ 个端点** | **不完整 -5** |

**CLAUDE.md 列出的 8 个：**
- POST /api/task
- GET /api/task/{job_id}
- POST /api/task/{job_id}/resume
- POST /api/task/{job_id}/stop
- GET /api/tasks
- GET /api/system
- GET /api/config
- GET /healthz

**代码中实际的完整 API 列表：**
- GET /api/config
- **GET /api/opencode/providers/{provider}/models** ⬅ 缺漏
- **POST /api/opencode/providers/{provider}/models** ⬅ 缺漏
- PUT /api/config ⬅ 缺漏
- GET /api/system
- **GET /api/metrics** ⬅ 缺漏
- **GET /api/health** ⬅ 缺漏 (不同于 /healthz)
- GET /healthz
- POST /api/task
- GET /api/task/{job_id}
- POST /api/task/{job_id}/resume
- POST /api/task/{job_id}/stop
- GET /api/tasks
- GET /

---

### ❌ 3. FastAPI 端口配置错误

| 文档 | 声称端口 | 实际端口 | 来源 |
|---|---|---|---|
| CLAUDE.md | **8001** | **8000** | `os.environ.get("PORT", "8000")` |

**代码行：** `main.py:5499`

---

### ❌ 4. 代码行号不匹配

| 函数 | CLAUDE.md | Wiki | 实际 | 差异 |
|---|---|---|---|---|
| `build_fuzz_workflow()` | 16271 | 16300 | **16459** | CLAUDE ±188, Wiki ±159 |
| `_recommended_next_step()` | 16194 | 16194 | **16353** | ±159 |

---

### ⚠️ 5. Wiki 工作流状态图问题

#### 5.1 缺失节点路由
Wiki `工作流状态机.md` 的 Mermaid 图中缺少以下实际存在的路由：
- `improve-harness → vuln-hunt` (当 `in_place + vuln` 时)
- `improve-harness → vuln-hunt` (覆盖率循环中)

#### 5.2 状态图节点不一致
Wiki `Home.md` 的 Mermaid 图（22-48 行）使用简化的 14 节点模型，但表格列出 15 节点。

---

### ✓ 6. 正确的项目元素

这些在 CLAUDE.md 和代码中一致：
- ✓ 14 个实际工作流节点
- ✓ 7 个 Docker 镜像
- ✓ ~200 个 State 字段
- ✓ PostgreSQL 16 后端
- ✓ Next.js 14 前端
- ✓ deepseek-v4-pro LLM (via litellm proxy)
- ✓ 环境变量配置机制（虽然列表不完整）

---

## 修正清单

### 📝 CLAUDE.md 需要修正

1. **FastAPI 端口**
   - 当前：`(port 8001)`
   - 应改为：`(port 8000, configurable via PORT env var)`

2. **build_fuzz_workflow 行号**
   - 当前：`workflow_graph.py:16271`
   - 应改为：`workflow_graph.py:16459`

3. **API 路由完整性**
   - 当前：列出 8 个端点
   - 应改为：列出完整 13+ 个或明确说"主要端点"

### 📝 SHERPA_WIKI 需要修正

#### Home.md
1. **LangGraph 节点数**
   - 当前：`15 节点 LangGraph 状态机`
   - 应改为：`14 节点 LangGraph 状态机`

2. **关键数字表**
   - 当前：`LangGraph 节点数 | 15`
   - 应改为：`LangGraph 节点数 | 14`

#### 工作流状态机.md
1. **节点总数**
   - 当前：`15 节点有限状态机`
   - 应改为：`14 节点有限状态机`

2. **节点列表**
   - 移除覆盖率循环节点表中的 `fix_build`
   - 移除 Crash 处理节点表中的 `fix-crash` 和 `fix-harness`
   - 调整三个分类的节点数：主线(6) + 覆盖率(4) + Crash(4) = 14

3. **build_fuzz_workflow 行号**
   - 当前：`workflow_graph.py:16300`
   - 应改为：`workflow_graph.py:16459`

4. **_recommended_next_step 行号**
   - 当前：`workflow_graph.py:16194`
   - 应改为：`workflow_graph.py:16353`

5. **Mermaid 状态图 (工作流状态机.md 第 45-88 行)**
   - 移除以下节点（代码中不存在）：
     - `fix_build`
     - `fix-crash`
     - `fix-harness`
   - 添加缺失的路由：
     - `improve-harness → vuln-hunt` (当 `in_place + vuln`)

---

## 环境变量完整性评估

CLAUDE.md 列出的 env vars 都存在于代码中（✓），但列表不完整：
- 列出：10 个关键 env vars
- 实际：30+ 个 env vars
- **建议**：补充更多常用的环境变量或添加"关键变量"标签

---

## 总结

| 项目 | 严重性 | 数量 |
|---|---|---|
| **行号偏差** | 高 | 2 处 (±159-188 行) |
| **节点数错误** | 高 | 3 处 |
| **API 不完整** | 中 | 1 处 (缺 5 个端点) |
| **端口错误** | 中 | 1 处 |
| **路由缺漏** | 低 | 1 处 |

---

## 验证方法

```bash
# 验证节点数
grep "graph.add_node" workflow_graph.py | wc -l
# 输出: 14 ✓

# 验证端口
grep 'os.environ.get("PORT"' main.py
# 输出: port = int(os.environ.get("PORT", "8000"))

# 验证 API 路由
grep "@app\." main.py | wc -l
# 输出: 13+ ✓

# 验证行号
grep -n "def build_fuzz_workflow" workflow_graph.py
# 输出: 16459
```

