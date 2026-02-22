// ===== 持久化配置（加载/保存） =====
async function loadConfigIntoForm() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`HTTP 错误 ${res.status}`);
    const cfg = await res.json();

    setValue("deepseek_api_key", cfg.openai_api_key || "");
    setValue("deepseek_base_url", cfg.openai_base_url || "https://api.deepseek.com/v1");
    setValue("deepseek_model", cfg.openai_model || cfg.opencode_model || "deepseek-reasoner");

    setValue("cfg_time_budget", String(cfg.fuzz_time_budget ?? 900));
    setValue("cfg_docker_image", cfg.fuzz_docker_image || "auto");

    setValue("oss_fuzz_dir", cfg.oss_fuzz_dir || "");

    setValue("sherpa_git_mirrors", cfg.sherpa_git_mirrors || "");
    setValue("sherpa_docker_http_proxy", cfg.sherpa_docker_http_proxy || "");
    setValue("sherpa_docker_https_proxy", cfg.sherpa_docker_https_proxy || "");
    setValue("sherpa_docker_no_proxy", cfg.sherpa_docker_no_proxy || "");
    setValue("sherpa_docker_proxy_host", cfg.sherpa_docker_proxy_host || "host.docker.internal");

    // Mirror config into the fuzz form defaults.
    setValue("total_time_budget", String(cfg.fuzz_time_budget ?? 900));
    setValue("run_time_budget", String(cfg.fuzz_time_budget ?? 900));
    setValue("docker_image", cfg.fuzz_docker_image || "auto");
  } catch (e) {
    // Silent: page still usable.
    console.warn("load config failed", e);
  }
}

function gatherConfigFromForm() {
  const timeBudgetRaw = (getValue("cfg_time_budget") || "900").trim();
  const fuzzTimeBudget = Number.parseInt(timeBudgetRaw || "900", 10);

  if (!Number.isFinite(fuzzTimeBudget) || fuzzTimeBudget <= 0) {
    throw new Error("请输入有效的 Fuzz 默认运行时长（秒）");
  }

  const deepseekBase = (getValue("deepseek_base_url") || "").trim() || "https://api.deepseek.com/v1";
  const deepseekModel = (getValue("deepseek_model") || "").trim() || "deepseek-reasoner";
  const normalizedModel = deepseekModel.replace(/^deepseek\//, "");
  const opencodeModel = deepseekModel.includes("/") ? deepseekModel : `deepseek/${normalizedModel}`;

  return {
    openai_api_key: (getValue("deepseek_api_key") || "").trim() || null,
    openai_base_url: deepseekBase,
    openai_model: normalizedModel,
    opencode_model: opencodeModel,
    openrouter_api_key: null,
    openrouter_base_url: "",
    openrouter_model: "",

    fuzz_time_budget: fuzzTimeBudget,
    fuzz_use_docker: true,
    fuzz_docker_image: (getValue("cfg_docker_image") || "").trim() || "auto",

    oss_fuzz_dir: (getValue("oss_fuzz_dir") || "").trim() || "",

    sherpa_git_mirrors: (getValue("sherpa_git_mirrors") || "").trim() || "",
    sherpa_docker_http_proxy: (getValue("sherpa_docker_http_proxy") || "").trim() || "",
    sherpa_docker_https_proxy: (getValue("sherpa_docker_https_proxy") || "").trim() || "",
    sherpa_docker_no_proxy: (getValue("sherpa_docker_no_proxy") || "").trim() || "",
    sherpa_docker_proxy_host: (getValue("sherpa_docker_proxy_host") || "").trim() || "host.docker.internal",

    version: 1,
  };
}

// ===== 全局状态 =====
const ACTIVE_TASK_STORAGE_KEY = "sherpa_active_task_id";
let _systemPollTimer = null;
let _sessionListPollTimer = null;
let _taskPollTimer = null;
let _taskPollInFlight = false;

let _activeTaskId = "";
let _activeTaskLastLog = "";
let _activeTaskLastStatus = "";
let _sessionItems = [];

// ===== 系统状态监控 =====
function formatDuration(sec) {
  if (!Number.isFinite(sec) || sec < 0) return "--";
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${m}m ${r}s`;
  if (m > 0) return `${m}m ${r}s`;
  return `${r}s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "--";
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  if (mb < 1024) return `${mb.toFixed(1)} MB`;
  const gb = mb / 1024;
  return `${gb.toFixed(1)} GB`;
}

function formatIsoTime(iso) {
  if (!iso) return "--";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleString("zh-CN", { hour12: false });
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setStatus(ok, message) {
  const dot = document.getElementById("system_status_dot");
  const text = document.getElementById("system_status_text");
  if (dot) {
    dot.className = "dot" + (ok ? " ok" : " warn");
  }
  if (text) text.textContent = message;
}

function setSessionStatusLine(message) {
  setText("task_session_status", message);
}

async function pollSystemStatus() {
  try {
    const res = await fetch("/api/system");
    if (!res.ok) throw new Error(`HTTP 错误 ${res.status}`);
    const data = await res.json();

    setStatus(true, "联机");
    setText("system_time", data.server_time_iso || "--");
    setText("system_uptime", formatDuration(data.uptime_sec));

    const jobs = data.jobs || {};
    setText("system_jobs_total", String(jobs.total ?? "--"));
    setText("system_jobs_queued", String(jobs.queued ?? "--"));
    setText("system_jobs_running", String(jobs.running ?? "--"));
    setText("system_jobs_success", String(jobs.success ?? "--"));
    setText("system_jobs_error", String(jobs.error ?? "--"));

    const logs = data.logs || {};
    const logInfo = logs.exists ? `${logs.dir} (${formatBytes(logs.size_bytes)})` : "未创建";
    setText("system_logs", logInfo);

    const cfg = data.config || {};
    setText("system_ossfuzz_dir", cfg.oss_fuzz_dir || "--");

    const listEl = document.getElementById("system_jobs_list");
    if (listEl) {
      const items = data.active_jobs || [];
      if (!items.length) {
        listEl.textContent = "暂无运行中的任务";
      } else {
        const lines = items.map((j) => {
          const id = (j.job_id || "").slice(0, 8);
          const st = j.status || "unknown";
          const repo = j.repo || "";
          return `#${id} ${st} ${repo}`;
        });
        listEl.textContent = lines.join("\n");
      }
    }
  } catch (err) {
    setStatus(false, "离线");
    setText("system_time", "--");
  }
}

function startSystemPolling() {
  if (_systemPollTimer) {
    clearInterval(_systemPollTimer);
  }
  pollSystemStatus();
  _systemPollTimer = setInterval(pollSystemStatus, 2000);
}

function startSessionListPolling() {
  if (_sessionListPollTimer) {
    clearInterval(_sessionListPollTimer);
  }
  _sessionListPollTimer = setInterval(() => {
    const preferred = _activeTaskId || (getValue("task_session_id") || "").trim();
    loadTaskSessions(preferred).catch((err) => {
      console.warn("refresh sessions failed", err);
    });
  }, 5000);
}

async function saveConfigFromForm() {
  const statusEl = document.getElementById("cfg_status");
  statusEl.style.display = "block";
  statusEl.className = "result-box loading";
  statusEl.innerHTML = '<span class="status-icon">⏳</span> 正在保存配置...';

  try {
    const cfg = gatherConfigFromForm();
    const res = await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    if (!res.ok) {
      let detail = `HTTP 错误 ${res.status}`;
      try {
        const errObj = await res.json();
        detail = errObj?.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const data = await res.json();
    if (!data.ok) throw new Error("保存失败");

    // Update fuzz form defaults immediately.
    setValue("total_time_budget", String(cfg.fuzz_time_budget));
    setValue("run_time_budget", String(cfg.fuzz_time_budget));
    setValue("docker_image", cfg.fuzz_docker_image);

    statusEl.className = "result-box success";
    statusEl.innerHTML = '<span class="status-icon">✅</span> 配置已保存并立即生效（已持久化）。';
  } catch (err) {
    statusEl.className = "result-box error";
    statusEl.innerHTML = `<span class="status-icon">❌</span> <strong>保存失败：</strong> ${escapeHtml(err.message)}`;
  }
}

// ===== 会话列表与监控 =====
function formatSessionLabel(item) {
  const shortId = (item.job_id || "").slice(0, 8);
  const st = item.status || "unknown";
  const child = item.children_status || {};
  const childDone = Number(child.success || 0) + Number(child.error || 0);
  const childTotal = Number(item.child_count || child.total || 0);
  const repo = item.repo && item.repo !== "batch" ? item.repo : "batch";
  const updated = formatIsoTime(item.updated_at_iso);
  return `#${shortId} | ${st} | 子任务 ${childDone}/${childTotal} | ${repo} | ${updated}`;
}

function updateSessionHint() {
  const hintEl = document.getElementById("task_session_hint");
  const selectEl = document.getElementById("task_session_select");
  if (!hintEl || !selectEl) return;
  const selectedId = (selectEl.value || "").trim();
  const item = _sessionItems.find((x) => x.job_id === selectedId);
  if (!item) {
    hintEl.textContent = _sessionItems.length ? "请选择会话并点击“绑定监控”。" : "暂无历史会话。";
    return;
  }
  const child = item.children_status || {};
  hintEl.textContent =
    `状态: ${item.status || "unknown"} | 子任务: ${Number(child.total || item.child_count || 0)} ` +
    `(running=${Number(child.running || 0)}, success=${Number(child.success || 0)}, error=${Number(child.error || 0)})`;
}

function rememberActiveTask(jobId) {
  try {
    localStorage.setItem(ACTIVE_TASK_STORAGE_KEY, jobId);
  } catch (_) {}
}

function recallActiveTask() {
  try {
    return localStorage.getItem(ACTIVE_TASK_STORAGE_KEY) || "";
  } catch (_) {
    return "";
  }
}

function setTaskSelection(jobId) {
  const normalized = (jobId || "").trim();
  setValue("task_session_id", normalized);
  const selectEl = document.getElementById("task_session_select");
  if (!selectEl) return;
  if (normalized && Array.from(selectEl.options).some((opt) => opt.value === normalized)) {
    selectEl.value = normalized;
  }
  updateSessionHint();
}

async function loadTaskSessions(preferredId = "") {
  const selectEl = document.getElementById("task_session_select");
  const hintEl = document.getElementById("task_session_hint");
  if (!selectEl) return [];

  try {
    let source = "tasks";
    let items = [];
    const res = await fetch("/api/tasks?limit=80");
    if (res.ok) {
      const payload = await res.json();
      items = Array.isArray(payload.items) ? payload.items : [];
    } else {
      source = "system";
      const sysRes = await fetch("/api/system");
      if (!sysRes.ok) throw new Error(`HTTP 错误 ${res.status}/${sysRes.status}`);
      const sys = await sysRes.json();
      const active = Array.isArray(sys.active_jobs) ? sys.active_jobs : [];
      items = active
        .filter((j) => (j.kind || "") === "task")
        .map((j) => ({
          job_id: j.job_id,
          status: j.status || "running",
          repo: j.repo || "batch",
          updated_at_iso: j.updated_at ? new Date(Number(j.updated_at) * 1000).toISOString() : "",
          children_status: { total: 0, queued: 0, running: 0, success: 0, error: 0 },
          child_count: 0,
        }));
    }
    _sessionItems = items;

    selectEl.innerHTML = "";
    if (!items.length) {
      const emptyOpt = document.createElement("option");
      emptyOpt.value = "";
      emptyOpt.textContent = "暂无会话";
      selectEl.appendChild(emptyOpt);
      selectEl.disabled = true;
      if (hintEl) hintEl.textContent = "暂无历史会话。";
      return items;
    }

    selectEl.disabled = false;
    for (const item of items) {
      const opt = document.createElement("option");
      opt.value = item.job_id || "";
      opt.textContent = formatSessionLabel(item);
      selectEl.appendChild(opt);
    }

    const manualId = (getValue("task_session_id") || "").trim();
    const remembered = recallActiveTask();
    const running = items.find((x) => x.status === "running");
    const fallback = (running && running.job_id) || (items[0] && items[0].job_id) || "";
    const desired = (preferredId || manualId || remembered || fallback || "").trim();
    const exists = desired && items.some((x) => x.job_id === desired);
    const selected = exists ? desired : fallback;
    selectEl.value = selected;
    setValue("task_session_id", selected);
    updateSessionHint();
    if (source === "system" && hintEl) {
      const base = hintEl.textContent || "";
      hintEl.textContent = `${base}（后端未重载新版接口，仅显示运行中会话）`;
    }
    return items;
  } catch (err) {
    selectEl.innerHTML = "";
    const failOpt = document.createElement("option");
    failOpt.value = "";
    failOpt.textContent = "加载会话失败";
    selectEl.appendChild(failOpt);
    selectEl.disabled = true;
    if (hintEl) hintEl.textContent = `会话列表加载失败: ${err.message}`;
    return [];
  }
}

function getTaskUiElements() {
  return {
    panelEl: document.getElementById("task_panels"),
    statusEl: document.getElementById("fuzz_status"),
    progressEl: document.getElementById("fuzz_progress"),
    errorPanelEl: document.getElementById("fuzz_error_panel"),
    errorEl: document.getElementById("fuzz_error"),
    logEl: document.getElementById("fuzz_log"),
  };
}

function ensureTaskUiVisible() {
  const { panelEl } = getTaskUiElements();
  if (panelEl) panelEl.style.display = "grid";
}

function extractPrimaryChild(taskObj) {
  const children = Array.isArray(taskObj.children) ? taskObj.children : [];
  if (!children.length) return null;
  return (
    children.find((c) => c.status === "running") ||
    children.find((c) => c.status === "queued") ||
    children.find((c) => c.status === "error") ||
    children[0]
  );
}

function stopTaskPolling() {
  if (_taskPollTimer) {
    clearInterval(_taskPollTimer);
    _taskPollTimer = null;
  }
}

function mapStatusLabel(status) {
  const st = String(status || "unknown");
  if (st === "queued") return "排队中";
  if (st === "running") return "运行中";
  if (st === "success") return "已完成";
  if (st === "error") return "失败";
  return "未知";
}

function mapStatusClass(status) {
  const st = String(status || "unknown");
  if (st === "queued" || st === "running" || st === "success" || st === "error") return st;
  return "unknown";
}

function mapStepLabel(step) {
  const labels = {
    init: "初始化",
    plan: "计划",
    synthesize: "生成",
    build: "构建",
    fix_build: "修复构建",
    run: "运行",
    fix_crash: "修复崩溃",
    report: "报告",
    summarize: "总结",
  };
  if (labels[step]) return labels[step];
  return String(step || "")
    .split("_")
    .filter(Boolean)
    .join(" ");
}

function mapStepStatusLabel(stepStatus) {
  if (stepStatus === "running") return "running";
  if (stepStatus === "done") return "done";
  if (stepStatus === "error") return "error";
  return "pending";
}

function mapEventTag(kind, level) {
  if (kind === "workflow") return "WF";
  if (kind === "job") return "JOB";
  if (kind === "helper") return "HELPER";
  if (kind === "command") return "CMD";
  if (kind === "warning") return "WARN";
  if (kind === "failure") return "ERROR";
  if (level === "warn") return "WARN";
  if (level === "error") return "ERROR";
  return "INFO";
}

function isWorkflowStepFailureLine(line) {
  const s = String(line || "");
  return /\[wf[^\]]*\]\s*<-\s*[a-zA-Z_]+\b.*\berr=/i.test(s) || /\[wf[^\]]*\]\s*<-\s*[a-zA-Z_]+\b.*\bfailed\b/i.test(s);
}

function isExplicitFailureLine(line) {
  const s = String(line || "").trim();
  if (!s) return false;
  if (isWorkflowStepFailureLine(s)) return true;
  if (/^\s*ERROR:\s+/i.test(s)) return true;
  if (/^\s*Error:\s+/i.test(s) && /(failed|exception|traceback|rc=\d+|timeout|dial tcp|no such host|cannot|unable)/i.test(s)) return true;
  if (/fuzz_unharnessed_repo\.[A-Za-z_]+Error:/.test(s)) return true;
  if (/^\s*Traceback \(most recent call last\):/.test(s)) return true;
  if (/^\s*raise\s+[A-Za-z_]*Error\b/.test(s)) return true;
  if (/Docker build failed \(rc=\d+\)/i.test(s)) return true;
  if (/error during connect:/i.test(s)) return true;
  if (/dial tcp: .* no such host/i.test(s)) return true;
  return false;
}

function isWarningLine(line) {
  const s = String(line || "").trim();
  if (!s) return false;
  if (/^\s*WARN(?:ING)?[:\s]/i.test(s)) return true;
  if (/\bDEPRECATED:/i.test(s)) return true;
  if (/\bwarning:\b/i.test(s)) return true;
  return false;
}

function classifyLogEvent(line) {
  const s = String(line || "").trim();
  if (!s) return null;
  if (isExplicitFailureLine(s)) return { level: "error", kind: "failure", text: s };
  if (isWarningLine(s)) return { level: "warn", kind: "warning", text: s };
  if (/^\[wf\b/i.test(s)) return { level: "info", kind: "workflow", text: s };
  if (/^\[job\b/i.test(s)) return { level: "info", kind: "job", text: s };
  if (/^\[OpenCodeHelper\]/i.test(s)) return { level: "info", kind: "helper", text: s };
  if (/^\[\*\]\s*➜/.test(s)) return { level: "info", kind: "command", text: s };
  if (/^\s*错误[:：]/.test(s)) return { level: "warn", kind: "warning", text: s };
  if (/Wrote file successfully|done flag detected|artifact_prefix=|workflow end/i.test(s)) {
    return { level: "info", kind: "info", text: s };
  }
  return null;
}

function renderTaskStatusPanel(jobId, status, childStatus, result, logFile) {
  const total = Number(childStatus.total || 0);
  const running = Number(childStatus.running || 0);
  const success = Number(childStatus.success || 0);
  const error = Number(childStatus.error || 0);
  const queued = Number(childStatus.queued || 0);
  const done = success + error;

  const summaryLabel = total
    ? `${done}/${total}`
    : "0/0";

  const resultText = result ? `结果：${result}` : "结果将在任务完成后显示。";
  return `
    <div class="status-head">
      <span class="status-pill ${mapStatusClass(status)}">${escapeHtml(mapStatusLabel(status))}</span>
      <span class="status-job-id">#${escapeHtml(String(jobId || "").slice(0, 12))}</span>
    </div>
    <div class="status-meta-grid">
      <div class="status-meta-item"><div class="key">子任务</div><div class="val">${escapeHtml(summaryLabel)}</div></div>
      <div class="status-meta-item"><div class="key">running / queued</div><div class="val">${running} / ${queued}</div></div>
      <div class="status-meta-item"><div class="key">success / error</div><div class="val">${success} / ${error}</div></div>
      <div class="status-meta-item"><div class="key">日志文件</div><div class="val">${escapeHtml(logFile || "--")}</div></div>
    </div>
    <div class="status-note">${escapeHtml(resultText)}</div>
  `;
}

function renderProgressPanel(parsed) {
  const statusByStep = parsed.steps || {};
  const ordered = ["init", "plan", "synthesize", "build", "fix_build", "run", "fix_crash", "report"];
  for (const step of parsed.stepOrder || []) {
    if (!ordered.includes(step)) ordered.push(step);
  }
  if (!ordered.length) {
    return '<div class="empty-note">尚未捕获到 workflow 阶段日志。</div>';
  }

  const items = ordered
    .map((step) => {
      const st = statusByStep[step] || "pending";
      return `
        <div class="progress-item">
          <span class="progress-dot ${st}"></span>
          <span>${escapeHtml(mapStepLabel(step))}</span>
          <span class="progress-status">${escapeHtml(mapStepStatusLabel(st))}</span>
        </div>
      `;
    })
    .join("");

  const repoRoot = parsed.repoRoot
    ? `<div class="progress-footnote">repo_root: ${escapeHtml(parsed.repoRoot)}</div>`
    : "";
  return `<div class="progress-list">${items}</div>${repoRoot}`;
}

function renderErrorPanel(status, backendError, parsed) {
  const { errorPanelEl, errorEl } = getTaskUiElements();
  if (!errorPanelEl || !errorEl) return;
  const hasBackendError = !!(backendError && backendError !== "unknown error");
  const hasParsedError = Array.isArray(parsed.failureLines) && parsed.failureLines.length > 0;
  const shouldShow = status === "error" || (status !== "success" && hasBackendError);
  if (!shouldShow) {
    errorPanelEl.style.display = "none";
    errorEl.innerHTML = "";
    return;
  }

  const primary = hasBackendError
    ? backendError
    : (hasParsedError ? parsed.failureLines[parsed.failureLines.length - 1] : "任务失败，但未返回明确错误。");
  const lines = hasParsedError ? parsed.failureLines.slice(-3) : [];
  const details = lines.length
    ? `<div style="margin-top:8px;"><strong>最近失败事件</strong><br>${lines.map((x) => escapeHtml(x)).join("<br>")}</div>`
    : "";

  errorPanelEl.style.display = "block";
  errorEl.innerHTML = `
    <div class="error-box">
      <div class="error-title">Task Failure</div>
      <div>${escapeHtml(primary)}</div>
      ${details}
    </div>
  `;
}

function renderLogPanel(parsed) {
  const events = Array.isArray(parsed.events) ? parsed.events : [];
  const shown = events.slice(-120);
  const stats = { info: 0, warn: 0, error: 0 };
  for (const event of shown) {
    if (event.level === "error") stats.error += 1;
    else if (event.level === "warn") stats.warn += 1;
    else stats.info += 1;
  }

  const header = `
    <div class="log-header">
      <div class="log-meta">显示 ${shown.length} 条关键事件（总日志 ${parsed.totalLines} 行）</div>
      <div class="log-counter">
        <span class="level-info">info ${stats.info}</span>
        <span class="level-warn">warn ${stats.warn}</span>
        <span class="level-error">error ${stats.error}</span>
      </div>
    </div>
  `;

  if (!shown.length) {
    return `${header}<div class="empty-note">暂无关键日志，任务可能仍在初始化。</div>`;
  }

  const lines = shown
    .map((event) => {
      return `
        <div class="event-line level-${event.level}">
          <span class="event-tag">${escapeHtml(mapEventTag(event.kind, event.level))}</span>
          <code class="event-text">${escapeHtml(event.text)}</code>
        </div>
      `;
    })
    .join("");

  return `${header}<div class="event-log-list">${lines}</div>`;
}

function renderTaskState(jobId, taskObj) {
  const { statusEl, progressEl, logEl } = getTaskUiElements();
  if (!statusEl || !progressEl || !logEl) return false;

  const st = taskObj.status || "unknown";
  const child = extractPrimaryChild(taskObj);
  const err = (child && child.error) || taskObj.error;
  const result = (child && child.result) || taskObj.result;
  const log = String(((child && child.log) || taskObj.log || "") ?? "");
  const logFile = (child && child.log_file) || taskObj.log_file;
  const childStatus = taskObj.children_status || {};
  const parsed = parseWorkflowLog(log);
  const renderKey = `${st}|${result || ""}|${err || ""}|${log}`;

  statusEl.innerHTML = renderTaskStatusPanel(jobId, st, childStatus, result, logFile);
  renderErrorPanel(st, err, parsed);

  if (renderKey !== _activeTaskLastLog || st !== _activeTaskLastStatus) {
    _activeTaskLastLog = renderKey;
    _activeTaskLastStatus = st;
    progressEl.innerHTML = renderProgressPanel(parsed);
    logEl.innerHTML = renderLogPanel(parsed);
    logEl.scrollTop = logEl.scrollHeight;
  }

  if (st === "queued" || st === "running") {
    setSessionStatusLine(`已绑定会话 #${jobId.slice(0, 8)}，正在实时监控。`);
    return false;
  }
  if (st === "success") {
    setSessionStatusLine(`会话 #${jobId.slice(0, 8)} 已完成。`);
    return true;
  }
  if (st === "error") {
    setSessionStatusLine(`会话 #${jobId.slice(0, 8)} 已失败。`);
    return true;
  }
  setSessionStatusLine(`会话 #${jobId.slice(0, 8)} 状态: ${st}`);
  return false;
}

async function pollActiveTaskOnce() {
  if (!_activeTaskId || _taskPollInFlight) return;
  _taskPollInFlight = true;
  try {
    const r = await fetch(`/api/task/${encodeURIComponent(_activeTaskId)}`);
    if (!r.ok) throw new Error(`轮询失败 HTTP ${r.status}`);
    const taskObj = await r.json();
    if (taskObj.error === "job_not_found") {
      const { statusEl, progressEl, logEl, errorPanelEl, errorEl } = getTaskUiElements();
      if (statusEl) {
        statusEl.innerHTML = `<div class="empty-note">会话不存在或已清理：${escapeHtml(_activeTaskId)}</div>`;
      }
      if (progressEl) progressEl.innerHTML = '<div class="empty-note">无法继续获取阶段进度。</div>';
      if (logEl) logEl.innerHTML = '<div class="empty-note">无法继续获取日志。</div>';
      if (errorPanelEl && errorEl) {
        errorPanelEl.style.display = "block";
        errorEl.innerHTML = `<div class="error-box"><div class="error-title">Task Missing</div><div>会话不存在或已清理。</div></div>`;
      }
      stopTaskPolling();
      return;
    }
    if (taskObj.error === "job_not_task") {
      const { statusEl, progressEl, logEl, errorPanelEl, errorEl } = getTaskUiElements();
      if (statusEl) {
        statusEl.innerHTML = `<div class="empty-note">该 ID 不是 task 会话：${escapeHtml(_activeTaskId)}</div>`;
      }
      if (progressEl) progressEl.innerHTML = '<div class="empty-note">无法继续获取阶段进度。</div>';
      if (logEl) logEl.innerHTML = '<div class="empty-note">无法继续获取日志。</div>';
      if (errorPanelEl && errorEl) {
        errorPanelEl.style.display = "block";
        errorEl.innerHTML = `<div class="error-box"><div class="error-title">Invalid Session</div><div>该 ID 对应的作业类型不是 task。</div></div>`;
      }
      stopTaskPolling();
      return;
    }

    const done = renderTaskState(_activeTaskId, taskObj);
    if (done) {
      stopTaskPolling();
      await loadTaskSessions(_activeTaskId);
    }
  } catch (err) {
    const { statusEl, errorPanelEl, errorEl } = getTaskUiElements();
    if (statusEl) statusEl.innerHTML = `<div class="empty-note">监控请求失败，请检查网络或后端状态。</div>`;
    if (errorPanelEl && errorEl) {
      errorPanelEl.style.display = "block";
      errorEl.innerHTML = `<div class="error-box"><div class="error-title">Polling Error</div><div>${escapeHtml(err.message)}</div></div>`;
    }
  } finally {
    _taskPollInFlight = false;
  }
}

function startTaskPolling(jobId) {
  const targetId = (jobId || "").trim();
  if (!targetId) return;

  _activeTaskId = targetId;
  _activeTaskLastLog = "";
  _activeTaskLastStatus = "";
  rememberActiveTask(targetId);
  setTaskSelection(targetId);
  ensureTaskUiVisible();

  const { statusEl, progressEl, logEl, errorPanelEl, errorEl } = getTaskUiElements();
  if (statusEl) {
    statusEl.innerHTML = renderTaskStatusPanel(targetId, "running", {}, "", "");
  }
  if (progressEl) {
    progressEl.innerHTML = '<div class="empty-note">等待任务初始化...</div>';
  }
  if (logEl) {
    logEl.innerHTML = '<div class="empty-note">等待任务日志输出...</div>';
  }
  if (errorPanelEl && errorEl) {
    errorPanelEl.style.display = "none";
    errorEl.innerHTML = "";
  }

  stopTaskPolling();
  pollActiveTaskOnce();
  _taskPollTimer = setInterval(() => {
    pollActiveTaskOnce();
  }, 2000);
}

function bindSelectedSession() {
  const manual = (getValue("task_session_id") || "").trim();
  const selectVal = (getValue("task_session_select") || "").trim();
  const targetId = manual || selectVal;
  if (!targetId) {
    alert("请选择会话或输入任务 ID");
    return;
  }
  startTaskPolling(targetId);
}

// ===== 模糊测试功能 =====
document.getElementById("fuzz_btn")?.addEventListener("click", async () => {
  const codeUrl = (document.getElementById("code_url")?.value || "").trim();
  const email = (document.getElementById("email")?.value || "").trim();
  const totalTimeBudgetRaw = (document.getElementById("total_time_budget")?.value || "900").trim();
  const runTimeBudgetRaw = (document.getElementById("run_time_budget")?.value || "900").trim();
  const useDocker = true;
  const dockerImage = (document.getElementById("docker_image")?.value || "auto").trim();
  const btn = document.getElementById("fuzz_btn");
  const { statusEl, progressEl, logEl, errorPanelEl, errorEl } = getTaskUiElements();

  if (!codeUrl) {
    alert("请输入代码仓库地址");
    return;
  }

  const totalTimeBudget = Number.parseInt(totalTimeBudgetRaw || "900", 10);
  if (!Number.isFinite(totalTimeBudget) || totalTimeBudget <= 0) {
    alert("请输入有效的总时长（秒）");
    return;
  }

  const runTimeBudget = Number.parseInt(runTimeBudgetRaw || "900", 10);
  if (!Number.isFinite(runTimeBudget) || runTimeBudget <= 0) {
    alert("请输入有效的单次运行时长（秒）");
    return;
  }

  if (!statusEl || !progressEl || !logEl || !btn) return;

  // 禁用按钮，显示加载状态。
  btn.disabled = true;
  ensureTaskUiVisible();
  statusEl.innerHTML = renderTaskStatusPanel("pending", "queued", {}, "", "");
  progressEl.innerHTML = '<div class="empty-note">等待任务初始化...</div>';
  logEl.innerHTML = '<div class="empty-note">等待任务日志输出...</div>';
  if (errorPanelEl && errorEl) {
    errorPanelEl.style.display = "none";
    errorEl.innerHTML = "";
  }

  try {
    const res = await fetch("/api/task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jobs: [
          {
            code_url: codeUrl,
            email: email || null,
            time_budget: totalTimeBudget,
            total_time_budget: totalTimeBudget,
            run_time_budget: runTimeBudget,
            docker: useDocker,
            docker_image: dockerImage,
          },
        ],
      }),
    });

    if (!res.ok) {
      let detail = `HTTP 错误 ${res.status}`;
      try {
        const errObj = await res.json();
        detail = errObj?.detail || detail;
      } catch (_) {}
      throw new Error(detail);
    }

    const data = await res.json();
    const jobId = (data && data.job_id) || "";
    if (!jobId) throw new Error("服务端未返回 job_id");

    setSessionStatusLine(`已创建会话 #${jobId.slice(0, 8)}，准备监控。`);
    await loadTaskSessions(jobId);
    startTaskPolling(jobId);
  } catch (err) {
    statusEl.innerHTML = '<div class="empty-note">任务提交失败。</div>';
    if (errorPanelEl && errorEl) {
      errorPanelEl.style.display = "block";
      errorEl.innerHTML = `<div class="error-box"><div class="error-title">Submit Error</div><div>${escapeHtml(err.message)}</div></div>`;
    }
  } finally {
    btn.disabled = false;
  }
});

// ===== 页面事件 =====
document.getElementById("cfg_save_btn")?.addEventListener("click", async () => {
  await saveConfigFromForm();
});

document.getElementById("refresh_system_btn")?.addEventListener("click", async () => {
  await pollSystemStatus();
});

document.getElementById("refresh_sessions_btn")?.addEventListener("click", async () => {
  const preferred = _activeTaskId || (getValue("task_session_id") || "").trim();
  await loadTaskSessions(preferred);
});

document.getElementById("bind_session_btn")?.addEventListener("click", () => {
  bindSelectedSession();
});

document.getElementById("task_session_select")?.addEventListener("change", () => {
  const val = (getValue("task_session_select") || "").trim();
  setValue("task_session_id", val);
  updateSessionHint();
});

document.addEventListener("DOMContentLoaded", async () => {
  await loadConfigIntoForm();
  startSystemPolling();
  await loadTaskSessions();
  startSessionListPolling();

  const remembered = recallActiveTask();
  if (remembered) {
    startTaskPolling(remembered);
    return;
  }
  const running = _sessionItems.find((x) => x.status === "running");
  if (running && running.job_id) {
    startTaskPolling(running.job_id);
  }
});

// ===== 工具函数 =====
function escapeHtml(text) {
  const map = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return String(text ?? "").replace(/[&<>"']/g, (m) => map[m]);
}

function getValue(id) {
  return document.getElementById(id)?.value;
}

function setValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.value = value;
}

function parseWorkflowLog(log) {
  const lines = (log || "").split(/\r?\n/);
  const steps = {};
  const stepOrder = [];
  let repoRoot = "";
  const failureLines = [];
  const events = [];

  const startRe = /\[wf[^\]]*\]\s*->\s*([a-zA-Z_]+)/;
  const endRe = /\[wf[^\]]*\]\s*<-\s*([a-zA-Z_]+)/;
  const repoRe = /repo_root=([^\s]+)/;

  for (const line of lines) {
    const clean = String(line || "").trim();
    if (!clean) continue;

    const event = classifyLogEvent(clean);
    if (event) events.push(event);
    if (isExplicitFailureLine(clean)) {
      failureLines.push(clean.replace(/^\[.*?\]\s*/, ""));
    }

    const repoMatch = line.match(repoRe);
    if (repoMatch && !repoRoot) repoRoot = repoMatch[1];

    const startMatch = line.match(startRe);
    if (startMatch) {
      const step = startMatch[1];
      if (!stepOrder.includes(step)) stepOrder.push(step);
      if (steps[step] !== "done" && steps[step] !== "error") {
        steps[step] = "running";
      }
    }

    const endMatch = line.match(endRe);
    if (endMatch) {
      const step = endMatch[1];
      if (!stepOrder.includes(step)) stepOrder.push(step);
      if (isWorkflowStepFailureLine(line)) {
        steps[step] = "error";
      } else {
        steps[step] = "done";
      }
    }
  }

  if (!events.length) {
    const tail = lines
      .map((x) => String(x || "").trim())
      .filter(Boolean)
      .slice(-20)
      .map((x) => ({ level: "info", kind: "info", text: x }));
    events.push(...tail);
  }

  return {
    steps,
    stepOrder,
    repoRoot,
    failureLines,
    events,
    totalLines: lines.filter((x) => String(x || "").trim()).length,
  };
}
