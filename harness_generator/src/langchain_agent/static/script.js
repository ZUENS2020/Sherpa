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
    setChecked("cfg_use_docker", true);
    setDisabled("cfg_use_docker", true);
    setValue("cfg_docker_image", cfg.fuzz_docker_image || "auto");

    setValue("oss_fuzz_dir", cfg.oss_fuzz_dir || "");

    setValue("sherpa_git_mirrors", cfg.sherpa_git_mirrors || "");
    setValue("sherpa_docker_http_proxy", cfg.sherpa_docker_http_proxy || "");
    setValue("sherpa_docker_https_proxy", cfg.sherpa_docker_https_proxy || "");
    setValue("sherpa_docker_no_proxy", cfg.sherpa_docker_no_proxy || "");
    setValue("sherpa_docker_proxy_host", cfg.sherpa_docker_proxy_host || "host.docker.internal");

    // Mirror config into the fuzz form defaults.
    setValue("time_budget", String(cfg.fuzz_time_budget ?? 900));
    setChecked("use_docker", true);
    setDisabled("use_docker", true);
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
    setValue("time_budget", String(cfg.fuzz_time_budget));
    setChecked("use_docker", true);
    setDisabled("use_docker", true);
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
    statusEl: document.getElementById("fuzz_status"),
    progressEl: document.getElementById("fuzz_progress"),
    logEl: document.getElementById("fuzz_log"),
  };
}

function ensureTaskUiVisible() {
  const { statusEl, progressEl, logEl } = getTaskUiElements();
  if (statusEl) statusEl.style.display = "block";
  if (progressEl) progressEl.style.display = "block";
  if (logEl) logEl.style.display = "block";
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
  const childSummary = Number(childStatus.total || 0)
    ? `子任务: ${Number(childStatus.running || 0)} running / ${Number(childStatus.success || 0)} success / ${Number(childStatus.error || 0)} error`
    : "子任务: 0";

  if (log !== _activeTaskLastLog || st !== _activeTaskLastStatus) {
    _activeTaskLastLog = log;
    _activeTaskLastStatus = st;
    const summary = summarizeLog(log, st, result, err, logFile);
    logEl.className = "result-box";
    logEl.innerHTML = `<strong>进度摘要（实时）：</strong><pre style="white-space: pre-wrap; margin-top: 8px;">${escapeHtml(summary)}</pre>`;
    logEl.scrollTop = logEl.scrollHeight;
    progressEl.className = "result-box";
    progressEl.innerHTML = renderProgressFromLog(log);
  }

  if (st === "queued" || st === "running") {
    statusEl.className = "result-box loading";
    statusEl.innerHTML =
      `<span class="status-icon">⏳</span> 当前状态：<strong>${escapeHtml(st)}</strong>（任务：${escapeHtml(jobId)}）<br>` +
      `${escapeHtml(childSummary)}`;
    setSessionStatusLine(`已绑定会话 #${jobId.slice(0, 8)}，正在实时监控。`);
    return false;
  }

  if (st === "success") {
    statusEl.className = "result-box success";
    statusEl.innerHTML =
      `<span class="status-icon">✅</span> 完成（任务：<strong>${escapeHtml(jobId)}</strong>）<br>` +
      `${escapeHtml(childSummary)}<br>${escapeHtml(result || "")}`;
    setSessionStatusLine(`会话 #${jobId.slice(0, 8)} 已完成。`);
    return true;
  }

  statusEl.className = "result-box error";
  statusEl.innerHTML =
    `<span class="status-icon">❌</span> 失败（任务：<strong>${escapeHtml(jobId)}</strong>）<br>` +
    `${escapeHtml(childSummary)}<br>${escapeHtml(err || "unknown error")}`;
  setSessionStatusLine(`会话 #${jobId.slice(0, 8)} 已失败。`);
  return true;
}

async function pollActiveTaskOnce() {
  if (!_activeTaskId || _taskPollInFlight) return;
  _taskPollInFlight = true;
  try {
    const r = await fetch(`/api/task/${encodeURIComponent(_activeTaskId)}`);
    if (!r.ok) throw new Error(`轮询失败 HTTP ${r.status}`);
    const taskObj = await r.json();
    if (taskObj.error === "job_not_found") {
      const { statusEl } = getTaskUiElements();
      if (statusEl) {
        statusEl.className = "result-box error";
        statusEl.innerHTML =
          `<span class="status-icon">❌</span> 会话不存在或已清理：<strong>${escapeHtml(_activeTaskId)}</strong>`;
      }
      stopTaskPolling();
      return;
    }
    if (taskObj.error === "job_not_task") {
      const { statusEl } = getTaskUiElements();
      if (statusEl) {
        statusEl.className = "result-box error";
        statusEl.innerHTML =
          `<span class="status-icon">❌</span> 该 ID 不是 task 会话：<strong>${escapeHtml(_activeTaskId)}</strong>`;
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
    const { statusEl } = getTaskUiElements();
    if (statusEl) {
      statusEl.className = "result-box error";
      statusEl.innerHTML = `<span class="status-icon">❌</span> 监控失败：${escapeHtml(err.message)}`;
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

  const { statusEl, progressEl, logEl } = getTaskUiElements();
  if (statusEl) {
    statusEl.className = "result-box loading";
    statusEl.innerHTML =
      `<span class="status-icon">⏳</span> 已绑定会话：<strong>${escapeHtml(targetId)}</strong>，正在拉取状态...`;
  }
  if (progressEl) {
    progressEl.className = "result-box";
    progressEl.innerHTML = "等待任务初始化...";
  }
  if (logEl) {
    logEl.className = "result-box";
    logEl.innerHTML = "";
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
  const timeBudgetRaw = (document.getElementById("time_budget")?.value || "900").trim();
  const useDocker = true;
  const dockerImage = (document.getElementById("docker_image")?.value || "auto").trim();
  const btn = document.getElementById("fuzz_btn");
  const { statusEl, progressEl, logEl } = getTaskUiElements();

  if (!codeUrl) {
    alert("请输入代码仓库地址");
    return;
  }

  const timeBudget = Number.parseInt(timeBudgetRaw || "900", 10);
  if (!Number.isFinite(timeBudget) || timeBudget <= 0) {
    alert("请输入有效的运行时长（秒）");
    return;
  }

  if (!statusEl || !progressEl || !logEl || !btn) return;

  // 禁用按钮，显示加载状态。
  btn.disabled = true;
  ensureTaskUiVisible();
  statusEl.className = "result-box loading";
  statusEl.innerHTML = '<span class="status-icon">⏳</span> 已提交任务，正在排队...';
  progressEl.className = "result-box";
  progressEl.innerHTML = "等待任务初始化...";
  logEl.className = "result-box";
  logEl.innerHTML = "";

  try {
    const res = await fetch("/api/task", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jobs: [
          {
            code_url: codeUrl,
            email: email || null,
            time_budget: timeBudget,
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
    statusEl.className = "result-box error";
    statusEl.innerHTML = `<span class="status-icon">❌</span> <strong>测试失败：</strong> ${escapeHtml(err.message)}`;
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

function getChecked(id) {
  return !!document.getElementById(id)?.checked;
}

function setChecked(id, checked) {
  const el = document.getElementById(id);
  if (el) el.checked = !!checked;
}

function setDisabled(id, disabled) {
  const el = document.getElementById(id);
  if (el) el.disabled = !!disabled;
}

function summarizeLog(log, status, result, err, logFile) {
  const lines = (log || "").split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const important = [];
  const wfRe = /^\[wf\b.*\]/;
  const jobRe = /^\[job\b.*\]/;
  const ocRe = /^\[OpenCodeHelper\]/;
  const buildRe = /(build failed|linking|cmake|compile|error:|fatal|undefined reference)/i;
  const crashRe = /(crash|artifact|asan|ubsan|msan|tsan|segmentation|sanitizer)/i;
  const stepRe = /(->|<-|workflow end|pass [a-e]|ready)/i;
  const writeRe = /(← Write|Wrote file successfully|\bWrite\s+fuzz\/)/i;
  const ocStateRe = /(running…|done flag detected|diff produced|sentinel)/i;

  for (const line of lines) {
    if (wfRe.test(line) && stepRe.test(line)) {
      important.push(line);
      continue;
    }
    if (jobRe.test(line) && (line.includes("start") || line.includes("params"))) {
      important.push(line);
      continue;
    }
    if (ocRe.test(line) && ocStateRe.test(line)) {
      important.push(line);
      continue;
    }
    if (writeRe.test(line)) {
      important.push(line);
      continue;
    }
    if (buildRe.test(line) || crashRe.test(line)) {
      important.push(line);
      continue;
    }
  }

  const tail = important.slice(-10);
  const header = [];
  header.push(`Status: ${status || "unknown"}`);
  if (result) header.push(`Result: ${result}`);
  if (err) header.push(`Error: ${err}`);
  if (logFile) header.push(`Log file: ${logFile}`);

  if (!tail.length) {
    tail.push("暂无可展示的摘要，任务仍在初始化或无关键事件。");
  }

  return header.concat(["", ...tail]).join("\n");
}

function renderProgressFromLog(log) {
  const parsed = parseWorkflowLog(log || "");
  const steps = ["init", "plan", "synthesize", "build", "fix_build", "run", "fix_crash"];
  const labels = {
    init: "初始化",
    plan: "计划",
    synthesize: "生成",
    build: "构建",
    fix_build: "修复构建",
    run: "运行",
    fix_crash: "修复崩溃",
  };

  const statusToColor = {
    pending: "#9ca3af",
    running: "#f59e0b",
    done: "#10b981",
    error: "#ef4444",
  };

  const items = steps
    .map((s) => {
      const st = parsed.steps[s] || "pending";
      const color = statusToColor[st] || "#9ca3af";
      return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <span style="width:8px;height:8px;border-radius:50%;background:${color};display:inline-block;"></span>
      <span>${labels[s]}</span>
      <span style="color:#6d6f73;font-size:12px;">(${st})</span>
    </div>`;
    })
    .join("");

  const repoLine = parsed.repoRoot
    ? `<div style="margin-top:6px;color:#6d6f73;font-size:12px;">输出目录：${escapeHtml(parsed.repoRoot)}</div>`
    : "";
  const lastErr = parsed.lastError
    ? `<div style="margin-top:8px;color:#b91c1c;font-size:12px;">错误：${escapeHtml(parsed.lastError)}</div>`
    : "";
  return `<strong>阶段进度：</strong><div style="margin-top:8px;">${items}</div>${repoLine}${lastErr}`;
}

function parseWorkflowLog(log) {
  const lines = (log || "").split(/\r?\n/);
  const steps = {};
  let repoRoot = "";
  let lastError = "";

  const startRe = /\[wf[^\]]*\]\s*->\s*([a-zA-Z_]+)/;
  const endRe = /\[wf[^\]]*\]\s*<-\s*([a-zA-Z_]+)/;
  const repoRe = /repo_root=([^\s]+)/;

  for (const line of lines) {
    const repoMatch = line.match(repoRe);
    if (repoMatch && !repoRoot) repoRoot = repoMatch[1];

    const startMatch = line.match(startRe);
    if (startMatch) {
      steps[startMatch[1]] = "running";
    }

    const endMatch = line.match(endRe);
    if (endMatch) {
      const step = endMatch[1];
      if (/err=|failed|error/i.test(line)) {
        steps[step] = "error";
        lastError = line.replace(/^\[.*?\]\s*/, "");
      } else {
        steps[step] = "done";
      }
    }

    // Avoid treating generic OpenCode progress/env lines as errors.
    const opencodeErrorLike =
      /\b(OpenCodeHelper|opencode)\b/i.test(line) &&
      /(error|failed|exception|timeout)/i.test(line);
    if (/Missing fuzz\/build\.py/i.test(line) || opencodeErrorLike) {
      lastError = line.replace(/^\[.*?\]\s*/, "");
    }
  }

  return { steps, repoRoot, lastError };
}
