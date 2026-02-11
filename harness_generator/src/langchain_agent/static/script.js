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
    setChecked("cfg_use_docker", cfg.fuzz_use_docker !== false);
    setValue("cfg_docker_image", cfg.fuzz_docker_image || "auto");

    setValue("oss_fuzz_dir", cfg.oss_fuzz_dir || "");

    setValue("sherpa_git_mirrors", cfg.sherpa_git_mirrors || "");
    setValue("sherpa_docker_http_proxy", cfg.sherpa_docker_http_proxy || "");
    setValue("sherpa_docker_https_proxy", cfg.sherpa_docker_https_proxy || "");
    setValue("sherpa_docker_no_proxy", cfg.sherpa_docker_no_proxy || "");
    setValue("sherpa_docker_proxy_host", cfg.sherpa_docker_proxy_host || "host.docker.internal");

    // Mirror config into the fuzz form defaults
    setValue("time_budget", String(cfg.fuzz_time_budget ?? 900));
    setChecked("use_docker", cfg.fuzz_use_docker !== false);
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
    fuzz_use_docker: !!getChecked("cfg_use_docker"),
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

// ===== 系统状态监控 =====
let _systemPollTimer = null;

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
    if (!res.ok) throw new Error(`HTTP 错误 ${res.status}`);
    const data = await res.json();
    if (!data.ok) throw new Error("保存失败");

    // Update fuzz form defaults immediately
    setValue("time_budget", String(cfg.fuzz_time_budget));
    setChecked("use_docker", cfg.fuzz_use_docker);
    setValue("docker_image", cfg.fuzz_docker_image);

    statusEl.className = "result-box success";
    statusEl.innerHTML = '<span class="status-icon">✅</span> 配置已保存并立即生效（已持久化）。';
  } catch (err) {
    statusEl.className = "result-box error";
    statusEl.innerHTML = `<span class="status-icon">❌</span> <strong>保存失败：</strong> ${escapeHtml(err.message)}`;
  }
}

document.getElementById("cfg_save_btn")?.addEventListener("click", async () => {
  await saveConfigFromForm();
});

document.getElementById("refresh_system_btn")?.addEventListener("click", async () => {
  await pollSystemStatus();
});

document.addEventListener("DOMContentLoaded", async () => {
  await loadConfigIntoForm();
  startSystemPolling();
});

// ===== 模糊测试功能 =====
document.getElementById("fuzz_btn").addEventListener("click", async () => {
  const codeUrl = document.getElementById("code_url").value.trim();
  const email = document.getElementById("email").value.trim();
  const timeBudgetRaw = (document.getElementById("time_budget")?.value || "900").trim();
  const useDocker = !!document.getElementById("use_docker")?.checked;
  const dockerImage = (document.getElementById("docker_image")?.value || "sherpa-fuzz:latest").trim();
  const statusEl = document.getElementById("fuzz_status");
  const progressEl = document.getElementById("fuzz_progress");
  const logEl = document.getElementById("fuzz_log");
  const btn = document.getElementById("fuzz_btn");

  if (!codeUrl) {
    alert("请输入代码仓库地址");
    return;
  }

  const timeBudget = Number.parseInt(timeBudgetRaw || "900", 10);
  if (!Number.isFinite(timeBudget) || timeBudget <= 0) {
    alert("请输入有效的运行时长（秒）");
    return;
  }

  // 禁用按钮，显示加载状态
  btn.disabled = true;
  statusEl.style.display = "block";
  statusEl.className = "result-box loading";
  statusEl.innerHTML = '<span class="status-icon">⏳</span> 已提交任务，正在排队...';
  progressEl.style.display = "block";
  progressEl.className = "result-box";
  progressEl.innerHTML = "等待任务初始化...";
  logEl.style.display = "block";
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
            docker_image: dockerImage
          }
        ]
      }),
    });

    if (!res.ok) {
      throw new Error(`HTTP 错误 ${res.status}`);
    }

    const data = await res.json();
    if (!data.job_id) {
      throw new Error("服务端未返回 job_id");
    }

    const jobId = data.job_id;
    statusEl.className = "result-box loading";
    statusEl.innerHTML = `<span class="status-icon">⏳</span> 任务已创建：<strong>${escapeHtml(jobId)}</strong>（轮询中...）`;

    // Poll status until finished.
    let lastLog = "";
    const poll = async () => {
      const r = await fetch(`/api/task/${encodeURIComponent(jobId)}`);
      if (!r.ok) throw new Error(`轮询失败 HTTP ${r.status}`);
      const j = await r.json();
      if (j.error === "job_not_found") throw new Error("任务不存在或已被清理");

      const st = j.status || "unknown";
      const child = Array.isArray(j.children) && j.children.length ? j.children[0] : null;
      const err = (child && child.error) || j.error;
      const result = (child && child.result) || j.result;
      const log = ((child && child.log) || j.log || "").toString();
      const logFile = (child && child.log_file) || j.log_file;
      if (log !== lastLog) {
        lastLog = log;
        const summary = summarizeLog(log, st, result, err, logFile);
        logEl.innerHTML = `<strong>进度摘要（实时）：</strong><pre style="white-space: pre-wrap; margin-top: 8px;">${escapeHtml(summary)}</pre>`;
        logEl.scrollTop = logEl.scrollHeight;

        const progress = renderProgressFromLog(log);
        progressEl.innerHTML = progress;
      }

      if (st === "queued" || st === "running") {
        statusEl.className = "result-box loading";
        statusEl.innerHTML = `<span class="status-icon">⏳</span> 当前状态：<strong>${escapeHtml(st)}</strong>（任务：${escapeHtml(jobId)}）`;
        return false;
      }

      if (st === "success") {
        statusEl.className = "result-box success";
        statusEl.innerHTML = `<span class="status-icon">✅</span> 完成（任务：<strong>${escapeHtml(jobId)}</strong>）<br>${escapeHtml(result || "")}`;
        return true;
      }

      statusEl.className = "result-box error";
      statusEl.innerHTML = `<span class="status-icon">❌</span> 失败（任务：<strong>${escapeHtml(jobId)}</strong>）<br>${escapeHtml(err || "unknown error")}`;
      return true;
    };

    for (let i = 0; i < 3600; i++) {
      const done = await poll();
      if (done) break;
      await new Promise((r) => setTimeout(r, 2000));
    }
  } catch (err) {
    statusEl.className = "result-box error";
    statusEl.innerHTML = `<span class="status-icon">❌</span> <strong>测试失败：</strong> ${escapeHtml(err.message)}`;
  } finally {
    btn.disabled = false;
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
  return text.replace(/[&<>"']/g, (m) => map[m]);
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

function summarizeLog(log, status, result, err, logFile) {
  const lines = (log || "").split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const important = [];
  const wfRe = /^\[wf\b.*\]/;
  const jobRe = /^\[job\b.*\]/;
  const buildRe = /(build failed|linking|cmake|compile|error:|fatal|undefined reference)/i;
  const crashRe = /(crash|artifact|asan|ubsan|msan|tsan|segmentation|sanitizer)/i;
  const stepRe = /(->|<-|workflow end|pass [a-e]|ready)/i;

  for (const line of lines) {
    if (wfRe.test(line) && stepRe.test(line)) {
      important.push(line);
      continue;
    }
    if (jobRe.test(line) && (line.includes("start") || line.includes("params"))) {
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

  const items = steps.map((s) => {
    const st = parsed.steps[s] || "pending";
    const color = statusToColor[st] || "#9ca3af";
    return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
      <span style="width:8px;height:8px;border-radius:50%;background:${color};display:inline-block;"></span>
      <span>${labels[s]}</span>
      <span style="color:#6d6f73;font-size:12px;">(${st})</span>
    </div>`;
  }).join("");

  const repoLine = parsed.repoRoot ? `<div style="margin-top:6px;color:#6d6f73;font-size:12px;">输出目录：${escapeHtml(parsed.repoRoot)}</div>` : "";
  const lastErr = parsed.lastError ? `<div style="margin-top:8px;color:#b91c1c;font-size:12px;">错误：${escapeHtml(parsed.lastError)}</div>` : "";
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

    if (/Missing fuzz\/build\.py|OpenCodeHelper|opencode/i.test(line)) {
      lastError = line.replace(/^\[.*?\]\s*/, "");
    }
  }

  return { steps, repoRoot, lastError };
}
