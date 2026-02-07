// ===== 持久化配置（加载/保存） =====
async function loadConfigIntoForm() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`HTTP 错误 ${res.status}`);
    const cfg = await res.json();

    setValue("openai_api_key", cfg.openai_api_key || "");
    setValue("openai_base_url", cfg.openai_base_url || "");
    setValue("openrouter_api_key", cfg.openrouter_api_key || "");
    setValue("openrouter_base_url", cfg.openrouter_base_url || "https://openrouter.ai/api/v1");
    setValue("openrouter_model", cfg.openrouter_model || "anthropic/claude-3.5-sonnet");

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

  return {
    openai_api_key: (getValue("openai_api_key") || "").trim() || null,
    openai_base_url: (getValue("openai_base_url") || "").trim() || "",
    openrouter_api_key: (getValue("openrouter_api_key") || "").trim() || null,
    openrouter_base_url: (getValue("openrouter_base_url") || "").trim() || "https://openrouter.ai/api/v1",
    openrouter_model: (getValue("openrouter_model") || "").trim() || "anthropic/claude-3.5-sonnet",

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

document.addEventListener("DOMContentLoaded", async () => {
  await loadConfigIntoForm();
});

// ===== 模糊测试功能 =====
document.getElementById("fuzz_btn").addEventListener("click", async () => {
  const codeUrl = document.getElementById("code_url").value.trim();
  const email = document.getElementById("email").value.trim();
  const timeBudgetRaw = (document.getElementById("time_budget")?.value || "900").trim();
  const useDocker = !!document.getElementById("use_docker")?.checked;
  const dockerImage = (document.getElementById("docker_image")?.value || "sherpa-fuzz:latest").trim();
  const statusEl = document.getElementById("fuzz_status");
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
  logEl.style.display = "block";
  logEl.className = "result-box";
  logEl.innerHTML = "";

  try {
    const res = await fetch("/fuzz_code", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        code_url: codeUrl, 
        email: email || null,
        time_budget: timeBudget,
        docker: useDocker,
        docker_image: dockerImage
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
      const r = await fetch(`/api/fuzz/${encodeURIComponent(jobId)}`);
      if (!r.ok) throw new Error(`轮询失败 HTTP ${r.status}`);
      const j = await r.json();
      if (j.error === "job_not_found") throw new Error("任务不存在或已被清理");

      const st = j.status || "unknown";
      const err = j.error;
      const result = j.result;
      const log = (j.log || "").toString();
      if (log !== lastLog) {
        lastLog = log;
        logEl.innerHTML = `<strong>运行日志（末尾截断）：</strong><pre style="white-space: pre-wrap; margin-top: 8px;">${escapeHtml(log)}</pre>`;
        logEl.scrollTop = logEl.scrollHeight;
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
