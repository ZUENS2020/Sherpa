// 测试工作流日志解析器

const testLog = `[wf step=0 last=- next=init] -> init
[job 0adcd662] start repo=https://github.com/user/repo.git
[job 0adcd662] params docker=True time_budget=900s
[wf step=1 last=init next=plan] <- init ok repo_root=C:\\output\\fuzz-samples-0adcd662 dt=2.45s
[wf step=1 last=init next=plan] -> plan
[wf step=2 last=plan next=synthesize] <- plan ok dt=5.23s
[wf step=2 last=plan next=synthesize] -> synthesize
[wf step=3 last=synthesize next=build] <- synthesize ok dt=8.12s
[wf step=3 last=synthesize next=build] -> build attempt=1/2 -> python3 fuzz/build.py
[wf step=3 last=build next=build] <- build err=local variable 'translated_cmd' referenced before assignment dt=22ms
[wf step=4 last=build next=build] -> decide (max_steps=20)
[wf step=4 last=build next=fix_build] <- decide (llm) next=fix_build hint=yes dt=2.83s
[wf step=4 last=build next=fix_build] -> fix_build`;

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
            console.log("[START]", line, "=> Step:", startMatch[1], "=> Status: running");
        }

        const endMatch = line.match(endRe);
        if (endMatch) {
            const step = endMatch[1];
            if (/err=|failed|error/i.test(line)) {
                steps[step] = "error";
                lastError = line.replace(/^\[.*?\]\s*/, "");
                console.log("[END ERROR]", line, "=> Step:", step, "=> Status: error");
            } else {
                steps[step] = "done";
                console.log("[END DONE]", line, "=> Step:", step, "=> Status: done");
            }
        }
    }

    return { steps, repoRoot, lastError };
}

function renderProgressFromLog(log) {
    const parsed = parseWorkflowLog(log || "");
    console.log("\n=== Parsed Steps ===");
    console.log(parsed);

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

    const items = Object.keys(parsed.steps).map((s) => {
        const st = parsed.steps[s] || "pending";
        const color = statusToColor[st] || "#9ca3af";
        return `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
            <span style="width:8px;height:8px;border-radius:50%;background:${color};display:inline-block;"></span>
            <span>${labels[s] || s}</span>
            <span style="color:#6d6f73;font-size:12px;">(${st})</span>
        </div>`;
    }).join("");

    return items || "没有检测到工作流步骤";
}

console.log("=== Testing Workflow Log Parser ===\n");
const result = renderProgressFromLog(testLog);
console.log("\n=== Rendered HTML ===");
console.log(result);
