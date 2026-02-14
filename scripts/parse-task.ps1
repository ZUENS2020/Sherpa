param(
    [Parameter(Mandatory = $true)]
    [string]$TaskId,

    [string]$BaseUrl = "http://localhost:8000",

    [string]$Network = ""
)

$pythonScript = @'
import json
import os
import re
import urllib.request

task_id = os.environ["TASK_ID"].strip()
base_url = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")

candidates = [
    f"{base_url}/api/task/{task_id}",
    f"http://host.docker.internal:8000/api/task/{task_id}",
    f"http://sherpa-web:8000/api/task/{task_id}",
]

data = None
used_url = None
last_error = None
for url in candidates:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
            used_url = url
            break
    except Exception as exc:
        last_error = exc

if data is None:
    raise SystemExit(f"Failed to fetch task JSON. last_error={last_error}")

print("source_url:", used_url)
print("task_id:", data.get("job_id"))
print("status:", data.get("status"))
print("created_at:", data.get("created_at"))
print("updated_at:", data.get("updated_at"))

children = data.get("children") or []
print("children:", len(children))
for idx, child in enumerate(children, start=1):
    log = child.get("log") or ""
    lines = log.splitlines()
    wf_lines = [ln for ln in lines if "[wf" in ln]
    key_lines = [
        ln for ln in lines
        if re.search(r"\[wf|OpenCodeHelper|build|error|fail|crash|timeout", ln, re.IGNORECASE)
    ]

    print(f"child[{idx}] id:", child.get("job_id"))
    print(f"child[{idx}] status:", child.get("status"))
    print(f"child[{idx}] log_lines:", len(lines))
    if wf_lines:
        print(f"child[{idx}] wf_last:", wf_lines[-1])
    print(f"child[{idx}] key_tail:")
    for line in key_lines[-8:]:
        print(" ", line)
'@

$dockerArgs = @(
    "run",
    "--rm",
    "--add-host", "host.docker.internal:host-gateway"
)

if ($Network -ne "") {
    $dockerArgs += @("--network", $Network)
}

$tempFile = [System.IO.Path]::GetTempFileName()
$tempScript = [System.IO.Path]::ChangeExtension($tempFile, ".py")
Move-Item -Path $tempFile -Destination $tempScript -Force
Set-Content -Path $tempScript -Value $pythonScript -Encoding UTF8

try {
    $dockerArgs += @(
        "-v", "${tempScript}:/tmp/task_parser.py:ro",
        "-e", "TASK_ID=$TaskId",
        "-e", "BASE_URL=$BaseUrl",
        "python:3.11-slim",
        "python",
        "/tmp/task_parser.py"
    )

    docker @dockerArgs
}
finally {
    Remove-Item -Path $tempScript -ErrorAction SilentlyContinue
}
