param(
  [string]$CodeUrl = "https://github.com/google/fuzzer-test-suite",
  [int]$TimeBudgetSec = 600,
  [int]$MaxTokens = 1000,
  [string]$Model = "anthropic/claude-3.5-sonnet",
  [switch]$Docker,
  [string]$DockerImage = "sherpa-fuzz-cpp:latest",
  [string]$BaseUrl = "http://127.0.0.1:8000",
  [int]$PollIntervalSec = 5,
  [int]$TailLines = 80
)

$ErrorActionPreference = "Stop"

function Test-PortListening {
  param([int]$Port)
  $conn = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
  return $null -ne $conn
}

function Wait-HttpOk {
  param([string]$Url, [int]$TimeoutSec)
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    try {
      $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 5
      if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 400) {
        return $true
      }
    } catch {
      Start-Sleep -Seconds 1
    }
  }
  return $false
}

function Tail-File {
  param([string]$Path, [int]$Lines)
  if (Test-Path $Path) {
    Get-Content -Path $Path -Tail $Lines
  } else {
    Write-Host "(log file not found yet) $Path" -ForegroundColor DarkGray
  }
}

function Show-DockerSnapshot {
  Write-Host "--- docker ps (sherpa-fuzz-cpp) ---" -ForegroundColor DarkCyan
  try {
    docker ps --format "table {{.Image}}\t{{.Status}}\t{{.Names}}" | Select-String -Pattern "sherpa-fuzz-cpp" -SimpleMatch
  } catch {
    Write-Host "docker ps failed: $($_.Exception.Message)" -ForegroundColor DarkGray
  }
}

$repoRoot = (Get-Location).Path
$serverScript = Join-Path $repoRoot "harness_generator/src/langchain_agent/main.py"
$pythonExe = Join-Path $repoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path $pythonExe)) {
  $pythonExe = "python"
}

if (-not (Test-Path $serverScript)) {
  throw "Server script not found: $serverScript"
}

if (-not $Docker.IsPresent) {
  $Docker = $true
}

Write-Host "[1/5] Ensure service is running on 127.0.0.1:8000" -ForegroundColor Cyan
if (-not (Test-PortListening -Port 8000)) {
  Write-Host "Starting server in background: $pythonExe $serverScript" -ForegroundColor DarkGray
  Start-Process -FilePath $pythonExe -ArgumentList @($serverScript) -WindowStyle Hidden | Out-Null
}

if (-not (Wait-HttpOk -Url "$BaseUrl/docs" -TimeoutSec 20)) {
  throw "Service not responding at $BaseUrl/docs"
}

Write-Host "[2/5] Submit fuzz job" -ForegroundColor Cyan
$bodyObj = @{
  code_url = $CodeUrl
  docker = [bool]$Docker
  docker_image = if ($Docker) { $DockerImage } else { "auto" }
  time_budget = $TimeBudgetSec
  max_tokens = $MaxTokens
  model = $Model
}
$bodyJson = $bodyObj | ConvertTo-Json
$submit = Invoke-RestMethod -Method Post -Uri "$BaseUrl/fuzz_code" -ContentType "application/json" -Body $bodyJson
$jobId = $submit.job_id
if (-not $jobId) {
  throw "No job_id returned: $($submit | ConvertTo-Json -Depth 10)"
}
Write-Host "job_id=$jobId" -ForegroundColor Green

$logPath = Join-Path $repoRoot "config/logs/jobs/$jobId.log"
Write-Host "log_path=$logPath" -ForegroundColor DarkGray

Write-Host "[3/5] Poll status + tail log (Ctrl+C to stop polling; job keeps running)" -ForegroundColor Cyan
$started = Get-Date
$maxWaitSec = [Math]::Max($TimeBudgetSec + 600, 900)
$deadline = (Get-Date).AddSeconds($maxWaitSec)

while ((Get-Date) -lt $deadline) {
  $job = Invoke-RestMethod -Uri "$BaseUrl/api/fuzz/$jobId"
  $ts = Get-Date -Format "HH:mm:ss"
  $status = $job.status

  Write-Host "[$ts] status=$status updated_at=$($job.updated_at)" -ForegroundColor Yellow

  if ($job.error) {
    Write-Host "error=$($job.error)" -ForegroundColor Red
  }

  Tail-File -Path $logPath -Lines $TailLines
  Show-DockerSnapshot
  Write-Host "" 

  if ($status -eq "success" -or $status -eq "error") {
    break
  }

  Start-Sleep -Seconds $PollIntervalSec
}

Write-Host "[4/5] Final status" -ForegroundColor Cyan
$final = Invoke-RestMethod -Uri "$BaseUrl/api/fuzz/$jobId"
$final | ConvertTo-Json -Depth 10

Write-Host "[5/5] Tips" -ForegroundColor Cyan
Write-Host "- Open the disk log to see full progress: $logPath" -ForegroundColor DarkGray
Write-Host "- If Docker containers are too short-lived, run 'docker ps -a' during Pass C/E." -ForegroundColor DarkGray
