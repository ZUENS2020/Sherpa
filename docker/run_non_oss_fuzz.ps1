param(
  [Parameter(Mandatory=$true)][string]$Repo,
  [string]$Ref = "",
  [string]$Image = "auto",
  [int]$TimeBudget = 900,
  [int]$MaxLen = 1024,
  [string]$Sanitizer = "address"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path "$PSScriptRoot\..\").Path
Write-Host "[+] Docker mode enabled (image=$Image). Image build is handled automatically by the tool when missing."

Write-Host "[+] Running non-OSS workflow (build/run inside Docker) ..."
$cmd = @(
  "python",
  "harness_generator/src/fuzz_unharnessed_repo.py",
  "--repo", $Repo,
  "--time-budget", "$TimeBudget",
  "--max-len", "$MaxLen",
  "--sanitizer", $Sanitizer,
  "--docker-image", $Image
)
if ($Ref -ne "") {
  $cmd += @("--ref", $Ref)
}

Push-Location $root
try {
  & $cmd[0] $cmd[1..($cmd.Length-1)]
} finally {
  Pop-Location
}
