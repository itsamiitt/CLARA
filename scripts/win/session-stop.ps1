# CLARA Stop hook (native Windows): one debounced nudge when a plan doc
# looks complete. Mirror of session-stop.sh: pure file reads of fastpath
# sidecars, always exits 0, fail-open on every branch. PowerShell 5.1.

$ErrorActionPreference = "SilentlyContinue"

# Kill switch: the proposals nudge is doc-curator behavior.
$docsFlag = "$env:CLARA_DOCS_ENABLED".Trim().ToLower()
if (@("0", "false", "no", "off") -contains $docsFlag) { exit 0 }

if ($env:CLARA_HOME) { $base = $env:CLARA_HOME }
else { $base = Join-Path $HOME ".clara" }
$proposalsDir = Join-Path $base "proposals"
if (-not (Test-Path $proposalsDir)) { exit 0 }

if ($env:CLAUDE_SESSION_ID) { $session = $env:CLAUDE_SESSION_ID }
else { $session = "pid$PID" }
$flagDir = Join-Path $base "session-flags"
$flag = Join-Path $flagDir "$session.done"
if (Test-Path $flag) { exit 0 }

# Repo root: walk up from the hook cwd looking for .git (no subprocess).
$dir = (Get-Location).Path
$root = $null
while ($dir) {
    if (Test-Path (Join-Path $dir ".git")) { $root = $dir; break }
    $parent = Split-Path -Parent $dir
    if (-not $parent -or ($parent -eq $dir)) { break }
    $dir = $parent
}
if (-not $root) { exit 0 }

$rid = $null
$marker = Join-Path $root ".git\clara-marker"
if (Test-Path $marker -PathType Leaf) {
    $lines = @(Get-Content $marker -TotalCount 1)
    if ($lines.Count -ge 1) { $rid = "$($lines[0])".Trim() }
}
if (-not $rid) {
    $index = Join-Path $proposalsDir "index.tsv"
    if (Test-Path $index) {
        # index.tsv records both native and MSYS spellings of each root.
        $rootNorm = $root.TrimEnd("\").ToLower()
        foreach ($line in Get-Content $index) {
            $parts = $line -split "`t"
            if ($parts.Count -lt 2) { continue }
            $idxRoot = $parts[0].Replace("/", "\").TrimEnd("\").ToLower()
            if ($idxRoot -eq $rootNorm) { $rid = $parts[1].Trim(); break }
        }
    }
}
if (-not $rid) { exit 0 }

$proposals = Join-Path $proposalsDir "$rid.txt"
if (-not (Test-Path $proposals -PathType Leaf)) { exit 0 }
$firstLine = Get-Content $proposals -TotalCount 1
if (-not $firstLine) { exit 0 }
$firstPath = ($firstLine -split "`t")[0]
if (-not $firstPath) { exit 0 }

Write-Output "CLARA: $firstPath looks complete - consider /clara:done"
$null = New-Item -ItemType Directory -Force $flagDir
$null = New-Item -ItemType File -Force $flag
exit 0
