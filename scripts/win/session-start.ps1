# CLARA SessionStart hook (native Windows): inject the memory context block.
#
# Contract: ALWAYS exits 0 — memory must never block a session. stdout is the
# context Claude sees; stderr carries diagnostics. Mirror of session-start.sh.
# PowerShell 5.1-compatible.

$ErrorActionPreference = "Continue"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($env:CLAUDE_PLUGIN_DATA) { $dataDir = $env:CLAUDE_PLUGIN_DATA }
elseif ($env:CLARA_HOME) { $dataDir = Join-Path $env:CLARA_HOME "plugin" }
else { $dataDir = Join-Path $HOME ".clara\plugin" }

if ($env:CLARA_HOME) { $base = $env:CLARA_HOME }
else { $base = Join-Path $HOME ".clara" }

# Session-cwd hint for the long-lived MCP server (see session-start.sh).
if ($env:CLAUDE_SESSION_ID) {
    try {
        $hintDir = Join-Path $base "session-cwd"
        $null = New-Item -ItemType Directory -Force $hintDir
        Set-Content -Path (Join-Path $hintDir $env:CLAUDE_SESSION_ID) `
            -Value (Get-Location).Path -Encoding UTF8 -NoNewline
    } catch {}
}

& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scriptDir "bootstrap.ps1")
$rc = $LASTEXITCODE

if ($rc -eq 3) {
    Write-Output "CLARA is installing in the background - memory will be available next session."
    exit 0
}
if ($rc -ne 0) {
    # Bootstrap already explained itself on stderr; never block the session.
    exit 0
}

# Resolve the venv python: junction `current` or the pointer-file fallback.
$py = $null
$current = Join-Path $dataDir "current"
foreach ($cand in @((Join-Path $current "Scripts\python.exe"), (Join-Path $current "bin\python"))) {
    if (Test-Path $cand) { $py = $cand; break }
}
if (-not $py) {
    $pointer = Join-Path $dataDir "current.path"
    if (Test-Path $pointer) {
        $venv = (Get-Content $pointer -Raw).Trim()
        $cand = Join-Path $venv "Scripts\python.exe"
        if (Test-Path $cand) { $py = $cand }
    }
}

if ($py) {
    try {
        & $py -m clara.fastpath.context --cwd (Get-Location).Path
    } catch {}
}
exit 0
