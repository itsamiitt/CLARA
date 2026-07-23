# CLARA MCP launcher (native Windows): bootstrap, then run clara-mcp.exe.
#
# Fallback/manual path — the plugin's mcpServers entry normally spawns the
# shim exe directly. stdout discipline: only the MCP protocol touches stdout.
# PowerShell 5.1.

$ErrorActionPreference = "Continue"

function Write-ClaraLog([string]$Message) {
    [Console]::Error.WriteLine("clara: $Message")
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($env:CLAUDE_PLUGIN_DATA) { $dataDir = $env:CLAUDE_PLUGIN_DATA }
elseif ($env:CLARA_HOME) { $dataDir = Join-Path $env:CLARA_HOME "plugin" }
else { $dataDir = Join-Path $HOME ".clara\plugin" }

& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scriptDir "bootstrap.ps1")
$rc = $LASTEXITCODE

function Find-ClaraMcp([string]$DataDir) {
    foreach ($cand in @(
        (Join-Path $DataDir "shim\clara-mcp.exe"),
        (Join-Path $DataDir "current\Scripts\clara-mcp.exe")
    )) {
        if (Test-Path $cand -PathType Leaf) { return $cand }
    }
    $pointer = Join-Path $DataDir "current.path"
    if (Test-Path $pointer) {
        $venv = (Get-Content $pointer -Raw).Trim()
        $cand = Join-Path $venv "Scripts\clara-mcp.exe"
        if (Test-Path $cand -PathType Leaf) { return $cand }
    }
    return $null
}

if ($rc -eq 3) {
    $waited = 0
    while ($waited -lt 20) {
        if (Find-ClaraMcp $dataDir) { break }
        Start-Sleep -Seconds 1
        $waited += 1
    }
}

$bin = Find-ClaraMcp $dataDir
if ($bin) {
    & $bin
    exit $LASTEXITCODE
}

Write-ClaraLog "memory server not ready - install still running or failed; see $dataDir\install.log"
exit 1
