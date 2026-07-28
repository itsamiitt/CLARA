# CLARA PostToolUse hook (native Windows fallback): journal file/repo changes.
#
# Reached only when change-capture.cmd could not find the venv python at the
# standard junction path — it resolves the pointer file instead. stdin (the
# hook JSON) is piped through to the module. Fail-open: any miss exits 0
# silently. PowerShell 5.1.

$ErrorActionPreference = "Continue"

$memFlag = "$env:CLARA_MEMORY_ENABLED".Trim().ToLower()
if (@("0", "false", "no", "off") -contains $memFlag) {
    $null = $input  # drain stdin
    exit 0
}

if ($env:CLAUDE_PLUGIN_DATA) { $dataDir = $env:CLAUDE_PLUGIN_DATA }
elseif ($env:CLARA_HOME) { $dataDir = Join-Path $env:CLARA_HOME "plugin" }
else { $dataDir = Join-Path $HOME ".clara\plugin" }

$py = $null
$direct = Join-Path $dataDir "current\Scripts\python.exe"
if (Test-Path $direct -PathType Leaf) { $py = $direct }
if (-not $py) {
    $pointer = Join-Path $dataDir "current.path"
    if (Test-Path $pointer -PathType Leaf) {
        try {
            $venv = (Get-Content $pointer -Raw).Trim()
            $cand = Join-Path $venv "Scripts\python.exe"
            if (Test-Path $cand -PathType Leaf) { $py = $cand }
        } catch {}
    }
}
if (-not $py) {
    $null = $input  # drain stdin
    exit 0
}

$input | & $py -m clara.fastpath.change_capture | Out-Null
exit 0
