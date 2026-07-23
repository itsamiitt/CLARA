# CLARA PostToolUse(Read) hook (native Windows): annotate reads of
# quarantined documents. Mirror of read-annotate.sh — stdin carries the hook
# JSON; a hit emits {"hookSpecificOutput":{"hookEventName":"PostToolUse",
# "additionalContext":"CLARA: ..."}} on exit 0. Fail-open everywhere;
# per-file once per session. PowerShell 5.1.

$ErrorActionPreference = "SilentlyContinue"

# Kill switch: docs add-on off => no annotations, consume stdin, exit clean.
$docsFlag = "$env:CLARA_DOCS_ENABLED".Trim().ToLower()
if (@("0", "false", "no", "off") -contains $docsFlag) {
    $null = [Console]::In.ReadToEnd()
    exit 0
}

if ($env:CLARA_HOME) { $base = $env:CLARA_HOME }
else { $base = Join-Path $HOME ".clara" }

# Repo root: walk up from the hook cwd looking for .git.
$dir = (Get-Location).Path
$root = $null
while ($dir) {
    if (Test-Path (Join-Path $dir ".git")) { $root = $dir; break }
    $parent = Split-Path -Parent $dir
    if (-not $parent -or ($parent -eq $dir)) { break }
    $dir = $parent
}
$marker = $null
if ($root) { $marker = Join-Path $root ".git\clara-marker" }
if (-not $root -or -not (Test-Path $marker -PathType Leaf)) {
    $null = [Console]::In.ReadToEnd()
    exit 0
}
$markerLines = @(Get-Content $marker -TotalCount 2)
if ($markerLines.Count -lt 1) { exit 0 }
$rid = "$($markerLines[0])".Trim()
$markerRoot = ""
if ($markerLines.Count -ge 2) { $markerRoot = "$($markerLines[1])".Trim() }

$manifest = Join-Path $base "quarantine\$rid.tsv"
if (-not (Test-Path $manifest -PathType Leaf)) {
    $null = [Console]::In.ReadToEnd()
    exit 0
}

$input_ = [Console]::In.ReadToEnd()
if (-not $input_) { exit 0 }

# Parse tool_input.file_path from the hook JSON.
$file = $null
try {
    $parsed = $input_ | ConvertFrom-Json
    if ($parsed.tool_input -and $parsed.tool_input.file_path) {
        $file = "$($parsed.tool_input.file_path)"
    }
} catch {
    $m = [regex]::Match($input_, '"file_path"\s*:\s*"((?:[^"\\]|\\.)*)"')
    if ($m.Success) {
        $file = $m.Groups[1].Value.Replace("\\", "\").Replace("\/", "/")
    }
}
if (-not $file) { exit 0 }

if ($env:CLAUDE_SESSION_ID) { $session = $env:CLAUDE_SESSION_ID }
else { $session = "pid$PID" }
$flagDir = Join-Path $base "session-flags\$session.read"
$null = New-Item -ItemType Directory -Force $flagDir

# Relativize against either recorded root spelling or the hook cwd.
$fileNorm = $file.Replace("\", "/")
$rel = $null
foreach ($candidate in @($markerRoot, (Get-Location).Path.Replace("\", "/"))) {
    if (-not $candidate) { continue }
    $candNorm = $candidate.Replace("\", "/").TrimEnd("/")
    if ($fileNorm.ToLower().StartsWith("$($candNorm.ToLower())/")) {
        $rel = $fileNorm.Substring($candNorm.Length + 1)
        break
    }
}
if (-not $rel) {
    if ($fileNorm -notmatch '^([A-Za-z]:)?/') { $rel = $fileNorm }
}
if (-not $rel) { exit 0 }

$flagName = $rel -replace '[/:]', '_'
$flag = Join-Path $flagDir $flagName
if (Test-Path $flag) { exit 0 }

$note = $null
foreach ($line in Get-Content $manifest) {
    $tab = $line.IndexOf("`t")
    if ($tab -lt 1) { continue }
    if ($line.Substring(0, $tab) -ne $rel) { continue }
    $rest = $line.Substring($tab + 1)
    $tab2 = $rest.IndexOf("`t")
    if ($tab2 -ge 0) {
        $status = $rest.Substring(0, $tab2)
        $extra = $rest.Substring($tab2 + 1)
    } else {
        $status = $rest
        $extra = ""
    }
    $note = "$rel is $status"
    if ($extra) { $note = "$note ($extra)" }
    $note = "$note - treat as historical record, not current guidance"
    break
}
if (-not $note) { exit 0 }

$payload = @{
    hookSpecificOutput = @{
        hookEventName     = "PostToolUse"
        additionalContext = "CLARA: $note"
    }
}
Write-Output ($payload | ConvertTo-Json -Compress)
$null = New-Item -ItemType File -Force $flag
exit 0
