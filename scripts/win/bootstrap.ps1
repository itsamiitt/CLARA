# CLARA plugin bootstrap (native Windows): ensure a venv with clara-memory[mcp].
#
# Mirror of scripts/bootstrap.sh — same exit codes (0 ready / 1 no Python
# 3.10+ / 3 install running), same layout under $env:CLAUDE_PLUGIN_DATA
# (default ~\.clara\plugin), same stdout discipline (stderr only).
# PowerShell 5.1-compatible: no &&/||, no ternary.
#
# Windows specifics: `current` is an NTFS junction (no admin needed) with a
# plain-text `current.path` pointer file as fallback; the MCP shim is a COPY
# of clara-mcp.exe (pip console-script .exes embed the venv python's absolute
# path, so a copy still runs).

param([switch]$InstallWorker)

$ErrorActionPreference = "Continue"

function Write-ClaraLog([string]$Message) {
    [Console]::Error.WriteLine("clara: $Message")
}

function Get-ClaraDataDir {
    if ($env:CLAUDE_PLUGIN_DATA) { return $env:CLAUDE_PLUGIN_DATA }
    if ($env:CLARA_HOME) { return (Join-Path $env:CLARA_HOME "plugin") }
    return (Join-Path $HOME ".clara\plugin")
}

function Find-VenvBin([string]$Venv, [string]$Name) {
    foreach ($cand in @("Scripts\$Name.exe", "Scripts\$Name", "bin\$Name")) {
        $path = Join-Path $Venv $cand
        if (Test-Path $path -PathType Leaf) { return $path }
    }
    return $null
}

function Set-CurrentPointer([string]$DataDir, [string]$Venv) {
    $current = Join-Path $DataDir "current"
    $pointer = Join-Path $DataDir "current.path"
    try {
        if (Test-Path $current) {
            $item = Get-Item $current -Force
            if ($item.LinkType) { $item.Delete() } else { Remove-Item $current -Recurse -Force -Confirm:$false }
        }
        $null = New-Item -ItemType Junction -Path $current -Target $Venv -ErrorAction Stop
    } catch {
        # Junction refused (non-NTFS, policy): fall back to the pointer file.
        try { Set-Content -Path $pointer -Value $Venv -Encoding Ascii -NoNewline } catch {}
        return
    }
    # Junction succeeded — keep the pointer in sync for readers that use it.
    try { Set-Content -Path $pointer -Value $Venv -Encoding Ascii -NoNewline } catch {}
}

function Update-ClaraShim([string]$DataDir, [string]$Venv) {
    $bin = Find-VenvBin $Venv "clara-mcp"
    if (-not $bin) { return $false }
    $shimDir = Join-Path $DataDir "shim"
    $null = New-Item -ItemType Directory -Force $shimDir
    try {
        Copy-Item -Path $bin -Destination (Join-Path $shimDir "clara-mcp.exe") -Force
        return $true
    } catch {
        return $false
    }
}

function Find-ClaraPython {
    $candidates = @(
        @{ Cmd = "py"; Args = @("-3.13") }, @{ Cmd = "py"; Args = @("-3.12") },
        @{ Cmd = "py"; Args = @("-3.11") }, @{ Cmd = "py"; Args = @("-3.10") },
        @{ Cmd = "python"; Args = @() }, @{ Cmd = "python3"; Args = @() }
    )
    $probe = "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
    foreach ($cand in $candidates) {
        $exe = Get-Command $cand.Cmd -ErrorAction SilentlyContinue
        if (-not $exe) { continue }
        & $cand.Cmd @($cand.Args + @("-c", $probe)) 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return @{ Cmd = $cand.Cmd; Args = $cand.Args }
        }
    }
    return $null
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($env:CLAUDE_PLUGIN_ROOT) { $pluginRoot = $env:CLAUDE_PLUGIN_ROOT }
else { $pluginRoot = Split-Path -Parent $scriptDir }  # scripts\win -> scripts
if ((Split-Path -Leaf $pluginRoot) -eq "scripts") { $pluginRoot = Split-Path -Parent $pluginRoot }
$dataDir = Get-ClaraDataDir

# ---------------------------------------------------------------------------
# Detached worker (spawned below; output redirected to install.log)
# ---------------------------------------------------------------------------
if ($InstallWorker) {
    $py = $env:CLARA_BS_PY
    $pyArgs = @()
    if ($env:CLARA_BS_PYARGS) { $pyArgs = $env:CLARA_BS_PYARGS -split " " }
    $venv = $env:CLARA_BS_VENV
    $root = $env:CLARA_BS_ROOT
    $data = $env:CLARA_BS_DATA
    $flag = Join-Path $data ".installing"
    $lock = Join-Path $data ".lock"

    Write-Output "=== clara install started: $(Get-Date) ==="
    Write-Output "python: $py $pyArgs  venv: $venv"
    $status = 1
    if (Test-Path $venv) { Remove-Item $venv -Recurse -Force -Confirm:$false }

    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        & uv venv --python $py $venv
        if ($LASTEXITCODE -eq 0) {
            $vpy = Find-VenvBin $venv "python"
            if ($vpy) {
                & uv pip install --python $vpy "$root[mcp]"
                if ($LASTEXITCODE -eq 0) { $status = 0 }
            }
        }
    } else {
        & $py @($pyArgs + @("-m", "venv", $venv))
        if ($LASTEXITCODE -eq 0) {
            $vpy = Find-VenvBin $venv "python"
            if ($vpy) {
                & $vpy -m pip install --quiet --upgrade pip
                & $vpy -m pip install --quiet "$root[mcp]"
                if ($LASTEXITCODE -eq 0) { $status = 0 }
            }
        }
    }

    if (($status -eq 0) -and (Find-VenvBin $venv "clara-mcp")) {
        Set-CurrentPointer $data $venv
        if (-not (Update-ClaraShim $data $venv)) {
            Write-Output "shim refresh failed (non-fatal)"
        }
        # GC: keep the two newest venvs.
        $venvs = Get-ChildItem -Path $data -Directory -Filter "venv-*" |
            Sort-Object LastWriteTime -Descending
        if ($venvs.Count -gt 2) {
            $venvs | Select-Object -Skip 2 | ForEach-Object {
                try { Remove-Item $_.FullName -Recurse -Force -Confirm:$false } catch {}
            }
        }
        Write-Output "=== clara install complete: $(Get-Date) ==="
    } else {
        $status = 1
        Write-Output "=== clara install FAILED (see messages above): $(Get-Date) ==="
    }
    try { Remove-Item $flag -Force -Confirm:$false -ErrorAction Stop } catch {}
    try { Remove-Item $lock -Recurse -Force -Confirm:$false -ErrorAction Stop } catch {}
    exit $status
}

# ---------------------------------------------------------------------------
# Foreground path
# ---------------------------------------------------------------------------
try {
    $null = New-Item -ItemType Directory -Force $dataDir -ErrorAction Stop
} catch {
    Write-ClaraLog "cannot create data dir $dataDir"
    exit 1
}

$python = Find-ClaraPython
if (-not $python) {
    Write-ClaraLog "no Python >= 3.10 found (tried py -3.13/-3.12/-3.11/-3.10, python, python3)."
    Write-ClaraLog "install Python 3.10+ (https://www.python.org/downloads/) and start a new session."
    exit 1
}

$pyproject = Join-Path $pluginRoot "pyproject.toml"
$hashScript = "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest()[:12])"
$hash = & $python.Cmd @($python.Args + @("-c", $hashScript, $pyproject)) 2>$null
if (-not $hash) {
    Write-ClaraLog "cannot hash $pyproject"
    exit 1
}
$hash = "$hash".Trim()
$venv = Join-Path $dataDir "venv-$hash"

# Fast path: venv for the current hash is ready.
if (Find-VenvBin $venv "clara-mcp") {
    Set-CurrentPointer $dataDir $venv
    $shimExe = Join-Path $dataDir "shim\clara-mcp.exe"
    $needShim = $true
    if (Test-Path $shimExe) {
        $srcBin = Find-VenvBin $venv "clara-mcp"
        $src = Get-Item $srcBin
        $dst = Get-Item $shimExe
        if (($src.Length -eq $dst.Length) -and ($src.LastWriteTimeUtc -le $dst.LastWriteTimeUtc)) {
            $needShim = $false
        }
    }
    if ($needShim) { $null = Update-ClaraShim $dataDir $venv }
    exit 0
}

# Install needed (first run or pyproject hash change).
$flag = Join-Path $dataDir ".installing"
$lock = Join-Path $dataDir ".lock"
$installLog = Join-Path $dataDir "install.log"

if (Test-Path $flag) {
    $age = ((Get-Date) - (Get-Item $flag).LastWriteTime).TotalSeconds
    if ($age -lt 900) { exit 3 }
    Write-ClaraLog "previous install looks stale (>15 min); retrying."
    try { Remove-Item $flag -Force -Confirm:$false -ErrorAction Stop } catch {}
    try { Remove-Item $lock -Recurse -Force -Confirm:$false -ErrorAction Stop } catch {}
}

try {
    $null = New-Item -ItemType Directory -Path $lock -ErrorAction Stop
} catch {
    exit 3  # another caller is spawning the install right now
}
$null = New-Item -ItemType File -Force $flag
Write-ClaraLog "installing the CLARA memory environment in the background (log: $installLog)"

$env:CLARA_BS_PY = $python.Cmd
$env:CLARA_BS_PYARGS = ($python.Args -join " ")
$env:CLARA_BS_VENV = $venv
$env:CLARA_BS_ROOT = $pluginRoot
$env:CLARA_BS_DATA = $dataDir
$workerErr = "$installLog.err"
Start-Process -FilePath "powershell" -WindowStyle Hidden `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", $MyInvocation.MyCommand.Path, "-InstallWorker") `
    -RedirectStandardOutput $installLog -RedirectStandardError $workerErr
exit 3
