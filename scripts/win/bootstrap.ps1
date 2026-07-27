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
    # clara-mcp is required: the plugin's mcpServers entry spawns this exact
    # path. clara is best-effort but matters just as much in practice -- a
    # plugin-only install never puts the CLI on PATH, so `clara doctor`,
    # `clara sync` and the /clara:sync and /clara:docs slash commands (which
    # shell out to it) all failed with "command not found" for exactly the
    # users who installed the recommended way. Mirrors ensure_shim in
    # scripts/bootstrap.sh.
    $bin = Find-VenvBin $Venv "clara-mcp"
    if (-not $bin) { return $false }
    $shimDir = Join-Path $DataDir "shim"
    $null = New-Item -ItemType Directory -Force $shimDir
    try {
        Copy-Item -Path $bin -Destination (Join-Path $shimDir "clara-mcp.exe") -Force
    } catch {
        return $false
    }
    $cli = Find-VenvBin $Venv "clara"
    if ($cli) {
        try { Copy-Item -Path $cli -Destination (Join-Path $shimDir "clara.exe") -Force } catch {}
    }
    return $true
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

# ---------------------------------------------------------------------------
# Portable Python fallback
#
# Last resort when no system Python >= 3.10 exists. Without this, a user with
# no Python hits a dead end ("install Python and start a new session") that a
# non-technical user cannot action — and on managed machines the usual fixes
# are blocked outright (winget installs fail with 1625 "Organization policies
# are preventing installation").
#
# This path deliberately avoids anything requiring elevation: the official
# CPython NuGet package is a plain zip, so it extracts into the plugin's own
# data dir with no installer, no admin, and no PATH change. The download is
# pinned by version AND verified by SHA256 before a single byte is executed;
# a mismatch aborts rather than running unverified code. Set
# CLARA_NO_AUTO_PYTHON=1 to opt out entirely.
# ---------------------------------------------------------------------------

$PORTABLE_PY_VERSION = "3.12.10"
# SHA256 of https://www.nuget.org/api/v2/package/python/3.12.10 — cross-checked
# against NuGet's published SHA512/packageSize for this exact version.
$PORTABLE_PY_SHA256 = "0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8"

function Get-PortablePython([string]$DataDir) {
    $root = Join-Path $DataDir "pytools\python-$PORTABLE_PY_VERSION"
    $exe = Join-Path $root "tools\python.exe"
    if (Test-Path $exe) { return $exe }

    if ($env:CLARA_NO_AUTO_PYTHON) {
        Write-ClaraLog "no Python found and CLARA_NO_AUTO_PYTHON is set; not downloading one."
        return $null
    }

    Write-ClaraLog "no system Python found - fetching a private CPython $PORTABLE_PY_VERSION (no admin required)."
    $tmp = Join-Path $env:TEMP "clara-python-$PORTABLE_PY_VERSION.zip"
    $url = "https://www.nuget.org/api/v2/package/python/$PORTABLE_PY_VERSION"
    try {
        $prev = $ProgressPreference
        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing -ErrorAction Stop
        $ProgressPreference = $prev
    } catch {
        Write-ClaraLog "could not download Python: $($_.Exception.Message)"
        return $null
    }

    $actual = (Get-FileHash $tmp -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $PORTABLE_PY_SHA256) {
        Write-ClaraLog "downloaded Python failed checksum verification - refusing to use it."
        Write-ClaraLog "  expected $PORTABLE_PY_SHA256"
        Write-ClaraLog "  actual   $actual"
        try { Remove-Item $tmp -Force -Confirm:$false -ErrorAction Stop } catch {}
        return $null
    }

    try {
        if (Test-Path $root) { Remove-Item $root -Recurse -Force -Confirm:$false }
        $null = New-Item -ItemType Directory -Force (Split-Path -Parent $root)
        Expand-Archive -Path $tmp -DestinationPath $root -Force -ErrorAction Stop
    } catch {
        Write-ClaraLog "could not unpack Python: $($_.Exception.Message)"
        return $null
    } finally {
        try { Remove-Item $tmp -Force -Confirm:$false -ErrorAction Stop } catch {}
    }

    if (-not (Test-Path $exe)) {
        Write-ClaraLog "unpacked Python is missing python.exe - ignoring it."
        return $null
    }
    # Make sure pip exists; the venv build needs it.
    & $exe -m ensurepip --upgrade 2>$null | Out-Null
    & $exe -c "import sys, venv; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-ClaraLog "unpacked Python is unusable - ignoring it."
        return $null
    }
    Write-ClaraLog "private Python ready at $exe"
    return $exe
}


# Hash the package sources, and reinstall the package when they have moved on.
#
# The venv is keyed on pyproject.toml, which describes *dependencies*. Editing
# clara/*.py leaves that key unchanged, so the fast path used to conclude "venv
# is current" and keep serving a stale copy of the package -- users never
# received code-only updates. Dependencies are expensive to rebuild and change
# rarely; the package is cheap to reinstall and changes constantly, so the two
# are tracked separately. Mirrors hash_sources/ensure_current_sources in
# scripts/bootstrap.sh.
$SOURCE_HASH_SCRIPT = @'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(root.rglob("*.py")):
    digest.update(path.relative_to(root).as_posix().encode())
    digest.update(path.read_bytes())
print(digest.hexdigest()[:12])
'@

function Get-ClaraSourceHash([string]$PythonExe, [string]$Root) {
    try {
        $out = $SOURCE_HASH_SCRIPT | & $PythonExe - (Join-Path $Root "clara") 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return "$out".Trim() }
    } catch {}
    return $null
}

function Update-ClaraSources([string]$DataDir, [string]$Venv) {
    $vpy = Find-VenvBin $Venv "python"
    if (-not $vpy) { return }
    $want = Get-ClaraSourceHash $vpy $pluginRoot
    if (-not $want) { return }
    $marker = Join-Path $DataDir "source.hash"
    $have = $null
    if (Test-Path $marker) {
        try { $have = (Get-Content $marker -Raw -ErrorAction Stop).Trim() } catch {}
    }
    if ($want -eq $have) { return }
    Write-ClaraLog "plugin code changed - refreshing the installed package"
    Push-Location $pluginRoot
    try {
        & $vpy -m pip install --quiet --no-deps . 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            try { Set-Content -Path $marker -Value $want -Encoding Ascii -NoNewline } catch {}
        } else {
            Write-ClaraLog "package refresh failed; continuing with the installed code"
        }
    } finally {
        Pop-Location
    }
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
        # Resolve the concrete interpreter the probe selected BEFORE handing it
        # to uv. `uv venv --python py` would resolve the py launcher's *default*
        # interpreter (possibly < 3.10), ignoring the "-3.12" in $pyArgs and
        # then failing the requires-python>=3.10 install in a loop.
        $uvPython = $py
        if ($pyArgs.Count -gt 0) {
            $resolved = (& $py @($pyArgs + @("-c", "import sys; print(sys.executable)")) 2>$null | Select-Object -First 1)
            if ($LASTEXITCODE -eq 0 -and $resolved) { $uvPython = $resolved.Trim() }
        }
        & uv venv --python $uvPython $venv
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
        # Record what was just installed so the fast path can detect drift.
        $vpyDone = Find-VenvBin $venv "python"
        if ($vpyDone) {
            $installedHash = Get-ClaraSourceHash $vpyDone $env:CLARA_BS_ROOT
            if ($installedHash) {
                try {
                    Set-Content -Path (Join-Path $data "source.hash") `
                        -Value $installedHash -Encoding Ascii -NoNewline
                } catch {}
            }
        }
        # GC: keep the two newest venvs.
        $venvs = Get-ChildItem -Path $data -Directory -Filter "venv-*" |
            Sort-Object LastWriteTime -Descending
        if ($venvs.Count -gt 2) {
            $venvs | Select-Object -Skip 2 | ForEach-Object {
                try { Remove-Item $_.FullName -Recurse -Force -Confirm:$false } catch {}
            }
        }
        # Second sweep, by shape rather than by name. Observed on a real
        # install: six abandoned venvs totalling ~580 MB, each with a
        # pyvenv.cfg whose `command =` line still named the venv-<hash> path it
        # was built at, so something outside CLARA had renamed the directory
        # (no rename exists in this repo). The name-based GC cannot reclaim
        # those. Identify a venv by the file that defines one, and delete only
        # what is provably not in use.
        Get-ChildItem -Path $data -Directory | ForEach-Object {
            if ($_.Name -like "venv-*") { return }      # handled above
            if ($_.Name -in @("current", "shim", "pytools")) { return }
            if ($_.FullName -eq $venv) { return }
            if (-not (Test-Path (Join-Path $_.FullName "pyvenv.cfg"))) { return }
            try { Remove-Item $_.FullName -Recurse -Force -Confirm:$false } catch {}
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

$pyproject = Join-Path $pluginRoot "pyproject.toml"
$hashScript = "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest()[:12])"

# An already-installed venv ships its own interpreter, so a healthy install must
# NOT depend on a system Python still being on PATH. (It previously did — the
# probe ran before the fast path and exited 1, silently disabling memory
# injection on any machine whose PATH lost Python after install, even with the
# venv intact.) Prefer the installed interpreter; probe PATH only when an
# install or upgrade is genuinely required.
$installedVenv = $null
$currentDir = Join-Path $dataDir "current"
if (Test-Path $currentDir) {
    if (Find-VenvBin $currentDir "python") { $installedVenv = $currentDir }
}
if (-not $installedVenv) {
    $pointerFile = Join-Path $dataDir "current.path"
    if (Test-Path $pointerFile) {
        $ptr = (Get-Content $pointerFile -Raw -ErrorAction SilentlyContinue)
        if ($ptr) {
            $ptr = $ptr.Trim()
            if ($ptr -and (Test-Path $ptr) -and (Find-VenvBin $ptr "python")) { $installedVenv = $ptr }
        }
    }
}

$python = $null
$hash = $null
if ($installedVenv) {
    $installedPy = Find-VenvBin $installedVenv "python"
    $candidateHash = & $installedPy "-c" $hashScript $pyproject 2>$null
    if ($candidateHash) {
        $candidateHash = "$candidateHash".Trim()
        if (Find-VenvBin (Join-Path $dataDir "venv-$candidateHash") "clara-mcp") {
            # Installed venv already matches the current pyproject: nothing to
            # build, and no system Python is required to say so.
            $python = @{ Cmd = $installedPy; Args = @() }
            $hash = $candidateHash
        }
    }
}

if (-not $python) {
    $python = Find-ClaraPython
    if (-not $python) {
        if ($installedVenv) {
            # Installed but stale (pyproject changed) and no system Python to
            # rebuild with: keep serving the existing venv rather than
            # disabling memory entirely.
            Write-ClaraLog "no Python >= 3.10 found; continuing with the installed environment (upgrade skipped)."
            Set-CurrentPointer $dataDir $installedVenv
            if (-not (Test-Path (Join-Path $dataDir "shim\clara-mcp.exe"))) {
                $null = Update-ClaraShim $dataDir $installedVenv
            }
            exit 0
        }
        # No system Python: provision a private one rather than dead-ending on
        # an instruction a non-technical user cannot act on.
        $portable = Get-PortablePython $dataDir
        if ($portable) {
            $python = @{ Cmd = $portable; Args = @() }
        } else {
            Write-ClaraLog "no Python >= 3.10 found (tried py -3.13/-3.12/-3.11/-3.10, python, python3)."
            Write-ClaraLog "install Python 3.10+ (https://www.python.org/downloads/) and start a new session."
            exit 1
        }
    }
}

if (-not $hash) {
    $hash = & $python.Cmd @($python.Args + @("-c", $hashScript, $pyproject)) 2>$null
    if (-not $hash) {
        Write-ClaraLog "cannot hash $pyproject"
        exit 1
    }
    $hash = "$hash".Trim()
}
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
    # The venv matches pyproject, but the package inside it may predate the
    # newest clara/*.py — refresh it before declaring the environment ready.
    Update-ClaraSources $dataDir $venv
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
