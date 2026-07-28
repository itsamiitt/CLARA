:; exec sh "$(dirname "$0")/change-capture.sh" "$@" #
@echo off
setlocal
rem CLARA PostToolUse dispatcher: journal file/repo changes for the code index.
rem
rem Fires on every Edit/Write and every Bash command, so latency discipline
rem matches prompt-recall.cmd: run the venv's python.exe DIRECTLY (starting
rem PowerShell costs ~500 ms), and drop Bash payloads with no
rem git/package-manager keyword before paying for any interpreter (findstr on
rem a spooled copy is free by comparison). Fail-open everywhere: no
rem interpreter => drain stdin, exit 0, no output.
if /i "%CLARA_MEMORY_ENABLED%"=="0" goto drop
if /i "%CLARA_MEMORY_ENABLED%"=="false" goto drop
if /i "%CLARA_MEMORY_ENABLED%"=="no" goto drop
if /i "%CLARA_MEMORY_ENABLED%"=="off" goto drop
set "CLARA_DATA=%CLAUDE_PLUGIN_DATA%"
if not defined CLARA_DATA (
  set "CLARA_BASE=%CLARA_HOME%"
  if not defined CLARA_BASE set "CLARA_BASE=%USERPROFILE%\.clara"
  set "CLARA_DATA=%CLARA_BASE%\plugin"
)
set "SPOOL=%TEMP%\clara-capture-%RANDOM%%RANDOM%.json"
more > "%SPOOL%" 2>nul
findstr /c:"\"Bash\"" "%SPOOL%" >nul 2>&1
if errorlevel 1 goto dispatch
findstr /c:"git" /c:"npm" /c:"yarn" /c:"bun" /c:"pip" /c:"poetry" /c:"cargo" /c:"go get" /c:" uv " "%SPOOL%" >nul 2>&1
if errorlevel 1 goto cleanup
:dispatch
if exist "%CLARA_DATA%\current\Scripts\python.exe" (
  "%CLARA_DATA%\current\Scripts\python.exe" -m clara.fastpath.change_capture < "%SPOOL%" >nul 2>&1
  goto cleanup
)
if not exist "%CLARA_DATA%\current.path" goto cleanup
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\change-capture.ps1" < "%SPOOL%" >nul 2>&1
:cleanup
del "%SPOOL%" >nul 2>&1
endlocal
exit /b 0
:drop
more > nul 2>&1
endlocal
exit /b 0
