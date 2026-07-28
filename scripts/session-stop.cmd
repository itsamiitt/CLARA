:; exec sh "$(dirname "$0")/session-stop.sh" "$@" #
@echo off
setlocal
rem CLARA Stop hook dispatcher.
rem
rem Stop fires once per TURN, not once per session, so a twenty-message session
rem starts PowerShell twenty times -- ~537 ms each on native Windows (measured
rem 438-1248 ms), almost always to decide there is nothing to say.
rem
rem The nudge can only come from a proposals sidecar under the CLARA home, so
rem with no proposals directory there is no repository for which this hook could
rem produce output. That check is repo-independent and free in cmd.exe.
rem
rem Unlike the Read hook this one is not given stdin by the host, so there is
rem nothing to drain. Fail-open throughout: a nudge is a nicety, never a reason
rem to disturb the end of a turn.
set "CLARA_BASE=%CLARA_HOME%"
if not defined CLARA_BASE set "CLARA_BASE=%USERPROFILE%\.clara"
rem Stage 1: journal flush. Gated on the journal-dirty flags change-capture
rem leaves (free to test in cmd) and run through the venv's python DIRECTLY,
rem so the common no-work turn never starts PowerShell.
if /i "%CLARA_MEMORY_ENABLED%"=="0" goto nudge
if /i "%CLARA_MEMORY_ENABLED%"=="false" goto nudge
if /i "%CLARA_MEMORY_ENABLED%"=="no" goto nudge
if /i "%CLARA_MEMORY_ENABLED%"=="off" goto nudge
if not exist "%CLARA_BASE%\journal-dirty\*" goto nudge
set "CLARA_DATA=%CLAUDE_PLUGIN_DATA%"
if not defined CLARA_DATA set "CLARA_DATA=%CLARA_BASE%\plugin"
if exist "%CLARA_DATA%\current\Scripts\python.exe" (
  "%CLARA_DATA%\current\Scripts\python.exe" -m clara.fastpath.stop_flush >nul 2>&1
)
:nudge
if not exist "%CLARA_BASE%\proposals\" (
  endlocal
  exit /b 0
)
endlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\session-stop.ps1" %*
exit /b %ERRORLEVEL%
