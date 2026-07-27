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
if not exist "%CLARA_BASE%\proposals\" (
  endlocal
  exit /b 0
)
endlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\session-stop.ps1" %*
exit /b %ERRORLEVEL%
