:; exec sh "$(dirname "$0")/read-annotate.sh" "$@" #
@echo off
setlocal
rem CLARA PostToolUse(Read) dispatcher.
rem
rem This fires on EVERY Read tool call, and starting PowerShell costs ~500 ms
rem on native Windows (measured 412-1127 ms) -- paid even when the hook has
rem nothing to say, which is the common case. An annotation can only come from
rem a quarantine manifest under the CLARA home, so when no manifest exists at
rem all there is no repository for which this hook could produce output. That
rem check is repo-independent and free in cmd.exe, so do it here and skip the
rem interpreter entirely.
rem
rem stdin is drained before exiting: the host writes the hook JSON into this
rem process, and exiting with it unread can break that write. Fail-open
rem throughout -- an annotation is a nicety, never a reason to disturb a Read.
set "CLARA_BASE=%CLARA_HOME%"
if not defined CLARA_BASE set "CLARA_BASE=%USERPROFILE%\.clara"
if not exist "%CLARA_BASE%\quarantine\*.tsv" (
  more > nul 2>&1
  endlocal
  exit /b 0
)
endlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\read-annotate.ps1" %*
exit /b %ERRORLEVEL%
