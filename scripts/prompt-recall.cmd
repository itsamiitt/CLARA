:; exec sh "$(dirname "$0")/prompt-recall.sh" "$@" #
@echo off
setlocal
rem CLARA UserPromptSubmit dispatcher: per-prompt memory recall.
rem
rem Fires on EVERY prompt, so latency discipline is stricter than any other
rem hook. Starting PowerShell costs ~500 ms on native Windows (measured for
rem the Read hook: 412-1127 ms), which is why this wrapper runs the venv's
rem python.exe DIRECTLY when the standard layouts expose it, and only falls
rem back to PowerShell for exotic layouts (pointer file with a junction-less
rem install). stdin (the hook JSON) passes straight through to the module.
rem Fail-open everywhere: no interpreter => drain stdin, exit 0, no output.
rem No parenthesized fallback block: %VAR% inside (...) expands at parse
rem time, which turned the computed data dir into a literal "\plugin"
rem whenever CLAUDE_PLUGIN_DATA was unset. Straight-line goto keeps
rem expansion at execution time.
set "CLARA_DATA=%CLAUDE_PLUGIN_DATA%"
if defined CLARA_DATA goto havedata
set "CLARA_BASE=%CLARA_HOME%"
if not defined CLARA_BASE set "CLARA_BASE=%USERPROFILE%\.clara"
set "CLARA_DATA=%CLARA_BASE%\plugin"
:havedata
if exist "%CLARA_DATA%\current\Scripts\python.exe" (
  "%CLARA_DATA%\current\Scripts\python.exe" -m clara.fastpath.prompt_recall
  endlocal
  exit /b 0
)
if not exist "%CLARA_DATA%\current.path" (
  rem Not installed yet -- recall is a nicety, never a nag.
  more > nul 2>&1
  endlocal
  exit /b 0
)
endlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\prompt-recall.ps1" %*
exit /b 0
