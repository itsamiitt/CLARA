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
rem
rem No parenthesized fallback block: %VAR% inside (...) expands at parse
rem time, which turned the computed data dir into a literal "\plugin".
rem Straight-line goto keeps expansion at execution time.
if /i "%CLARA_MEMORY_ENABLED%"=="0" goto drop
if /i "%CLARA_MEMORY_ENABLED%"=="false" goto drop
if /i "%CLARA_MEMORY_ENABLED%"=="no" goto drop
if /i "%CLARA_MEMORY_ENABLED%"=="off" goto drop
set "CLARA_DATA=%CLAUDE_PLUGIN_DATA%"
if defined CLARA_DATA goto havedata
set "CLARA_BASE=%CLARA_HOME%"
if not defined CLARA_BASE set "CLARA_BASE=%USERPROFILE%\.clara"
set "CLARA_DATA=%CLARA_BASE%\plugin"
:havedata
rem Spool name: %RANDOM% alone is time-seeded, so two hooks started in the
rem same second can collide; centisecond time plus two draws makes that
rem vanishingly unlikely.
set "CLARA_T=%TIME::=%"
set "CLARA_T=%CLARA_T:.=%"
set "CLARA_T=%CLARA_T: =0%"
set "SPOOL=%TEMP%\clara-capture-%CLARA_T%%RANDOM%%RANDOM%.json"
more > "%SPOOL%" 2>nul
rem The Bash keyword pre-filter keys on the tool_name FIELD, not a bare
rem Bash substring: an edited file's content may legitimately contain the
rem word Bash and must not push a file event through the keyword gate.
findstr /c:"\"tool_name\":\"Bash\"" /c:"\"tool_name\": \"Bash\"" "%SPOOL%" >nul 2>&1
if errorlevel 1 goto dispatch
findstr /c:"git" /c:"npm" /c:"yarn" /c:"bun" /c:"pip" /c:"poetry" /c:"cargo" /c:"go get" /c:"uv " "%SPOOL%" >nul 2>&1
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
