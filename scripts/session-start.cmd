:; exec sh "$(dirname "$0")/session-start.sh" "$@" #
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\session-start.ps1" %*
exit /b %ERRORLEVEL%
