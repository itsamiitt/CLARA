:; exec sh "$(dirname "$0")/session-stop.sh" "$@" #
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\session-stop.ps1" %*
exit /b %ERRORLEVEL%
