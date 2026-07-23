:; exec sh "$(dirname "$0")/read-annotate.sh" "$@" #
@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0win\read-annotate.ps1" %*
exit /b %ERRORLEVEL%
