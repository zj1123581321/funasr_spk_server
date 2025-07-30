@echo off
chcp 65001 >nul
title Stop FunASR Transcription Server
color 0C

echo ========================================
echo     Stop FunASR Transcription Server
echo ========================================
echo.

REM Find and terminate Python processes containing funasr keyword
echo [INFO] Looking for FunASR server processes...

REM Use wmic to find processes
for /f "tokens=2 delims=," %%i in ('wmic process where "name='python.exe' and commandline like '%%funasr%%'" get processid /format:csv ^| findstr /r "[0-9]"') do (
    echo [INFO] Found process PID: %%i
    taskkill /PID %%i /F
    if errorlevel 1 (
        echo [WARNING] Unable to terminate process %%i
    ) else (
        echo [SUCCESS] Terminated process %%i
    )
)

REM Backup method: terminate all python.exe processes (use with caution)
echo.
set /p choice="If above method failed, terminate all Python processes? (y/N): "
if /i "%choice%"=="y" (
    echo [WARNING] Terminating all Python processes...
    taskkill /IM python.exe /F >nul 2>&1
    if errorlevel 1 (
        echo [INFO] No Python processes found or already terminated
    ) else (
        echo [SUCCESS] Terminated Python processes
    )
)

echo.
echo [SUCCESS] Operation completed
pause