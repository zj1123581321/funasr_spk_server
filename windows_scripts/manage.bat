@echo off
chcp 65001 >nul
title FunASR Transcription Server Management Tool
color 0B

REM Get project root directory
set PROJECT_ROOT=%~dp0..

:menu
cls
echo =========================================
echo    FunASR Transcription Server Manager
echo =========================================
echo.
echo  1. Start Server (Show Window)
echo  2. Start Server (Background)
echo  3. Stop Server
echo  4. View Server Status
echo  5. Setup Auto-start
echo  6. Remove Auto-start
echo  7. View Log Files
echo  8. Clean Temporary Files
echo  9. Exit
echo.
set /p choice="Please select operation (1-9): "

if "%choice%"=="1" goto start_visible
if "%choice%"=="2" goto start_background
if "%choice%"=="3" goto stop_server
if "%choice%"=="4" goto show_status
if "%choice%"=="5" goto setup_autostart
if "%choice%"=="6" goto remove_autostart
if "%choice%"=="7" goto show_logs
if "%choice%"=="8" goto cleanup
if "%choice%"=="9" goto exit
goto invalid_choice

:start_visible
echo.
echo [INFO] Starting server (show window)...
start "FunASR Server" "%~dp0start_server.bat"
echo [SUCCESS] Server start command executed
pause
goto menu

:start_background
echo.
echo [INFO] Starting server (background)...
call "%~dp0start_server_background.bat"
echo [SUCCESS] Server started in background
pause
goto menu

:stop_server
echo.
call "%~dp0stop_server.bat"
goto menu

:show_status
echo.
echo [INFO] Checking server status...
echo.
tasklist /FI "IMAGENAME eq python.exe" /FO TABLE | findstr /i python >nul
if errorlevel 1 (
    echo [STATUS] Server not running
) else (
    echo [STATUS] Python processes detected:
    tasklist /FI "IMAGENAME eq python.exe" /FO TABLE
)
echo.
pause
goto menu

:setup_autostart
echo.
echo [INFO] Setting up auto-start requires administrator privileges...
echo [INFO] Starting PowerShell script...
powershell -ExecutionPolicy Bypass -File "%~dp0setup_autostart.ps1" -Install
pause
goto menu

:remove_autostart
echo.
echo [INFO] Removing auto-start requires administrator privileges...
echo [INFO] Starting PowerShell script...
powershell -ExecutionPolicy Bypass -File "%~dp0setup_autostart.ps1" -Uninstall
pause
goto menu

:show_logs
echo.
echo [INFO] Opening log directory...
if exist "%PROJECT_ROOT%\logs" (
    explorer "%PROJECT_ROOT%\logs"
    echo [SUCCESS] Log directory opened
) else (
    echo [WARNING] Log directory does not exist
)
pause
goto menu

:cleanup
echo.
echo [INFO] Cleaning temporary files...
if exist "%PROJECT_ROOT%\temp" (
    rmdir /s /q "%PROJECT_ROOT%\temp" 2>nul
    mkdir "%PROJECT_ROOT%\temp" 2>nul
    echo [SUCCESS] Temporary files cleaned
) else (
    echo [INFO] Temporary directory does not exist
)

if exist "%PROJECT_ROOT%\uploads" (
    echo [INFO] Upload directory found, clean it?
    set /p clean_uploads="Clean upload files? (y/N): "
    if /i "!clean_uploads!"=="y" (
        rmdir /s /q "%PROJECT_ROOT%\uploads" 2>nul
        mkdir "%PROJECT_ROOT%\uploads" 2>nul
        echo [SUCCESS] Upload files cleaned
    )
)
pause
goto menu

:invalid_choice
echo.
echo [ERROR] Invalid selection, please try again
pause
goto menu

:exit
echo.
echo Thank you for using!
exit /b 0