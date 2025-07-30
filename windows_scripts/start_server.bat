@echo off
chcp 65001 >nul
title FunASR Transcription Server
color 0A

echo ========================================
echo     FunASR Transcription Server
echo ========================================
echo.

REM Get project root directory
cd /d "%~dp0.."

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not installed or not in PATH
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found, creating...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies need to be installed
if not exist "venv\Lib\site-packages\funasr" (
    echo [INFO] First run detected, installing dependencies...
    if exist "requirements.txt" (
        pip install -r requirements.txt
        if errorlevel 1 (
            echo [ERROR] Failed to install dependencies
            pause
            exit /b 1
        )
    ) else (
        echo [WARNING] requirements.txt file not found
    )
)

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "temp" mkdir temp
if not exist "uploads" mkdir uploads
if not exist "models" mkdir models
if not exist "data" mkdir data

REM Set environment variables
set PYTHONPATH=%CD%
set PYTHONIOENCODING=utf-8

echo [INFO] Starting server...
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Start server
python src\main.py

REM Display error if program exits abnormally
if errorlevel 1 (
    echo.
    echo [ERROR] Server exited abnormally, error code: %errorlevel%
    echo Please check log files for details
)

echo.
echo Press any key to exit...
pause >nul