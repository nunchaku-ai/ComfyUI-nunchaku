@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  ComfyUI-nunchaku Launcher
echo ============================================
echo.

REM ============================================
REM  Check if dependencies are installed
REM ============================================
if not exist ".installed" (
    echo ERROR: Dependencies not installed yet.
    echo.
    echo Please run install.bat first to install all required dependencies.
    echo.
    exit /b 1
)

if not exist "python_embeded\python.exe" (
    echo ERROR: Python not found.
    echo.
    echo Please run install.bat to install dependencies.
    echo.
    exit /b 1
)

REM ============================================
REM  Launch ComfyUI
REM ============================================
echo Starting ComfyUI...
echo.
echo If this is your first time running ComfyUI, it will download
echo some additional models which may take a few minutes.
echo.

python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build

REM ============================================
REM  Handle errors
REM ============================================
if %errorlevel% neq 0 (
    echo.
    echo ============================================
    echo  ComfyUI exited with an error
    echo ============================================
    echo.
    echo Common issues:
    echo   - NVIDIA drivers outdated: Update to the latest drivers
    echo   - c10.dll error: Install Visual C++ Redistributable
    echo     https://aka.ms/vc14/vc_redist.x64.exe
    echo   - CUDA error: Ensure you have an NVIDIA GPU with CUDA support
    echo.
    echo For more help, visit:
    echo https://github.com/nunchaku-ai/ComfyUI-nunchaku/issues
    echo.
)
