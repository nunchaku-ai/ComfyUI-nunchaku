@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  ComfyUI-nunchaku Portable Builder (CI)
echo ============================================
echo.

set WORK_DIR=%CD%
set SCRIPT_DIR=%~dp0
set BUILD_DIR=%WORK_DIR%\ComfyUI_nunchaku_portable

REM Use environment variable or default
if not defined NUNCHAKU_VERSION set NUNCHAKU_VERSION=1.2.0

echo Build Configuration:
echo   Nunchaku: v!NUNCHAKU_VERSION!
echo.

REM ============================================
REM  Step 1: Create build directory
REM ============================================
echo [1/4] Creating build directory...
if exist "%BUILD_DIR%" (
    echo Removing existing directory...
    rd /s /q "%BUILD_DIR%"
)
mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

REM ============================================
REM  Step 2: Clone ComfyUI
REM ============================================
echo [2/4] Cloning ComfyUI...
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
if %errorlevel% neq 0 (
    echo ERROR: Failed to clone ComfyUI
    goto :error
)

REM ============================================
REM  Step 3: Clone ComfyUI-nunchaku plugin
REM ============================================
echo [3/4] Adding ComfyUI-nunchaku plugin...
git clone --depth 1 https://github.com/nunchaku-ai/ComfyUI-nunchaku.git "ComfyUI\custom_nodes\ComfyUI-nunchaku"
if %errorlevel% neq 0 (
    echo ERROR: Failed to clone ComfyUI-nunchaku
    goto :error
)

REM ============================================
REM  Step 4: Copy installer scripts
REM ============================================
echo [4/4] Creating installer scripts...

copy "%SCRIPT_DIR%templates\install.bat" "%BUILD_DIR%\" >nul
copy "%SCRIPT_DIR%templates\run.bat" "%BUILD_DIR%\" >nul

REM Create config file with version info
(
echo NUNCHAKU_VERSION=!NUNCHAKU_VERSION!
echo PYTHON_VERSION=3.11.11
echo TORCH_VERSION=2.9.1
echo CUDA_VERSION=cu128
) > config.txt

REM Create README
(
echo ComfyUI-nunchaku Minimal Portable Package
echo ==========================================
echo.
echo This is a minimal portable package that requires initial setup.
echo.
echo FIRST TIME SETUP:
echo   1. Run install.bat to install all dependencies
echo      - This will download Python, PyTorch, and nunchaku
echo      - Internet connection required
echo      - Takes about 10-15 minutes
echo.
echo   2. After installation completes, run run.bat to start ComfyUI
echo.
echo SUBSEQUENT USAGE:
echo   - Simply run run.bat to start ComfyUI
echo.
echo Configuration:
echo   - Nunchaku: v!NUNCHAKU_VERSION!
echo   - Python: 3.11.11
echo   - PyTorch: 2.9.1+cu128
echo   - Build Date: %DATE% %TIME%
echo.
echo For more information, visit:
echo   https://github.com/nunchaku-ai/ComfyUI-nunchaku
) > README.txt

echo.
echo ============================================
echo  Build completed successfully!
echo ============================================
echo.
echo Output: %BUILD_DIR%
echo.
echo Package contents:
echo   - ComfyUI with nunchaku plugin
echo   - install.bat: First-time setup script
echo   - run.bat: Launch ComfyUI
echo   - README.txt: Usage instructions
echo.

cd /d "%WORK_DIR%"
exit /b 0

:error
echo.
echo Build failed!
cd /d "%WORK_DIR%"
exit /b 1
