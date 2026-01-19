@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  ComfyUI-nunchaku Portable Builder
echo ============================================
echo.

set WORK_DIR=%CD%
set SCRIPT_DIR=%~dp0
set BUILD_DIR=%WORK_DIR%\ComfyUI_nunchaku_portable
set PYTHON_VERSION=3.11.11

REM ============================================
REM  Version Selection Menu
REM ============================================

:select_cuda
echo [CUDA Version]
echo   1. CUDA 12.6 (cu126)
echo   2. CUDA 12.8 (cu128) - Recommended
echo.
set /p CUDA_CHOICE="Select CUDA version (1-2) [default: 2]: "
if "!CUDA_CHOICE!"=="" set CUDA_CHOICE=2
if "!CUDA_CHOICE!"=="1" (
    set CUDA_INDEX=cu126
) else if "!CUDA_CHOICE!"=="2" (
    set CUDA_INDEX=cu128
) else (
    echo Invalid choice. Please try again.
    goto :select_cuda
)
echo   Selected: !CUDA_INDEX!
echo.

set TORCH_VERSION=2.9.1
set NUNCHAKU_TORCH=2.9

:select_nunchaku
echo [Nunchaku Version]
echo   1. v1.0.2 - Stable (Recommended)
echo   2. v1.1.0
echo   3. v1.2.0 - Latest
echo.
set /p NUNCHAKU_CHOICE="Select nunchaku version (1-3) [default: 1]: "
if "!NUNCHAKU_CHOICE!"=="" set NUNCHAKU_CHOICE=1
if "!NUNCHAKU_CHOICE!"=="1" (
    set NUNCHAKU_VERSION=1.0.2
) else if "!NUNCHAKU_CHOICE!"=="2" (
    set NUNCHAKU_VERSION=1.1.0
) else if "!NUNCHAKU_CHOICE!"=="3" (
    set NUNCHAKU_VERSION=1.2.0
) else (
    echo Invalid choice. Please try again.
    goto :select_nunchaku
)
echo   Selected: nunchaku v!NUNCHAKU_VERSION!
echo.

REM ============================================
REM  Confirm Selection
REM ============================================
echo ============================================
echo  Build Configuration
echo ============================================
echo   Python:   !PYTHON_VERSION!
echo   PyTorch:  !TORCH_VERSION!+!CUDA_INDEX!
echo   CUDA:     !CUDA_INDEX!
echo   Nunchaku: v!NUNCHAKU_VERSION!+torch!NUNCHAKU_TORCH!
echo ============================================
echo.
set /p CONFIRM="Proceed with this configuration? (Y/N) [default: Y]: "
if /i "!CONFIRM!"=="N" (
    echo.
    echo Build cancelled.
    goto :end
)
echo.

REM ============================================
REM  Step 1: Create build directory
REM ============================================
echo [1/7] Creating build directory...
if exist "%BUILD_DIR%" (
    echo.
    echo [WARNING] Build directory already exists:
    echo           %BUILD_DIR%
    echo.
    echo This will DELETE all existing files including:
    echo   - python_embeded
    echo   - ComfyUI
    echo   - All installed packages
    echo.
    set /p DELETE_CONFIRM="Delete and continue? (Y/N) [default: N]: "
    if /i not "!DELETE_CONFIRM!"=="Y" (
        echo.
        echo Build cancelled. Existing directory preserved.
        goto :end
    )
    echo Removing existing directory...
    rd /s /q "%BUILD_DIR%"
)
mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

mkdir python_embeded

REM ============================================
REM  Step 2: Download Python Standalone
REM ============================================
echo [2/7] Downloading Python Standalone...
set PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20250106/cpython-3.11.11+20250106-x86_64-pc-windows-msvc-shared-install_only.tar.gz
set PYTHON_ARCHIVE=python_portable.tar.gz

curl.exe -L -o "%PYTHON_ARCHIVE%" "%PYTHON_URL%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download Python standalone
    goto :error
)

REM Extract Python
echo Extracting Python...
tar -xf "%PYTHON_ARCHIVE%" -C python_embeded --strip-components=1
if %errorlevel% neq 0 (
    echo ERROR: Failed to extract Python
    goto :error
)
del "%PYTHON_ARCHIVE%"

REM ============================================
REM  Step 3: Clone ComfyUI and plugin
REM ============================================
echo [3/7] Cloning ComfyUI...
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
if %errorlevel% neq 0 (
    echo ERROR: Failed to clone ComfyUI
    goto :error
)

echo Adding ComfyUI-nunchaku plugin...
git clone --depth 1 https://github.com/nunchaku-ai/ComfyUI-nunchaku.git "ComfyUI\custom_nodes\ComfyUI-nunchaku"
if %errorlevel% neq 0 (
    echo ERROR: Failed to clone ComfyUI-nunchaku
    goto :error
)

REM ============================================
REM  Step 4: Setup pip
REM ============================================
echo [4/7] Setting up pip...
curl.exe -o get-pip.py https://bootstrap.pypa.io/get-pip.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to download get-pip.py
    goto :error
)
python_embeded\python.exe get-pip.py
del get-pip.py

REM ============================================
REM  Step 5: Install dependencies
REM ============================================
echo [5/7] Installing dependencies...

REM Install PyTorch with CUDA support
echo Installing PyTorch !TORCH_VERSION!+!CUDA_INDEX!...
python_embeded\python.exe -m pip install torch==!TORCH_VERSION!+!CUDA_INDEX! torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/!CUDA_INDEX!

REM Install ComfyUI dependencies
echo Installing ComfyUI dependencies...
python_embeded\python.exe -m pip install -r ComfyUI\requirements.txt

REM Install ComfyUI-nunchaku dependencies
echo Installing ComfyUI-nunchaku dependencies...
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-nunchaku\requirements.txt

REM ============================================
REM  Step 6: Install nunchaku
REM ============================================
echo [6/7] Installing nunchaku...
set NUNCHAKU_WHEEL=https://github.com/nunchaku-ai/nunchaku/releases/download/v!NUNCHAKU_VERSION!/nunchaku-!NUNCHAKU_VERSION!+torch!NUNCHAKU_TORCH!-cp311-cp311-win_amd64.whl

python_embeded\python.exe -m pip install "!NUNCHAKU_WHEEL!"
if %errorlevel% neq 0 (
    echo WARNING: Failed to install nunchaku from wheel
    echo You may need to install it manually.
)

REM ============================================
REM  Step 7: Create batch files
REM ============================================
echo [7/7] Creating batch files...

REM Copy files from templates
echo Copying files from templates...
copy "%SCRIPT_DIR%templates\run_nvidia_gpu.bat" "%BUILD_DIR%\" >nul

REM Create VERSION_INFO.txt
(
echo ComfyUI-nunchaku Portable Build Info
echo =====================================
echo Build Date: %DATE% %TIME%
echo.
echo Python:   !PYTHON_VERSION!
echo PyTorch:  !TORCH_VERSION!+!CUDA_INDEX!
echo CUDA:     !CUDA_INDEX!
echo Nunchaku: v!NUNCHAKU_VERSION!+torch!NUNCHAKU_TORCH!
) > VERSION_INFO.txt

echo.
echo ============================================
echo  Build completed successfully!
echo ============================================
echo.
echo Output: %BUILD_DIR%
echo.
echo To test: run_nvidia_gpu.bat
echo.

cd /d "%WORK_DIR%"
goto :end

:error
echo.
echo Build failed!
cd /d "%WORK_DIR%"
exit /b 1

:end
pause
