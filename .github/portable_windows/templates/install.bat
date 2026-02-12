@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  ComfyUI-nunchaku First-Time Setup
echo ============================================
echo.
echo This will download and install all required dependencies.
echo Internet connection required. This may take 10-15 minutes.
echo.

REM ============================================
REM  Load configuration
REM ============================================
if not exist "config.txt" (
    echo ERROR: config.txt not found
    goto :error
)

for /f "tokens=1,2 delims==" %%a in (config.txt) do (
    set %%a=%%b
)

echo Configuration:
echo   Python: %PYTHON_VERSION%
echo   PyTorch: %TORCH_VERSION%+%CUDA_VERSION%
echo   Nunchaku: v%NUNCHAKU_VERSION%
echo.

REM ============================================
REM  Check if already installed
REM ============================================
if exist ".installed" (
    echo.
    echo Dependencies already installed. Skipping installation.
    echo If you need to reinstall, delete the .installed file and run this again.
    echo.
    goto :end
)

REM ============================================
REM  Step 1: Download Python Standalone
REM ============================================
echo.
echo [1/5] Downloading Python Standalone (~200 MB)...
set PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/20250106/cpython-3.11.11+20250106-x86_64-pc-windows-msvc-shared-install_only.tar.gz
set PYTHON_ARCHIVE=python_portable.tar.gz

curl.exe -L -o "%PYTHON_ARCHIVE%" "%PYTHON_URL%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to download Python standalone
    echo Please check your internet connection and try again.
    goto :error
)

REM Extract Python
echo Extracting Python...
mkdir python_embeded
tar -xf "%PYTHON_ARCHIVE%" -C python_embeded --strip-components=1
if %errorlevel% neq 0 (
    echo ERROR: Failed to extract Python
    goto :error
)
del "%PYTHON_ARCHIVE%"

REM ============================================
REM  Step 2: Setup pip
REM ============================================
echo.
echo [2/5] Setting up pip...
curl.exe -o get-pip.py https://bootstrap.pypa.io/get-pip.py
if %errorlevel% neq 0 (
    echo ERROR: Failed to download get-pip.py
    goto :error
)
python_embeded\python.exe get-pip.py
del get-pip.py

REM ============================================
REM  Step 3: Install PyTorch (~1.5 GB)
REM ============================================
echo.
echo [3/5] Installing PyTorch (~1.5 GB, this may take a while)...
python_embeded\python.exe -m pip install torch==%TORCH_VERSION%+%CUDA_VERSION% torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/%CUDA_VERSION%
if %errorlevel% neq 0 (
    echo ERROR: Failed to install PyTorch
    goto :error
)

REM ============================================
REM  Step 4: Install ComfyUI dependencies
REM ============================================
echo.
echo [4/5] Installing ComfyUI dependencies...
if exist "ComfyUI\requirements.txt" (
    python_embeded\python.exe -m pip install -r ComfyUI\requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install ComfyUI dependencies
        goto :error
    )
)

if exist "ComfyUI\custom_nodes\ComfyUI-nunchaku\requirements.txt" (
    python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-nunchaku\requirements.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install ComfyUI-nunchaku dependencies
        goto :error
    )
)

REM ============================================
REM  Step 5: Install nunchaku
REM ============================================
echo.
echo [5/5] Installing nunchaku...
set NUNCHAKU_WHEEL=https://github.com/nunchaku-ai/nunchaku/releases/download/v%NUNCHAKU_VERSION%/nunchaku-%NUNCHAKU_VERSION%+torch2.9-cp311-cp311-win_amd64.whl

python_embeded\python.exe -m pip install "%NUNCHAKU_WHEEL%"
if %errorlevel% neq 0 (
    echo WARNING: Failed to install nunchaku from wheel
    echo You may need to install it manually or check if the version is available.
    echo Continuing anyway...
)

REM ============================================
REM  Create installation marker
REM ============================================
(
echo Installation completed successfully
echo Date: %DATE% %TIME%
echo Python: %PYTHON_VERSION%
echo PyTorch: %TORCH_VERSION%+%CUDA_VERSION%
echo Nunchaku: v%NUNCHAKU_VERSION%
) > .installed

echo.
echo ============================================
echo  Installation completed successfully!
echo ============================================
echo.
echo You can now run run.bat to start ComfyUI.
echo.
goto :end

:error
echo.
echo ============================================
echo  Installation failed!
echo ============================================
echo.
echo Please check:
echo   1. Internet connection is stable
echo   2. You have enough disk space (~3 GB required)
echo   3. Antivirus is not blocking downloads
echo.
echo If the problem persists, please report the issue at:
echo https://github.com/nunchaku-ai/ComfyUI-nunchaku/issues
echo.
exit /b 1

:end
