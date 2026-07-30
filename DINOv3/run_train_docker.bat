@echo off
setlocal

set VERSION=v2
if /I "%~1"=="v1" set VERSION=v1

if "%VERSION%"=="v1" (
    set SCRIPT_PATH=/workspace/training/v1_old/train_v1_old.py
) else (
    set SCRIPT_PATH=/workspace/training/v2_new/train_v2_new.py
)

echo =======================================================
echo     INITIALIZING DOCKER ENVIRONMENT FOR DINOv3 (GPU/CPU)
echo     Selected Training Version: %VERSION%
echo =======================================================

echo.
echo [1/2] Building Docker Image (Name: dinov3_env)...
docker build -t dinov3_env .

echo.
echo [2/2] Launching Container...

set GPU_FLAG=
nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [*] NVIDIA GPU detected. Enabling --gpus all.
    set GPU_FLAG=--gpus all
) else (
    echo [!] No NVIDIA GPU detected. Running on CPU.
)
echo.

docker run --rm %GPU_FLAG% --shm-size=8g -v "%cd%:/workspace" dinov3_env python %SCRIPT_PATH%
