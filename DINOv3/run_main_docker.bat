@echo off
echo =======================================================
echo     INITIALIZING DOCKER ENVIRONMENT FOR DINOv3 MAIN
echo =======================================================

echo.
echo [1/2] Building Docker Image (Name: dinov3_env)...
docker build -t dinov3_env .

echo.
echo [2/2] Running DINOv3 Inference Pipeline...

set GPU_FLAG=
nvidia-smi >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [*] NVIDIA GPU detected. Enabling --gpus all.
    set GPU_FLAG=--gpus all
) else (
    echo [!] No NVIDIA GPU detected. Running on CPU.
)
echo.

docker run --rm %GPU_FLAG% --shm-size=8g -v "%cd%:/workspace" dinov3_env python /workspace/main.py %*
