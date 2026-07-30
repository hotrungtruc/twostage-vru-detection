#!/bin/bash
echo "======================================================="
echo "    INITIALIZING DOCKER ENVIRONMENT FOR DINOv3 MAIN"
echo "======================================================="

echo ""
echo "[1/2] Building Docker Image (Name: dinov3_env)..."
docker build -t dinov3_env .

echo ""
echo "[2/2] Running DINOv3 Inference Pipeline..."
echo "Note: Running on Mac/Linux. To use NVIDIA GPUs on Linux, add '--gpus all' to the 'docker run' command below."
echo ""

docker run --rm --shm-size=8g -v "$(pwd):/workspace" dinov3_env python /workspace/main.py "$@"
