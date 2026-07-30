#!/bin/bash

VERSION="v2"
if [ "$1" == "v1" ]; then
    VERSION="v1"
fi

if [ "$VERSION" == "v1" ]; then
    SCRIPT_PATH="/workspace/training/v1_old/train_v1_old.py"
else
    SCRIPT_PATH="/workspace/training/v2_new/train_v2_new.py"
fi

echo "======================================================="
echo "    INITIALIZING DOCKER ENVIRONMENT FOR DINOv3"
echo "    Selected Training Version: $VERSION"
echo "======================================================="

echo ""
echo "[1/2] Building Docker Image (Name: dinov3_env)..."
docker build -t dinov3_env .

echo ""
echo "[2/2] Launching Container..."
echo "Note: Running on Mac/Linux. To use NVIDIA GPUs on Linux, add '--gpus all' to the 'docker run' command below."
echo ""

docker run --rm --shm-size=8g -v "$(pwd):/workspace" dinov3_env python $SCRIPT_PATH
