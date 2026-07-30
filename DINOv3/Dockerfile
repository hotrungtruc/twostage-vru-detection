# Use the official PyTorch base image with CUDA 12.1 and cuDNN
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the default working directory in the container
WORKDIR /workspace

# Set environment variables to ensure Python logs are printed to the terminal immediately
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install core OS libraries (required for OpenCV to function without errors)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install necessary Python libraries (timm, opencv, matplotlib, etc.)
# Using opencv-python-headless instead of opencv-python to avoid missing UI library errors in Docker
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command when starting the container
CMD ["/bin/bash"]
