# Base image with CUDA 11.7
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel packaging build

# Install required Python packages
RUN pip install \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy files
COPY ckpt ./ckpt
COPY utils ./utils
COPY interpolate_video.py ./

CMD [ "bash" ]