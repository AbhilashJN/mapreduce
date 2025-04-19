FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system tools, Python, vim, and tmux
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    vim \
    tmux && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install cython numpy scipy cupy

# Set working directory
WORKDIR /app

# Default command to keep container running
CMD ["tail", "-f", "/dev/null"]