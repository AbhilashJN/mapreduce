FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system tools and Python environment
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install cython numpy scipy cupy

# Set work directory
WORKDIR /app

# Keep container alive (or override this with `docker run ... your-script.py`)
CMD ["tail", "-f", "/dev/null"]