FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install system packages
RUN apt-get update && \
    apt-get install -y python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install Mars and dependencies
RUN pip3 install --upgrade pip && \
    pip3 install cython numpy scipy cupy && \
    pip3 install git+https://github.com/mars-project/mars.git

# Set working directory
WORKDIR /app

# Keep container alive
CMD ["tail", "-f", "/dev/null"]
