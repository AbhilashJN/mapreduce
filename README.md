# mapreduce
# about

A mapreduce implementation using Mars and building on existing research in optimizing MapReduce on GPUs.

# commands

Windows:

wsl -d Ubuntu

Need WSL for running a container for Nvidia container toolkit for using GPUs through a container on the host.
https://learn.microsoft.com/en-us/windows/wsl/setup/environment
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Once installed, check that it is working.
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

If you haven't already, build the image
docker build -t cuda .

Run the container with access to the GPUs.
docker run --gpus all -it --rm -v $(pwd)/app:/app cuda bash

# examples
# version
built using template version 3.