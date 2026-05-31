FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy installer and requirements
COPY install.py requirements-base.txt requirements-inference.txt ./
COPY trainers/ trainers/

# Install with known CUDA version, CLI-only, non-interactive
RUN python install.py --cuda 12.1 --cli-only --yes

# Copy the msseg package
COPY msseg/ msseg/

# Models should be mounted or included at build time
# To include: COPY msseg/cnn3d/ msseg/resencl3d/ msseg/conv25d/ into msseg/
# To mount: docker run -v /path/to/models:/app/msseg ...

ENTRYPOINT ["python", "-m", "msseg.cli", "--backend", "onnx"]

# Example:
# docker build -t msseg .
# docker run --gpus all -v /data:/data msseg \
#   --flair /data/flair.nii.gz --t1 /data/t1.nii.gz -o /data/seg.nii.gz
