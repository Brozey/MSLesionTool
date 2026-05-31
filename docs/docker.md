# Docker

A GPU inference container is provided for headless/HPC use. The image installs the CLI dependencies and
runs `python -m msseg.cli` as its entrypoint.

## Build

```bash
docker build -t msseg .
```

## Run

```bash
docker run --gpus all \
  -v /path/to/data:/data \
  -v /path/to/models:/app/msseg \
  msseg --flair /data/flair.nii.gz --t1 /data/t1.nii.gz -o /data/seg.nii.gz
```

- Mount your data directory to `/data`.
- Mount a directory containing the downloaded model weights to `/app/msseg` (the container does not bundle weights).
- `--gpus all` enables CUDA; omit it for CPU-only inference.

The default backend in the container is ONNX Runtime (`--backend onnx`); pass `--backend torch` to use PyTorch.
