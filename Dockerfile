# RTX 5090 (sm_120) requires CUDA >= 12.8 for compilation.
# nvidia/cuda:12.8.0-devel-ubuntu22.04 ships with nvcc 12.8.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    LDG_VOCAB_SIZE=2052

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        build-essential ninja-build git ffmpeg libsndfile1 \
        python3 python3-pip python3-dev ca-certificates curl && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch separately so layer caching is friendlier.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 "torch>=2.6"

COPY requirements.txt /app/requirements.txt
COPY qwen_megakernel/requirements.txt /app/qwen_megakernel/requirements.txt
RUN pip install -r requirements.txt && \
    pip install -r qwen_megakernel/requirements.txt

# Copy source. The kernel is JIT-compiled by torch.utils.cpp_extension on
# first import — bake that compile into the image so first request is hot.
COPY . /app

ENV PYTHONPATH=/app/qwen_megakernel:/app

# Pre-build the extension so the resulting image starts fast.
# NOTE: this step fails on CPU-only build hosts. To build on CPU and defer
# the JIT to runtime, comment this RUN.
RUN python -c "from qwen_megakernel.build import get_extension; get_extension()" \
    || echo "[warn] extension pre-build failed; will JIT on first run"

EXPOSE 8765

CMD ["bash", "scripts/run_server.sh"]
