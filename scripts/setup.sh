#!/usr/bin/env bash
# One-shot environment setup for an RTX 5090 box (Vast.ai, Ubuntu 22.04).
# Idempotent: safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[setup] RTX 5090 Megakernel + Qwen3-TTS + Pipecat"

# Confirm we are on a GPU box.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[setup] ERROR: nvidia-smi not found. Run this on the RTX 5090 host." >&2
    exit 1
fi
nvidia-smi
echo

# System dependencies
if command -v apt-get >/dev/null 2>&1; then
    sudo -n apt-get update -y >/dev/null 2>&1 || apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ninja-build git ffmpeg libsndfile1 python3-pip python3-venv \
        ca-certificates curl
fi

# Python virtualenv
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install -U pip wheel

# torch with CUDA 12.8 — required for sm_120 (Blackwell / RTX 5090)
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
echo "[setup] Installing torch from ${TORCH_INDEX}"
pip install --index-url "${TORCH_INDEX}" "torch>=2.6"

# Project deps
pip install -r requirements.txt
pip install -r qwen_megakernel/requirements.txt

# Pre-build the megakernel extension. The talker has a different vocab so we
# override LDG_VOCAB_SIZE; the upstream default (151936) is also kept as a
# fallback for benching against Qwen3-0.6B base.
export LDG_VOCAB_SIZE="${LDG_VOCAB_SIZE:-2052}"
echo "[setup] Building megakernel extension (LDG_VOCAB_SIZE=${LDG_VOCAB_SIZE})..."
PYTHONPATH="$REPO_ROOT/qwen_megakernel:${PYTHONPATH:-}" \
    python -c "from qwen_megakernel.build import get_extension; get_extension()"

echo
echo "[setup] Sanity check: running upstream bench..."
LDG_VOCAB_SIZE=151936 PYTHONPATH="$REPO_ROOT/qwen_megakernel:${PYTHONPATH:-}" \
    python -m qwen_megakernel.bench || true

echo
echo "[setup] Inspecting Qwen3-TTS state_dict..."
PYTHONPATH="$REPO_ROOT/qwen_megakernel:$REPO_ROOT:${PYTHONPATH:-}" \
    python -m tts_backend.inspect_weights

echo
echo "[setup] Done. Next:"
echo "  bash scripts/run_server.sh         # start the TTS WebSocket server"
echo "  bash scripts/benchmark.sh          # run throughput / TTFC / RTF benches"
