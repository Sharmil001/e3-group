#!/usr/bin/env bash
# Launch the TTS WebSocket inference server.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
[[ -d .venv ]] && source .venv/bin/activate

export PYTHONPATH="$REPO_ROOT/qwen_megakernel:$REPO_ROOT:${PYTHONPATH:-}"
export LDG_VOCAB_SIZE="${LDG_VOCAB_SIZE:-2052}"

export TTS_HOST="${TTS_HOST:-0.0.0.0}"
export TTS_PORT="${TTS_PORT:-8765}"
export TTS_MODEL="${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}"
export TTS_STRATEGY="${TTS_STRATEGY:-hf_swap}"
export TTS_EMIT_EVERY_FRAMES="${TTS_EMIT_EVERY_FRAMES:-1}"
export TTS_DECODE_WINDOW_FRAMES="${TTS_DECODE_WINDOW_FRAMES:-16}"
export TTS_WARM_ON_BOOT="${TTS_WARM_ON_BOOT:-1}"

echo "[run_server] http://${TTS_HOST}:${TTS_PORT} (ws://${TTS_HOST}:${TTS_PORT}/tts)"
exec python -m tts_backend.websocket_server
