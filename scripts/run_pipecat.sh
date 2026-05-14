#!/usr/bin/env bash
# Launch the Pipecat voice-agent pipeline.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
[[ -d .venv ]] && source .venv/bin/activate

export PYTHONPATH="$REPO_ROOT/qwen_megakernel:$REPO_ROOT:${PYTHONPATH:-}"
export TTS_WS_URL="${TTS_WS_URL:-ws://localhost:8765/tts}"

# Sanity: required for default services
: "${DEEPGRAM_API_KEY:?Set DEEPGRAM_API_KEY before running Pipecat}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running Pipecat}"

TRANSPORT="${PIPECAT_TRANSPORT:-${1:-daily}}"

if [[ "$TRANSPORT" == "daily" ]]; then
    : "${DAILY_ROOM_URL:?Set DAILY_ROOM_URL for the daily transport}"
    : "${DAILY_TOKEN:?Set DAILY_TOKEN for the daily transport}"
fi

exec python -m pipecat_app.pipeline --transport "$TRANSPORT"
