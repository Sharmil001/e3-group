#!/usr/bin/env bash
# Launch the Pipecat voice-agent pipeline via the pipecat runner.
#
# Transport modes:
#   local  — SmallWebRTC browser UI at /client/ on port 8080
#             (Vast.ai maps 8080 → external port 33868)
#   daily  — Daily.co WebRTC room (requires DAILY_ROOM_URL + DAILY_TOKEN)
#
# Usage:
#   bash scripts/run_pipecat.sh local       # default
#   bash scripts/run_pipecat.sh daily

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
[[ -d .venv ]] && source .venv/bin/activate

export PYTHONPATH="$REPO_ROOT/qwen_megakernel:$REPO_ROOT:${PYTHONPATH:-}"
export TTS_WS_URL="${TTS_WS_URL:-ws://localhost:8765/tts}"

: "${DEEPGRAM_API_KEY:?Set DEEPGRAM_API_KEY before running Pipecat}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running Pipecat}"

TRANSPORT="${PIPECAT_TRANSPORT:-${1:-local}}"

if [[ "$TRANSPORT" == "daily" ]]; then
    : "${DAILY_ROOM_URL:?Set DAILY_ROOM_URL for the daily transport}"
    : "${DAILY_TOKEN:?Set DAILY_TOKEN for the daily transport}"
    exec python -m pipecat_app.pipeline --transport daily
elif [[ "$TRANSPORT" == "local" ]]; then
    # SmallWebRTC on port 8080; Vast.ai maps this to external port 33868.
    # Open http://<vast_host>:33868/client/ in a browser to connect.
    exec python -m pipecat_app.pipeline --transport webrtc --host 0.0.0.0 --port 8080
else
    echo "Unknown transport: $TRANSPORT (use 'local' or 'daily')" >&2
    exit 1
fi
