#!/usr/bin/env bash
# Helper to record an end-to-end demo of the voice agent.
#
# The actual demo is browser-based (Daily room or local SmallWebRTC peer).
# Run this script on the operator's laptop, NOT the GPU host.
#
# Prereqs:
#   - macOS or Linux with ffmpeg installed
#   - The TTS server + Pipecat are already running on the GPU host
#   - You have a Daily room URL or local WebRTC URL open in a browser tab
#
# This script just records the screen + mic for `DURATION_S` seconds.

set -euo pipefail

OUT="${1:-demo.mp4}"
DURATION_S="${DURATION_S:-90}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found. brew install ffmpeg / apt-get install ffmpeg" >&2
    exit 1
fi

case "$(uname -s)" in
    Darwin)
        # macOS: list devices once with `ffmpeg -f avfoundation -list_devices true -i ""`
        # then set VIDEO_DEV and AUDIO_DEV accordingly. Defaults usually work:
        VIDEO_DEV="${VIDEO_DEV:-1}"     # Capture screen 0
        AUDIO_DEV="${AUDIO_DEV:-0}"     # Default mic
        ffmpeg -y -f avfoundation -framerate 30 -i "${VIDEO_DEV}:${AUDIO_DEV}" \
            -t "$DURATION_S" -c:v libx264 -pix_fmt yuv420p -c:a aac "$OUT"
        ;;
    Linux)
        # Linux: x11grab + pulse default source
        DISPLAY_NUM="${DISPLAY:-:0}"
        ffmpeg -y -video_size 1920x1080 -framerate 30 -f x11grab -i "$DISPLAY_NUM" \
            -f pulse -i default -t "$DURATION_S" \
            -c:v libx264 -pix_fmt yuv420p -c:a aac "$OUT"
        ;;
    *)
        echo "Unsupported OS: $(uname -s)" >&2
        exit 2
        ;;
esac

echo "Wrote $OUT"
