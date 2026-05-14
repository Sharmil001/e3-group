"""Time-to-first-audio-chunk (TTFC) for the end-to-end TTS pipeline.

Two measurement modes:

    --mode local
        Call `TTSStreamer.synthesize(text)` directly, measure from entry to
        the first AudioChunk yielded. Isolates the TTS engine itself.

    --mode ws
        Connect to a running TTS WebSocket server (`ws://...`), send a text
        request, and time from send → first binary frame. Includes loopback
        network cost (~0.5 ms on localhost). This is the number the Pipecat
        service experiences.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import numpy as np

from benchmarks._common import print_table, summarize


DEFAULT_TEXTS = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Streaming text-to-speech with a custom CUDA megakernel is fast.",
]


async def _bench_local(args) -> list[float]:
    from tts_backend.model import TTSConfig
    from tts_backend.streaming import TTSStreamer

    cfg = TTSConfig(
        model_name=os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        strategy=os.getenv("TTS_STRATEGY", "hf_swap"),
        emit_every_frames=int(os.getenv("TTS_EMIT_EVERY_FRAMES", "1")),
        decode_window_frames=int(os.getenv("TTS_DECODE_WINDOW_FRAMES", "16")),
        verbose=False,
    )
    streamer = TTSStreamer(cfg)
    for _ in range(args.warmup):
        await streamer.warm()

    samples = []
    for _ in range(args.runs):
        for text in DEFAULT_TEXTS:
            t0 = time.perf_counter()
            first = None
            async for chunk in streamer.synthesize(text):
                first = time.perf_counter()
                break
            assert first is not None
            samples.append((first - t0) * 1000.0)
    return samples


async def _bench_ws(args) -> list[float]:
    import websockets

    samples = []
    async with websockets.connect(args.url, max_size=2**24) as ws:
        # warmup
        for _ in range(args.warmup):
            await ws.send(json.dumps({"text": "Hello."}))
            await _drain(ws)

        for _ in range(args.runs):
            for text in DEFAULT_TEXTS:
                t0 = time.perf_counter()
                await ws.send(json.dumps({"text": text}))
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, bytes):
                        t_first = time.perf_counter()
                        samples.append((t_first - t0) * 1000.0)
                        break
                await _drain(ws)
    return samples


async def _drain(ws) -> None:
    """Read until we see the `stopped` event."""
    while True:
        msg = await ws.recv()
        if isinstance(msg, str):
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if payload.get("event") in ("stopped", "error"):
                return


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("local", "ws"), default="ws")
    p.add_argument("--url", default=os.getenv("TTS_WS_URL", "ws://localhost:8765/tts"))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=20)
    args = p.parse_args()

    if args.mode == "local":
        samples = asyncio.run(_bench_local(args))
    else:
        samples = asyncio.run(_bench_ws(args))

    rows = [summarize(f"TTFC ({args.mode})", samples)]
    print_table(rows)


if __name__ == "__main__":
    main()
