"""Real-time factor (RTF) = wall_time / audio_duration.

Measures end-to-end TTS RTF for a fixed set of utterances of varying length
(targeting ~1 s, ~5 s, ~10 s of audio). RTF < 1 is faster than real time; we
target < 0.15 (stretch) / < 0.3 (deliverable).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

import numpy as np

from benchmarks._common import print_table, summarize


UTTERANCES = {
    "short_~1s": "Hello, this is a short test sentence.",
    "medium_~5s": (
        "The megakernel runs the talker decoder for Qwen3-TTS entirely "
        "inside a single non-cooperative CUDA kernel launch."
    ),
    "long_~10s": (
        "Real-time text to speech requires careful streaming. Each 80 "
        "millisecond audio frame is the product of one talker step, one "
        "code predictor forward pass, and a chunked convolutional decoder "
        "that produces twenty four kilohertz pulse code modulation samples."
    ),
}


async def _run(args) -> dict[str, list[float]]:
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
    await streamer.warm()

    results: dict[str, list[float]] = {k: [] for k in UTTERANCES}
    for _ in range(args.runs):
        for label, text in UTTERANCES.items():
            t0 = time.perf_counter()
            samples = 0
            async for chunk in streamer.synthesize(text):
                samples += chunk.pcm.shape[0]
            wall = time.perf_counter() - t0
            duration = samples / streamer.engine.sample_rate
            if duration <= 0:
                continue
            results[label].append(wall / duration)
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=10)
    args = p.parse_args()

    results = asyncio.run(_run(args))
    rows = []
    for label, samples in results.items():
        if not samples:
            continue
        # Convert RTF (unitless) to ms so we can reuse the formatter; the units
        # column will read "ms" but the magnitudes are interpretable as RTF*1000.
        ms = [s * 1000.0 for s in samples]
        rows.append(summarize(f"RTF*1000 [{label}]", ms))
    print_table(rows)

    print("\nInterpret values above as RTF * 1000 (lower is better). "
          "Targets: <150 (stretch), <300 (deliverable).")


if __name__ == "__main__":
    main()
