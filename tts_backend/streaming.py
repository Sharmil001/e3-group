"""Streaming primitives: frame assembler, voice-clone prefill, audio emitter.

The TTS pipeline produces audio in this cadence:

    talker_step       ~ 1 ms       (megakernel)
    code_predictor    ~ 0.5-2 ms   (5-layer transformer, stock torch)
    1 frame           = 80 ms      audio    (12.5 Hz frame rate)
    decoder chunk     >= 1 frame   (configurable)

We want to emit a `TTSAudioRawFrame` to Pipecat as soon as the first decoder
chunk is ready. The streaming math:

    TTFC budget ≈ talker_step + code_predictor + 1*decoder_chunk_latency
                + first-byte-on-wire time

For TTFC < 60 ms we need decoder_chunk_latency under ~50 ms with
`decode_window_frames=1` (i.e. flush one frame at a time once warmed up).
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Optional

import numpy as np

from tts_backend.model import (
    FRAME_RATE_HZ,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    TTSConfig,
    TTSEngine,
)


@dataclass
class AudioChunk:
    """A single PCM chunk yielded to the consumer."""

    pcm: np.ndarray  # float32, mono, shape [N]
    sample_rate: int = SAMPLE_RATE
    is_final: bool = False


@dataclass
class StreamingStats:
    request_started_at: float = 0.0
    first_chunk_at: float = 0.0
    last_chunk_at: float = 0.0
    total_samples: int = 0
    n_chunks: int = 0

    @property
    def ttfc_ms(self) -> float:
        if self.first_chunk_at == 0.0:
            return float("nan")
        return (self.first_chunk_at - self.request_started_at) * 1000.0

    @property
    def audio_duration_s(self) -> float:
        return self.total_samples / SAMPLE_RATE

    @property
    def wall_clock_s(self) -> float:
        if self.last_chunk_at == 0.0:
            return 0.0
        return self.last_chunk_at - self.request_started_at

    @property
    def rtf(self) -> float:
        d = self.audio_duration_s
        return self.wall_clock_s / d if d > 0 else float("nan")

    def as_dict(self) -> dict:
        return dict(
            ttfc_ms=self.ttfc_ms,
            audio_duration_s=self.audio_duration_s,
            wall_clock_s=self.wall_clock_s,
            rtf=self.rtf,
            n_chunks=self.n_chunks,
            total_samples=self.total_samples,
        )


class TTSStreamer:
    """Async streaming facade around `TTSEngine`.

    The engine is intrinsically synchronous (GPU calls in a single CUDA
    stream). We offload it to a thread executor and surface results as an
    async iterator so the WebSocket server / Pipecat service can stay on
    the asyncio event loop.

    The streamer also implements voice-clone prefill: at construction time
    it walks the reference audio's codebook-0 sequence through the talker
    to warm the KV cache. This makes the first per-text run as fast as
    subsequent runs (no per-request prefill overhead beyond the user's text).
    """

    def __init__(self, cfg: Optional[TTSConfig] = None) -> None:
        self.cfg = cfg or TTSConfig()
        self.engine = TTSEngine(self.cfg)
        self._lock = asyncio.Lock()  # single-flight: kernel is single-stream

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        """Yield PCM chunks for `text`. Single inflight request at a time."""
        async with self._lock:
            stats = StreamingStats(request_started_at=time.perf_counter())
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Optional[AudioChunk]] = asyncio.Queue(maxsize=64)

            def producer() -> None:
                try:
                    for arr in self.engine.stream(text):
                        loop.call_soon_threadsafe(
                            queue.put_nowait, AudioChunk(pcm=arr)
                        )
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # EOS
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, _ErrorChunk(e))

            fut = loop.run_in_executor(None, producer)
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        stats.last_chunk_at = time.perf_counter()
                        break
                    if isinstance(item, _ErrorChunk):
                        raise item.exc
                    if stats.first_chunk_at == 0.0:
                        stats.first_chunk_at = time.perf_counter()
                    stats.n_chunks += 1
                    stats.total_samples += item.pcm.shape[0]
                    yield item
                stats.last_chunk_at = time.perf_counter()
            finally:
                await fut

            self.last_stats = stats

    async def warm(self) -> None:
        """Run a tiny synthesis to JIT-compile + warm caches."""
        async for _ in self.synthesize("Hello."):
            pass


class _ErrorChunk:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


# =============================================================================
# Voice-clone prefill helpers (used by TTSConfig.voice_clone_audio path)
# =============================================================================


def load_reference_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load `path` as float32 mono PCM at `target_sr`."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            raise RuntimeError(
                f"Reference audio is {sr} Hz, need {target_sr} Hz. "
                "Install librosa or pre-resample the file."
            )
    return audio.astype(np.float32)


__all__ = [
    "AudioChunk",
    "StreamingStats",
    "TTSStreamer",
    "load_reference_audio",
]
