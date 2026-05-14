from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np
import soundfile as sf

from tts_backend.model import SAMPLE_RATE, TTSConfig, TTSEngine


@dataclass
class AudioChunk:
    pcm: np.ndarray  # float32 mono
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
    """Async facade around TTSEngine — bridges blocking GPU calls to asyncio."""

    def __init__(self, cfg: Optional[TTSConfig] = None) -> None:
        self.cfg = cfg or TTSConfig()
        self.engine = TTSEngine(self.cfg)
        self._lock = asyncio.Lock()  # single-flight: megakernel has one global scratch buffer

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        async with self._lock:
            stats = StreamingStats(request_started_at=time.perf_counter())
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Optional[AudioChunk]] = asyncio.Queue(maxsize=64)

            def producer() -> None:
                try:
                    for arr in self.engine.stream(text):
                        loop.call_soon_threadsafe(queue.put_nowait, AudioChunk(pcm=arr))
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, _ErrorChunk(e))

            fut = loop.run_in_executor(None, producer)
            try:
                while True:
                    item = await queue.get()
                    if item is None:
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
        async for _ in self.synthesize("Hello."):
            pass


class _ErrorChunk:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


def load_reference_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            raise RuntimeError(f"Reference audio is {sr} Hz, need {target_sr} Hz. Install librosa or pre-resample.")
    return audio.astype(np.float32)


__all__ = ["AudioChunk", "StreamingStats", "TTSStreamer", "load_reference_audio"]
