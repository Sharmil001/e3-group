"""Pipecat TTS service backed by the megakernel Qwen3-TTS WebSocket server.

Subclasses `pipecat.services.tts_service.TTSService`. The service connects to
the local TTS WebSocket on first use, sends one sentence at a time, and
streams `TTSAudioRawFrame`s as soon as PCM bytes arrive — no full-utterance
buffering.

Default sample rate: 24000 Hz mono.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import AsyncGenerator, Optional

import numpy as np

try:
    import websockets
except ImportError as e:
    raise ImportError("`websockets` is required: pip install websockets") from e

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


log = logging.getLogger(__name__)


class MegakernelTTSService(TTSService):
    """Streams audio from the local megakernel TTS WebSocket server.

    Args:
        url: ws:// URL of the TTS backend (default `ws://localhost:8765/tts`)
        sample_rate: backend sample rate (default 24000)
        connect_timeout_s: max time to wait for the WS handshake
        chunk_size_samples: optional re-chunking to a fixed sample count
            before yielding. Useful to match a transport's preferred frame
            size. Set 0 to disable.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        sample_rate: int = 24000,
        connect_timeout_s: float = 5.0,
        chunk_size_samples: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = url or os.getenv("TTS_WS_URL", "ws://localhost:8765/tts")
        self._connect_timeout_s = connect_timeout_s
        self._chunk_size_samples = chunk_size_samples
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._lock = asyncio.Lock()

    async def _ensure_ws(self) -> websockets.WebSocketClientProtocol:
        if self._ws is not None and not self._ws.closed:
            return self._ws
        log.info("Connecting to TTS backend at %s", self._url)
        self._ws = await asyncio.wait_for(
            websockets.connect(self._url, max_size=2**24),
            timeout=self._connect_timeout_s,
        )
        return self._ws

    async def stop(self, frame=None) -> None:  # type: ignore[override]
        try:
            if self._ws is not None and not self._ws.closed:
                await self._ws.close()
        finally:
            await super().stop(frame) if frame is not None else None

    def can_generate_metrics(self) -> bool:  # type: ignore[override]
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:  # type: ignore[override]
        """Stream a single utterance. Yields TTSStartedFrame -> TTSAudioRawFrame*
        -> TTSStoppedFrame."""

        async with self._lock:  # single-flight per service instance
            ws = await self._ensure_ws()
            t_start = time.perf_counter()

            await ws.send(json.dumps({"text": text}))
            yield TTSStartedFrame()

            first_byte_yielded = False
            sample_rate = self.sample_rate
            leftover = np.empty(0, dtype=np.float32)

            try:
                while True:
                    msg = await ws.recv()
                    if isinstance(msg, str):
                        try:
                            payload = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        event = payload.get("event")
                        if event == "started":
                            sample_rate = int(payload.get("sample_rate", sample_rate))
                            continue
                        if event == "stopped":
                            stats = payload.get("stats", {})
                            log.info(
                                "TTS done: ttfc=%.1fms rtf=%.3f n=%d",
                                stats.get("ttfc_ms", float("nan")),
                                stats.get("rtf", float("nan")),
                                stats.get("n_chunks", -1),
                            )
                            break
                        if event == "error":
                            yield ErrorFrame(payload.get("message", "tts error"))
                            break
                        continue

                    # binary PCM float32
                    arr = np.frombuffer(msg, dtype=np.float32)
                    if self._chunk_size_samples > 0:
                        arr = np.concatenate([leftover, arr])
                        nfull = (arr.shape[0] // self._chunk_size_samples) * self._chunk_size_samples
                        emit = arr[:nfull]
                        leftover = arr[nfull:]
                        for i in range(0, emit.shape[0], self._chunk_size_samples):
                            chunk = emit[i : i + self._chunk_size_samples]
                            yield self._make_audio_frame(chunk, sample_rate)
                            if not first_byte_yielded:
                                first_byte_yielded = True
                                ttfc = (time.perf_counter() - t_start) * 1000.0
                                log.info("TTFC (service-side): %.1f ms", ttfc)
                    else:
                        yield self._make_audio_frame(arr, sample_rate)
                        if not first_byte_yielded:
                            first_byte_yielded = True
                            ttfc = (time.perf_counter() - t_start) * 1000.0
                            log.info("TTFC (service-side): %.1f ms", ttfc)
            finally:
                if self._chunk_size_samples > 0 and leftover.size > 0:
                    yield self._make_audio_frame(leftover, sample_rate)
                yield TTSStoppedFrame()

    @staticmethod
    def _make_audio_frame(pcm_f32: np.ndarray, sr: int) -> TTSAudioRawFrame:
        # Pipecat expects PCM-int16 little-endian bytes.
        clipped = np.clip(pcm_f32, -1.0, 1.0)
        pcm_i16 = (clipped * 32767.0).astype(np.int16)
        return TTSAudioRawFrame(
            audio=pcm_i16.tobytes(),
            sample_rate=sr,
            num_channels=1,
        )


__all__ = ["MegakernelTTSService"]
