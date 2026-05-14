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
        self._ws = None
        self._lock = asyncio.Lock()

    def _ws_is_open(self) -> bool:
        if self._ws is None:
            return False
        # websockets ≥ 12 uses ClientConnection with .state; older uses .closed
        try:
            from websockets.connection import State
            return self._ws.state == State.OPEN
        except (ImportError, AttributeError):
            pass
        try:
            return not self._ws.closed
        except AttributeError:
            return False

    async def _ensure_ws(self):
        if self._ws_is_open():
            return self._ws
        log.info("Connecting to TTS backend at %s", self._url)
        self._ws = await asyncio.wait_for(
            websockets.connect(self._url, max_size=2**24),
            timeout=self._connect_timeout_s,
        )
        return self._ws

    async def stop(self, frame=None) -> None:  # type: ignore[override]
        try:
            if self._ws_is_open():
                await self._ws.close()
        finally:
            await super().stop(frame) if frame is not None else None

    def can_generate_metrics(self) -> bool:  # type: ignore[override]
        return True

    async def run_tts(self, text: str, context_id: str = "") -> AsyncGenerator[Frame, None]:  # type: ignore[override]
        """Stream a single utterance — fresh WS connection per request.

        Yields: TTSStartedFrame → TTSAudioRawFrame* → TTSStoppedFrame
        """
        async with self._lock:  # single-flight: megakernel has one global scratch buffer
            t_start = time.perf_counter()
            yield TTSStartedFrame()

            sample_rate = self.sample_rate
            leftover = np.empty(0, dtype=np.float32)
            first_chunk = True

            try:
                async with websockets.connect(self._url, max_size=2**25) as ws:
                    await ws.send(json.dumps({"text": text}))
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
                            elif event in ("stopped", "error"):
                                if event == "error":
                                    yield ErrorFrame(payload.get("message", "tts error"))
                                else:
                                    log.info("TTS done: %.0f ms total", (time.perf_counter() - t_start) * 1000)
                                break
                            continue

                        # Binary frame: PCM float32
                        arr = np.frombuffer(msg, dtype=np.float32)
                        if first_chunk:
                            first_chunk = False
                            log.info("TTFC: %.0f ms", (time.perf_counter() - t_start) * 1000)

                        if self._chunk_size_samples > 0:
                            arr = np.concatenate([leftover, arr])
                            n = (arr.shape[0] // self._chunk_size_samples) * self._chunk_size_samples
                            leftover = arr[n:]
                            for i in range(0, n, self._chunk_size_samples):
                                yield self._make_audio_frame(arr[i:i + self._chunk_size_samples], sample_rate)
                        else:
                            yield self._make_audio_frame(arr, sample_rate)

            except Exception as exc:
                log.error("TTS stream error: %s", exc)
                yield ErrorFrame(str(exc))
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
