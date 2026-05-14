from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import Optional

import numpy as np
import websockets

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType, BaseTextAggregator

log = logging.getLogger(__name__)


class _WholeResponseAggregator(BaseTextAggregator):
    """Buffers the full LLM response so run_tts is called once, not per sentence."""

    def __init__(self):
        super().__init__(aggregation_type=AggregationType.SENTENCE)
        self._buf = ""

    @property
    def text(self) -> Aggregation:
        return Aggregation(text=self._buf, type=AggregationType.SENTENCE)

    async def aggregate(self, text: str):
        self._buf += text
        return
        yield  # noqa: unreachable — satisfies AsyncIterator return type

    async def flush(self) -> Aggregation | None:
        if self._buf:
            result, self._buf = self._buf, ""
            return Aggregation(text=result, type=AggregationType.SENTENCE)
        return None

    async def handle_interruption(self):
        self._buf = ""

    async def reset(self):
        self._buf = ""


class MegakernelTTSService(TTSService):
    """Pipecat TTS service backed by the megakernel Qwen3-TTS WebSocket server."""

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        sample_rate: int = 24000,
        chunk_size_samples: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._text_aggregator = _WholeResponseAggregator()
        self._url = url or os.getenv("TTS_WS_URL", "ws://localhost:8765/tts")
        self._chunk_size_samples = chunk_size_samples
        self._lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str = "") -> AsyncGenerator[Frame, None]:
        async with self._lock:
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
                                    log.info("TTS done: %.0f ms", (time.perf_counter() - t_start) * 1000)
                                break
                            continue

                        arr = np.frombuffer(msg, dtype=np.float32)
                        if first_chunk:
                            first_chunk = False
                            log.info("TTFC: %.0f ms", (time.perf_counter() - t_start) * 1000)

                        if self._chunk_size_samples > 0:
                            arr = np.concatenate([leftover, arr])
                            n = (arr.shape[0] // self._chunk_size_samples) * self._chunk_size_samples
                            leftover = arr[n:]
                            for i in range(0, n, self._chunk_size_samples):
                                yield self._make_audio_frame(arr[i : i + self._chunk_size_samples], sample_rate)
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
        pcm_i16 = (np.clip(pcm_f32, -1.0, 1.0) * 32767.0).astype(np.int16)
        return TTSAudioRawFrame(audio=pcm_i16.tobytes(), sample_rate=sr, num_channels=1)


__all__ = ["MegakernelTTSService"]
