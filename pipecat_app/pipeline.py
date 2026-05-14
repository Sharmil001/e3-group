"""Pipecat voice-agent pipeline:

    transport.input  -> STT  -> LLM  -> MegakernelTTSService -> transport.output

Default transport: Daily.co WebRTC (works in any browser; just need a room URL
+ token). Fall back to a local PipeCat WebRTC peer if `--transport local` is
passed.

Default services (all swappable via env / CLI):
    STT  : Deepgram (low-latency hosted; alternatives: Whisper, AssemblyAI)
    LLM  : OpenAI gpt-4o-mini (alternatives: Anthropic, local Ollama)
    TTS  : MegakernelTTSService (our backend)

Required env vars:
    DAILY_ROOM_URL         (when --transport daily)
    DAILY_TOKEN            (when --transport daily)
    DEEPGRAM_API_KEY       (for the default STT)
    OPENAI_API_KEY         (for the default LLM)
    TTS_WS_URL             (default ws://localhost:8765/tts)

Run:
    python -m pipecat_app.pipeline                 # daily transport (default)
    python -m pipecat_app.pipeline --transport local
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_app.service import MegakernelTTSService


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. Keep replies under two short"
    " sentences unless explicitly asked for more. Speak naturally."
)


def _build_transport(kind: str) -> Any:
    if kind == "daily":
        from pipecat.transports.services.daily import DailyParams, DailyTransport

        room_url = os.environ["DAILY_ROOM_URL"]
        token = os.environ["DAILY_TOKEN"]
        return DailyTransport(
            room_url,
            token,
            "Megakernel Voice Agent",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=False,
            ),
        )
    if kind == "local":
        # Local SmallWebRTC transport for browser-based testing without Daily.
        from pipecat.transports.network.small_webrtc import (
            SmallWebRTCConnection,
            SmallWebRTCTransport,
            TransportParams,
        )

        return SmallWebRTCTransport(
            connection=SmallWebRTCConnection(host="0.0.0.0", port=7860),
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )
    raise ValueError(f"unknown transport: {kind!r}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=("daily", "local"),
        default=os.getenv("PIPECAT_TRANSPORT", "daily"),
    )
    parser.add_argument(
        "--tts-url", default=os.getenv("TTS_WS_URL", "ws://localhost:8765/tts")
    )
    args = parser.parse_args()

    transport = _build_transport(args.transport)

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    context = OpenAILLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    ctx_aggr = llm.create_context_aggregator(context)

    tts = MegakernelTTSService(url=args.tts_url, sample_rate=24000)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            ctx_aggr.user(),
            llm,
            tts,
            transport.output(),
            ctx_aggr.assistant(),
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_out_sample_rate=24000,
        ),
    )

    @transport.event_handler("on_client_connected")  # type: ignore[misc]
    async def on_connect(transport, client):
        log.info("Client connected: %s", client)
        await task.queue_frames([ctx_aggr.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")  # type: ignore[misc]
    async def on_disconnect(transport, client):
        log.info("Client disconnected")
        await task.queue_frames([EndFrame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
