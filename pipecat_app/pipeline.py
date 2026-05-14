"""Pipecat voice-agent pipeline (Pipecat 1.1.0).

Uses the pipecat runner pattern: define a `bot(runner_args)` function and
let `pipecat.runner.run.main` handle server setup, WebRTC signalling, and
transport lifecycle.

Transports supported:
    webrtc  — SmallWebRTC (prebuilt browser UI at /client/)
    daily   — Daily.co WebRTC room (needs DAILY_ROOM_URL + DAILY_TOKEN env vars)

Required env vars:
    DEEPGRAM_API_KEY
    OPENAI_API_KEY
    TTS_WS_URL            (default ws://localhost:8765/tts)

    # daily only:
    DAILY_ROOM_URL
    DAILY_TOKEN

Run (WebRTC on mapped external port 8080 → Vast.ai 33868):
    python -m pipecat_app.pipeline --transport webrtc --host 0.0.0.0 --port 8080
Run (Daily):
    python -m pipecat_app.pipeline --transport daily
"""

from __future__ import annotations

import logging
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_app.service import MegakernelTTSService


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Reply in exactly ONE short sentence. "
    "Never use bullet points or multiple sentences. Be direct and natural."
)


async def bot(runner_args) -> None:  # type: ignore[type-arg]
    """Entry point called by the pipecat runner for each connected client."""
    from pipecat.runner.types import DailyRunnerArguments, SmallWebRTCRunnerArguments

    # ── Transport ─────────────────────────────────────────────────────────────
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        transport = SmallWebRTCTransport(
            webrtc_connection=runner_args.webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
            ),
        )
    elif isinstance(runner_args, DailyRunnerArguments):
        from pipecat.transports.daily.transport import DailyParams, DailyTransport

        transport = DailyTransport(
            runner_args.room_url,
            runner_args.token,
            "Megakernel Voice Agent",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
            ),
        )
    else:
        raise ValueError(f"Unknown runner args type: {type(runner_args)}")

    # ── Services ──────────────────────────────────────────────────────────────
    tts_url = os.getenv("TTS_WS_URL", "ws://localhost:8765/tts")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    tts = MegakernelTTSService(url=tts_url, sample_rate=24000)

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    ctx_aggr = LLMContextAggregatorPair(context)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    # LLMFullResponseAggregator batches the entire LLM reply into a single
    # TextFrame before passing it to TTS — one TTS call per turn, no
    # sentence-by-sentence splitting that would interleave across turns.
    llm_full = LLMFullResponseAggregator()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            ctx_aggr.user(),
            llm,
            llm_full,
            tts,
            transport.output(),
            ctx_aggr.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(audio_out_sample_rate=24000),
    )

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
