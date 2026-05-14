from __future__ import annotations

import logging
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import LLMFullResponseAggregator
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.run import main
from pipecat.runner.types import DailyRunnerArguments, SmallWebRTCRunnerArguments
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat_app.service import MegakernelTTSService

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

SYSTEM_PROMPT = (
    "You are a voice assistant. "
    "STRICT RULE: Your entire reply must be ONE single sentence only — never two. "
    "No lists, no bullet points, no follow-up sentences. "
    "End after the first period or exclamation mark. "
    "Keep it under 20 words. Be warm and natural."
)


async def bot(runner_args) -> None:
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        transport = SmallWebRTCTransport(
            webrtc_connection=runner_args.webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
            ),
        )
    elif isinstance(runner_args, DailyRunnerArguments):
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

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    tts = MegakernelTTSService(
        url=os.getenv("TTS_WS_URL", "ws://localhost:8765/tts"),
        sample_rate=24000,
    )

    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])
    ctx_aggr = LLMContextAggregatorPair(context)
    llm_full = LLMFullResponseAggregator()

    pipeline = Pipeline([
        transport.input(),
        stt,
        ctx_aggr.user(),
        llm,
        llm_full,
        tts,
        transport.output(),
        ctx_aggr.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(audio_out_sample_rate=24000))
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


if __name__ == "__main__":
    main()
