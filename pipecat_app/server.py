from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from aiortc import RTCIceServer
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from pipecat_app.service import MegakernelTTSService

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# openrelay.metered.ca free TURN — needed to traverse Docker/Vast.ai NAT
ICE_SERVERS = [
    {"urls": "stun:stun.l.google.com:19302"},
    {"urls": "turn:openrelay.metered.ca:80", "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": "turn:openrelay.metered.ca:443", "username": "openrelayproject", "credential": "openrelayproject"},
    {"urls": "turn:openrelay.metered.ca:443?transport=tcp", "username": "openrelayproject", "credential": "openrelayproject"},
]

AIORTC_ICE_SERVERS = [
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(urls=["turn:openrelay.metered.ca:80"], username="openrelayproject", credential="openrelayproject"),
    RTCIceServer(urls=["turn:openrelay.metered.ca:443?transport=tcp"], username="openrelayproject", credential="openrelayproject"),
]


async def run_bot(webrtc_connection: SmallWebRTCConnection) -> None:
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True, audio_out_sample_rate=24000),
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    tts = MegakernelTTSService(url=os.getenv("TTS_WS_URL", "ws://localhost:8765/tts"), sample_rate=24000)

    context = LLMContext(messages=[{
        "role": "system",
        "content": "You are a helpful, concise voice assistant. Keep replies to one short sentence.",
    }])
    ctx_aggr = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        ctx_aggr.user(),
        llm,
        tts,
        transport.output(),
        ctx_aggr.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(audio_out_sample_rate=24000))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        await asyncio.sleep(1)
        await task.queue_frames([LLMMessagesAppendFrame(
            messages=[{"role": "user", "content": "Say hello briefly."}],
            run_llm=True,
        )])

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


handler = SmallWebRTCRequestHandler(ice_servers=AIORTC_ICE_SERVERS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await handler.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/client", SmallWebRTCPrebuiltUI)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/client/")


@app.post("/start")
async def start():
    return {"sessionId": str(uuid.uuid4()), "iceConfig": {"iceServers": ICE_SERVERS}}


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    async def _callback(connection: SmallWebRTCConnection):
        background_tasks.add_task(run_bot, connection)

    return await handler.handle_web_request(request=request, webrtc_connection_callback=_callback)


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    await handler.handle_patch_request(request)
    return {"status": "success"}


@app.post("/sessions/{session_id}/{path:path}")
@app.get("/sessions/{session_id}/{path:path}")
@app.patch("/sessions/{session_id}/{path:path}")
async def session_proxy(session_id: str, path: str, request: Request, background_tasks: BackgroundTasks):
    if path.endswith("api/offer"):
        data = await request.json()
        if request.method == "POST":
            return await offer(
                SmallWebRTCRequest(
                    sdp=data["sdp"],
                    type=data["type"],
                    pc_id=data.get("pc_id"),
                    restart_pc=data.get("restart_pc"),
                    request_data=data.get("request_data") or data.get("requestData"),
                ),
                background_tasks,
            )
        if request.method == "PATCH":
            return await ice_candidate(
                SmallWebRTCPatchRequest(
                    pc_id=data["pc_id"],
                    candidates=[IceCandidate(**c) for c in data.get("candidates", [])],
                )
            )
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", "7860"))
    log.info("Starting on port %d — open http://localhost:%d/client/", port, port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
