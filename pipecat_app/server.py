"""Standalone FastAPI server for the voice-agent demo.

Replaces the pipecat runner with a custom setup so we can inject TURN
server credentials for both the browser and server-side ICE negotiation.
Without TURN, WebRTC fails when the GPU is behind Docker NAT (Vast.ai).

Uses openrelay.metered.ca free public TURN relay.

Run:
    python -m pipecat_app.server
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# ── TURN/STUN configuration ───────────────────────────────────────────────────
# Uses openrelay.metered.ca free public TURN relay to traverse Docker/Vast.ai NAT.
ICE_SERVERS = [
    {"urls": "stun:stun.l.google.com:19302"},
    {"urls": "stun:openrelay.metered.ca:80"},
    {
        "urls": "turn:openrelay.metered.ca:80",
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
    {
        "urls": "turn:openrelay.metered.ca:443",
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
    {
        "urls": "turn:openrelay.metered.ca:443?transport=tcp",
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
]

# aiortc ice_servers format
AIORTC_ICE_SERVERS = [
    {"urls": "stun:stun.l.google.com:19302"},
    {
        "urls": "turn:openrelay.metered.ca:80",
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
    {
        "urls": "turn:openrelay.metered.ca:443?transport=tcp",
        "username": "openrelayproject",
        "credential": "openrelayproject",
    },
]


async def run_bot(webrtc_connection: SmallWebRTCConnection) -> None:
    """Run the full voice-agent pipeline for one WebRTC connection."""
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.transports.base_transport import TransportParams
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

    from pipecat_app.service import MegakernelTTSService

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_sample_rate=24000,
        ),
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )
    tts = MegakernelTTSService(
        url=os.getenv("TTS_WS_URL", "ws://localhost:8765/tts"),
        sample_rate=24000,
    )

    context = LLMContext(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful, concise voice assistant. "
                    "Keep replies under two short sentences unless explicitly asked for more. "
                    "Speak naturally."
                ),
            }
        ]
    )
    ctx_aggr = LLMContextAggregatorPair(context)

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
        params=PipelineParams(audio_out_sample_rate=24000),
    )

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


# ── FastAPI app ───────────────────────────────────────────────────────────────

handler = SmallWebRTCRequestHandler(ice_servers=AIORTC_ICE_SERVERS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await handler.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the prebuilt Pipecat Playground frontend at /client/
app.mount("/client", SmallWebRTCPrebuiltUI)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/client/")


@app.post("/start")
async def start(request: Request):
    """Return a session ID and TURN ICE config to the browser."""
    return {
        "sessionId": str(uuid.uuid4()),
        "iceConfig": {"iceServers": ICE_SERVERS},
    }


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
    """Proxy Pipecat-Cloud-style /sessions/<id>/api/offer calls to /api/offer."""
    if path.endswith("api/offer"):
        data = await request.json()
        if request.method == "POST":
            webrtc_req = SmallWebRTCRequest(
                sdp=data["sdp"],
                type=data["type"],
                pc_id=data.get("pc_id"),
                restart_pc=data.get("restart_pc"),
                request_data=data.get("request_data") or data.get("requestData"),
            )
            return await offer(webrtc_req, background_tasks)
        if request.method == "PATCH":
            patch_req = SmallWebRTCPatchRequest(
                pc_id=data["pc_id"],
                candidates=[IceCandidate(**c) for c in data.get("candidates", [])],
            )
            return await ice_candidate(patch_req)
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", "7860"))
    log.info("Starting voice-agent server on port %d", port)
    log.info("Open http://localhost:%d/client/ in your browser", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
