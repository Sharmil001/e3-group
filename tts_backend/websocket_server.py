from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from tts_backend.model import TTSConfig
from tts_backend.streaming import TTSStreamer

log = logging.getLogger("tts_backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

_state: dict = {"streamer": None, "warm": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = TTSConfig(
        model_name=os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        strategy=os.getenv("TTS_STRATEGY", "hf_swap"),
        emit_every_frames=int(os.getenv("TTS_EMIT_EVERY_FRAMES", "1")),
        decode_window_frames=int(os.getenv("TTS_DECODE_WINDOW_FRAMES", "16")),
        voice_clone_audio=os.getenv("TTS_VOICE_CLONE_AUDIO") or None,
        voice_clone_text=os.getenv("TTS_VOICE_CLONE_TEXT") or None,
        verbose=True,
    )
    log.info("Loading TTS engine: %s", cfg)
    _state["streamer"] = TTSStreamer(cfg)
    if os.getenv("TTS_WARM_ON_BOOT", "1") == "1":
        log.info("Warming up...")
        try:
            await _state["streamer"].warm()
            _state["warm"] = True
            log.info("Warm. ttfc=%.1f ms", _state["streamer"].last_stats.ttfc_ms)
        except Exception as e:
            log.exception("Warmup failed: %s", e)
    yield
    log.info("Shutting down")


app = FastAPI(title="Megakernel Qwen3-TTS", lifespan=lifespan)


@app.get("/health")
async def health():
    return JSONResponse({"ok": _state["streamer"] is not None, "warm": _state["warm"]})


@app.post("/warm")
async def warm():
    if _state["streamer"] is None:
        return JSONResponse({"ok": False, "error": "streamer not loaded"}, status_code=503)
    await _state["streamer"].warm()
    _state["warm"] = True
    return JSONResponse({"ok": True, "stats": _state["streamer"].last_stats.as_dict()})


@app.websocket("/tts")
async def tts_ws(ws: WebSocket) -> None:
    await ws.accept()
    streamer: TTSStreamer = _state["streamer"]
    if streamer is None:
        await ws.send_json({"event": "error", "message": "streamer not loaded"})
        await ws.close()
        return

    try:
        while True:
            msg = await ws.receive_text()
            try:
                req = json.loads(msg)
            except json.JSONDecodeError:
                await ws.send_json({"event": "error", "message": "invalid json"})
                continue

            text = req.get("text")
            if not text:
                await ws.send_json({"event": "error", "message": "missing 'text'"})
                continue

            await ws.send_json({"event": "started", "sample_rate": streamer.engine.sample_rate})
            try:
                async for chunk in streamer.synthesize(text):
                    await ws.send_bytes(chunk.pcm.astype(np.float32).tobytes())
            except Exception as e:
                log.exception("synthesize error: %s", e)
                await ws.send_json({"event": "error", "message": str(e)})
                continue

            await ws.send_json({"event": "stopped", "stats": streamer.last_stats.as_dict()})
    except WebSocketDisconnect:
        return


@app.websocket("/tokens")
async def tokens_ws(ws: WebSocket) -> None:
    """Raw megakernel decode: prompt token IDs in, token stream out. Used for throughput benchmarks."""
    await ws.accept()
    streamer: TTSStreamer = _state["streamer"]
    if streamer is None:
        await ws.send_json({"event": "error", "message": "streamer not loaded"})
        await ws.close()
        return

    try:
        while True:
            msg = await ws.receive_text()
            try:
                req = json.loads(msg)
                token_ids = [int(t) for t in req["token_ids"]]
                max_tokens = int(req.get("max_tokens", 128))
            except Exception as e:
                await ws.send_json({"event": "error", "message": f"bad request: {e}"})
                continue

            impl = streamer.engine._impl
            talker = getattr(impl, "talker", None)
            if talker is None:
                await ws.send_json({"event": "error", "message": "talker unavailable"})
                continue

            await ws.send_json({"event": "started", "vocab_size": talker.vocab_size})
            started = asyncio.get_running_loop().time()
            out_tokens = []
            talker.reset()
            if len(token_ids) > 1:
                talker.prefill(token_ids[:-1])
            last = token_ids[-1] if token_ids else 0

            try:
                for _ in range(max_tokens):
                    last, _hidden = talker.step(last)
                    out_tokens.append(last)
                    await ws.send_json({"event": "token", "token": int(last), "position": int(talker.position)})
            except Exception as e:
                log.exception("token decode error: %s", e)
                await ws.send_json({"event": "error", "message": str(e)})
                continue

            elapsed = max(asyncio.get_running_loop().time() - started, 1e-9)
            await ws.send_json({
                "event": "stopped",
                "tokens": out_tokens,
                "tokens_per_sec": len(out_tokens) / elapsed,
            })
    except WebSocketDisconnect:
        return


def main() -> None:
    host = os.getenv("TTS_HOST", "0.0.0.0")
    port = int(os.getenv("TTS_PORT", "8765"))
    uvicorn.run(
        "tts_backend.websocket_server:app",
        host=host,
        port=port,
        log_level=os.getenv("TTS_LOG_LEVEL", "info"),
        access_log=False,
    )


if __name__ == "__main__":
    main()
