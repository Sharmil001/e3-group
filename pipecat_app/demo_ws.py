"""WebSocket-based voice demo server.

Avoids WebRTC entirely — audio flows browser → WebSocket (HTTPS) → server,
which works through the Cloudflare tunnel without any UDP/ICE requirement.

Flow per turn:
  Browser (MediaRecorder WebM/opus)
    → WebSocket /ws
    → Deepgram REST transcription
    → OpenAI gpt-4o-mini
    → TTS backend ws://localhost:8765/tts
    → PCM float32 chunks back over WebSocket
    → Browser (WebAudio playback)

Run:
    python -m pipecat_app.demo_ws
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
TTS_WS_URL = os.environ.get("TTS_WS_URL", "ws://localhost:8765/tts")

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. "
    "Keep replies under two short sentences. Speak naturally."
)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Megakernel TTS Voice Demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 680px; margin: 60px auto; padding: 0 20px; background: #0f0f10; color: #e8e8e8; }
  h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; }
  .sub { color: #888; font-size: .85rem; margin-bottom: 28px; }
  #chat { background: #1a1a1d; border-radius: 12px; padding: 16px; min-height: 200px; max-height: 400px; overflow-y: auto; margin-bottom: 20px; }
  .msg { margin: 8px 0; line-height: 1.5; }
  .msg.user { color: #7eb8f7; }
  .msg.bot  { color: #e8e8e8; }
  .msg.system { color: #666; font-size: .8rem; }
  #btn { display: block; width: 160px; margin: 0 auto; padding: 14px 0; border-radius: 50px; border: none;
         background: #2563eb; color: #fff; font-size: 1rem; font-weight: 600; cursor: pointer; transition: background .2s; }
  #btn:hover { background: #1d4ed8; }
  #btn.recording { background: #dc2626; animation: pulse 1s infinite; }
  #btn:disabled { background: #444; cursor: wait; }
  @keyframes pulse { 0%,100% { opacity:1 } 50% { opacity:.6 } }
  #status { text-align: center; margin-top: 12px; font-size: .82rem; color: #666; }
</style>
</head>
<body>
<h1>Megakernel Qwen3-TTS Voice Agent</h1>
<p class="sub">RTX 5090 · AlpinDale megakernel · 1,223 tok/s · Deepgram STT · GPT-4o-mini</p>
<div id="chat"><div class="msg system">Click the button and speak. Release to send.</div></div>
<button id="btn">🎙 Hold to speak</button>
<div id="status">Ready</div>

<script>
const chat = document.getElementById('chat');
const btn  = document.getElementById('btn');
const status = document.getElementById('status');

function addMsg(text, role) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = (role === 'user' ? '🎤 You: ' : role === 'bot' ? '🤖 Agent: ' : '') + text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${proto}://${location.host}/ws`);
ws.binaryType = 'arraybuffer';

let audioCtx = null;
let playQueue = Promise.resolve();
let sampleRate = 24000;

ws.onmessage = async (e) => {
  if (typeof e.data === 'string') {
    const msg = JSON.parse(e.data);
    if (msg.type === 'transcript') { addMsg(msg.text, 'user'); status.textContent = 'Thinking…'; }
    if (msg.type === 'llm_response') { addMsg(msg.text, 'bot'); status.textContent = 'Speaking…'; }
    if (msg.type === 'done') { btn.disabled = false; status.textContent = 'Ready'; }
    if (msg.type === 'error') { addMsg('Error: ' + msg.text, 'system'); btn.disabled = false; status.textContent = 'Ready'; }
    if (msg.type === 'audio_start') { sampleRate = msg.sample_rate || 24000; }
    return;
  }
  // binary: PCM float32 chunk
  if (!audioCtx) audioCtx = new AudioContext({ sampleRate });
  const pcm = new Float32Array(e.data);
  playQueue = playQueue.then(() => new Promise(resolve => {
    const buf = audioCtx.createBuffer(1, pcm.length, sampleRate);
    buf.getChannelData(0).set(pcm);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    src.onended = resolve;
    src.start();
  }));
};

let recorder = null;
let chunks = [];

btn.addEventListener('mousedown', startRec);
btn.addEventListener('touchstart', startRec, { passive: true });
btn.addEventListener('mouseup', stopRec);
btn.addEventListener('touchend', stopRec);

async function startRec(e) {
  e.preventDefault();
  if (btn.disabled) return;
  if (!audioCtx) audioCtx = new AudioContext({ sampleRate });
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  chunks = [];
  recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
  recorder.ondataavailable = ev => { if (ev.data.size > 0) chunks.push(ev.data); };
  recorder.start(100);
  btn.textContent = '🔴 Release to send';
  btn.classList.add('recording');
  status.textContent = 'Recording…';
}

async function stopRec() {
  if (!recorder || recorder.state === 'inactive') return;
  recorder.stop();
  recorder.stream.getTracks().forEach(t => t.stop());
  btn.textContent = '🎙 Hold to speak';
  btn.classList.remove('recording');
  btn.disabled = true;
  status.textContent = 'Processing…';
  await new Promise(r => recorder.onstop = r);
  const blob = new Blob(chunks, { type: 'audio/webm' });
  ws.send(blob);
}
</script>
</body>
</html>
"""


async def _webm_to_wav(audio_bytes: bytes) -> bytes:
    """Convert WebM/Opus from MediaRecorder to 16 kHz mono PCM WAV via ffmpeg."""
    log.info("ffmpeg input: %d bytes of WebM", len(audio_bytes))
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", "pipe:0",
        "-f", "wav", "-ar", "16000", "-ac", "1", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    wav, stderr = await proc.communicate(audio_bytes)
    log.info("ffmpeg output: %d bytes of WAV (rc=%d)", len(wav), proc.returncode)
    if proc.returncode != 0:
        log.error("ffmpeg stderr: %s", stderr.decode(errors="replace")[-500:])
    return wav


async def transcribe(audio_bytes: bytes) -> str:
    """Convert WebM to WAV then send to Deepgram REST API."""
    wav = await _webm_to_wav(audio_bytes)
    if len(wav) < 100:
        raise ValueError(f"ffmpeg produced no output ({len(wav)} bytes) — audio too short or corrupt")
    log.info("Sending %d bytes WAV to Deepgram", len(wav))
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            },
            content=wav,
        )
        log.info("Deepgram response: %d — %s", r.status_code, r.text[:300])
        r.raise_for_status()
        data = r.json()
        return data["results"]["channels"][0]["alternatives"][0]["transcript"]


async def llm_reply(history: list[dict]) -> str:
    """Get one-turn reply from OpenAI."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",
                "messages": history,
                "max_tokens": 128,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def stream_tts(text: str, ws: WebSocket) -> None:
    """Connect to the local TTS backend and forward PCM chunks to the browser."""
    import websockets

    async with websockets.connect(TTS_WS_URL, max_size=2**24) as tts_ws:
        await tts_ws.send(json.dumps({"text": text}))
        await ws.send_text(json.dumps({"type": "audio_start", "sample_rate": 24000}))
        while True:
            msg = await tts_ws.recv()
            if isinstance(msg, str):
                payload = json.loads(msg)
                if payload.get("event") in ("stopped", "error"):
                    break
            else:
                await ws.send_bytes(msg)


@app.get("/")
async def index():
    return HTMLResponse(HTML)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    try:
        while True:
            data = await ws.receive_bytes()

            # 1. Transcribe
            try:
                transcript = await transcribe(data)
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "text": f"STT failed: {e}"}))
                continue

            if not transcript.strip():
                await ws.send_text(json.dumps({"type": "error", "text": "No speech detected."}))
                continue

            await ws.send_text(json.dumps({"type": "transcript", "text": transcript}))
            history.append({"role": "user", "content": transcript})

            # 2. LLM
            try:
                reply = await llm_reply(history)
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "text": f"LLM failed: {e}"}))
                continue

            history.append({"role": "assistant", "content": reply})
            await ws.send_text(json.dumps({"type": "llm_response", "text": reply}))

            # 3. TTS → stream PCM back
            try:
                await stream_tts(reply, ws)
            except Exception as e:
                await ws.send_text(json.dumps({"type": "error", "text": f"TTS failed: {e}"}))

            await ws.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", "7861"))
    log.info("Demo server on http://0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
