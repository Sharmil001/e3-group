"""WebSocket-based voice demo server.

Avoids WebRTC entirely — audio flows browser → WebSocket (HTTPS) → server,
which works through the Cloudflare tunnel without any UDP/ICE requirement.

Flow per turn:
  Browser (Web Audio API raw PCM → WAV built client-side)
    → WebSocket /ws
    → Deepgram REST transcription (WAV, no ffmpeg)
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

// Separate AudioContext for playback (24 kHz TTS output)
let playCtx = null;
let playQueue = Promise.resolve();
let playSampleRate = 24000;

ws.onmessage = async (e) => {
  if (typeof e.data === 'string') {
    const msg = JSON.parse(e.data);
    if (msg.type === 'transcript') { addMsg(msg.text, 'user'); status.textContent = 'Thinking…'; }
    if (msg.type === 'llm_response') { addMsg(msg.text, 'bot'); status.textContent = 'Speaking…'; }
    if (msg.type === 'done') { btn.disabled = false; status.textContent = 'Ready'; }
    if (msg.type === 'error') { addMsg('Error: ' + msg.text, 'system'); btn.disabled = false; status.textContent = 'Ready'; }
    if (msg.type === 'audio_start') {
      playSampleRate = msg.sample_rate || 24000;
      if (!playCtx || playCtx.sampleRate !== playSampleRate) {
        if (playCtx) playCtx.close();
        playCtx = new AudioContext({ sampleRate: playSampleRate });
      }
    }
    return;
  }
  // binary: PCM float32 chunk from TTS
  if (!playCtx) playCtx = new AudioContext({ sampleRate: playSampleRate });
  const pcm = new Float32Array(e.data);
  playQueue = playQueue.then(() => new Promise(resolve => {
    const buf = playCtx.createBuffer(1, pcm.length, playSampleRate);
    buf.getChannelData(0).set(pcm);
    const src = playCtx.createBufferSource();
    src.buffer = buf;
    src.connect(playCtx.destination);
    src.onended = resolve;
    src.start();
  }));
};

// --- Recording state ---
let recCtx = null;
let recSource = null;
let recNode = null;
let recStream = null;
let pcmChunks = [];
let isRecording = false;

function buildWav(float32Samples, sampleRate) {
  // Convert Float32 → Int16
  const int16 = new Int16Array(float32Samples.length);
  for (let i = 0; i < float32Samples.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Samples[i]));
    int16[i] = s < 0 ? s * 32768 : s * 32767;
  }
  const dataBytes = int16.byteLength;
  const buf = new ArrayBuffer(44 + dataBytes);
  const v = new DataView(buf);
  const ws4 = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  ws4(0, 'RIFF'); v.setUint32(4, 36 + dataBytes, true);
  ws4(8, 'WAVE'); ws4(12, 'fmt ');
  v.setUint32(16, 16, true);      // chunk size
  v.setUint16(20, 1, true);       // PCM
  v.setUint16(22, 1, true);       // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); // byte rate
  v.setUint16(32, 2, true);       // block align
  v.setUint16(34, 16, true);      // bits/sample
  ws4(36, 'data'); v.setUint32(40, dataBytes, true);
  new Uint8Array(buf).set(new Uint8Array(int16.buffer), 44);
  return buf;
}

btn.addEventListener('mousedown', startRec);
btn.addEventListener('touchstart', startRec, { passive: true });
btn.addEventListener('mouseup', stopRec);
btn.addEventListener('touchend', stopRec);

async function startRec(e) {
  e.preventDefault();
  if (btn.disabled || isRecording) return;

  recStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true }, video: false });

  // Use the stream's native sample rate so the browser doesn't need to resample
  recCtx = new AudioContext();
  recSource = recCtx.createMediaStreamSource(recStream);

  // ScriptProcessor is deprecated but universally supported; AudioWorklet needs HTTPS origin
  const bufSize = 4096;
  recNode = recCtx.createScriptProcessor(bufSize, 1, 1);
  pcmChunks = [];
  isRecording = true;

  recNode.onaudioprocess = (ev) => {
    if (!isRecording) return;
    const data = ev.inputBuffer.getChannelData(0);
    pcmChunks.push(new Float32Array(data));
  };

  recSource.connect(recNode);
  recNode.connect(recCtx.destination); // must be connected to fire

  btn.textContent = '🔴 Release to send';
  btn.classList.add('recording');
  status.textContent = 'Recording…';
}

async function stopRec() {
  if (!isRecording) return;
  isRecording = false;

  recNode.disconnect();
  recSource.disconnect();
  recStream.getTracks().forEach(t => t.stop());

  const sr = recCtx.sampleRate;
  await recCtx.close();

  btn.textContent = '🎙 Hold to speak';
  btn.classList.remove('recording');
  btn.disabled = true;
  status.textContent = 'Processing…';

  // Concatenate all captured chunks
  const total = pcmChunks.reduce((s, c) => s + c.length, 0);
  const merged = new Float32Array(total);
  let off = 0;
  for (const c of pcmChunks) { merged.set(c, off); off += c.length; }

  if (merged.length < sr * 0.05) {
    // Less than 50 ms — definitely nothing
    addMsg('Recording too short, please hold longer.', 'system');
    btn.disabled = false;
    status.textContent = 'Ready';
    return;
  }

  const wav = buildWav(merged, sr);
  ws.send(JSON.stringify({ type: 'audio_wav', sample_rate: sr }));
  ws.send(wav);
}
</script>
</body>
</html>
"""


async def transcribe(wav_bytes: bytes) -> str:
    """Transcribe WAV bytes (built client-side) with Deepgram REST API."""
    log.info("Sending %d bytes WAV to Deepgram", len(wav_bytes))
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            },
            content=wav_bytes,
        )
        log.info("Deepgram %d: %s", r.status_code, r.text[:300])
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
    pending_meta: dict = {}
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.receive" and msg.get("text"):
                payload = json.loads(msg["text"])
                if payload.get("type") in ("audio_wav", "audio_mime"):
                    pending_meta = payload
                    log.info("Audio meta: %s", payload)
                continue
            data = msg.get("bytes") or b""
            if not data:
                continue

            meta = pending_meta
            pending_meta = {}
            log.info("Received %d bytes audio (meta=%s)", len(data), meta)

            # 1. Transcribe — data is a client-built WAV
            try:
                transcript = await transcribe(data)
            except Exception as e:
                log.exception("STT error")
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
                log.exception("LLM error")
                await ws.send_text(json.dumps({"type": "error", "text": f"LLM failed: {e}"}))
                continue

            history.append({"role": "assistant", "content": reply})
            await ws.send_text(json.dumps({"type": "llm_response", "text": reply}))

            # 3. TTS → stream PCM back
            try:
                await stream_tts(reply, ws)
            except Exception as e:
                log.exception("TTS error")
                await ws.send_text(json.dumps({"type": "error", "text": f"TTS failed: {e}"}))

            await ws.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    port = int(os.getenv("SERVER_PORT", "7861"))
    log.info("Demo server on http://0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
