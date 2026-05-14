"""WebSocket-based voice demo server — assignment deliverable.

Pipeline (matches Pipecat pipeline.py service chain):
  Browser mic (Web Audio API, raw PCM → WAV)
    → WebSocket /ws
    → Deepgram nova-2 STT (REST)
    → OpenAI gpt-4o-mini LLM
    → Qwen3-TTS megakernel (ws://localhost:8765/tts)
        └─ talker (28L Qwen3-0.6B, CUDA megakernel ~829 tok/s)
        └─ code predictor (5L PyTorch)
        └─ Mimi ConvNet decoder (24 kHz PCM)
    → PCM float32 chunks streamed frame-by-frame
    → Browser WebAudio playback

Metrics reported back to browser:
  • TTFC  — time from text→TTS-server to first PCM chunk (ms)
  • RTF   — total_gen_time / audio_duration
  • E2E   — mic-release to first audio chunk in browser (ms)
  • Chunks — number of PCM frames streamed (proves frame-by-frame, not buffered)

Run:
    python -m pipecat_app.demo_ws
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time

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

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Megakernel TTS Voice Demo</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 20px; background: #0a0a0c; color: #e8e8e8; }
  h1 { font-size: 1.35rem; font-weight: 700; margin-bottom: 2px; }
  .sub { color: #777; font-size: .82rem; margin-bottom: 20px; }

  /* Pipeline stages bar */
  #pipeline { display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }
  .stage { padding: 5px 10px; border-radius: 6px; font-size: .75rem; font-weight: 600;
           background: #1a1a1d; color: #555; border: 1px solid #222; transition: all .2s; }
  .stage.active { background: #1e3a5f; color: #7eb8f7; border-color: #2563eb; }
  .stage.done   { background: #1a3a20; color: #4ade80; border-color: #166534; }
  .stage.streaming { background: #3b1f00; color: #fb923c; border-color: #92400e;
                     animation: pulse .8s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.65} }

  /* Metrics panel */
  #metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 16px; }
  .metric { background: #111115; border: 1px solid #222; border-radius: 8px; padding: 10px 12px; text-align: center; }
  .metric .label { font-size: .68rem; color: #666; text-transform: uppercase; letter-spacing: .05em; }
  .metric .value { font-size: 1.25rem; font-weight: 700; margin-top: 2px; color: #e8e8e8; font-variant-numeric: tabular-nums; }
  .metric .value.good { color: #4ade80; }
  .metric .value.warn { color: #fb923c; }
  .metric .value.bad  { color: #f87171; }

  /* Chat */
  #chat { background: #111115; border: 1px solid #1e1e22; border-radius: 10px; padding: 14px;
          min-height: 180px; max-height: 360px; overflow-y: auto; margin-bottom: 16px; }
  .msg { margin: 7px 0; line-height: 1.5; font-size: .9rem; }
  .msg.user   { color: #7eb8f7; }
  .msg.bot    { color: #e8e8e8; }
  .msg.system { color: #555; font-size: .78rem; }

  /* Controls */
  #controls { display: flex; flex-direction: column; align-items: center; gap: 10px; }
  #btn { width: 180px; padding: 14px 0; border-radius: 50px; border: none;
         background: #2563eb; color: #fff; font-size: 1rem; font-weight: 700;
         cursor: pointer; transition: background .2s; }
  #btn:hover { background: #1d4ed8; }
  #btn.recording { background: #dc2626; animation: pulse 1s infinite; }
  #btn:disabled { background: #333; cursor: wait; }
  #status { font-size: .82rem; color: #666; }

  /* Streaming indicator */
  #stream-bar { height: 4px; width: 100%; max-width: 400px; background: #1a1a1d;
                border-radius: 2px; overflow: hidden; }
  #stream-fill { height: 100%; width: 0%; background: #fb923c; transition: width .1s; border-radius: 2px; }
</style>
</head>
<body>
<h1>Megakernel Qwen3-TTS · Voice Agent Demo</h1>
<p class="sub">RTX 5090 · AlpinDale megakernel · Qwen3-0.6B talker ~829 tok/s · Deepgram STT · GPT-4o-mini</p>

<!-- Pipeline stages -->
<div id="pipeline">
  <div class="stage" id="s-rec">🎙 Recording</div>
  <div class="stage" id="s-stt">📝 Deepgram STT</div>
  <div class="stage" id="s-llm">🤖 LLM (gpt-4o-mini)</div>
  <div class="stage" id="s-tts">⚡ Megakernel TTS</div>
  <div class="stage" id="s-play">🔊 Streaming playback</div>
</div>

<!-- Metrics -->
<div id="metrics">
  <div class="metric">
    <div class="label">TTS TTFC</div>
    <div class="value" id="m-ttfc">—</div>
  </div>
  <div class="metric">
    <div class="label">TTS RTF</div>
    <div class="value" id="m-rtf">—</div>
  </div>
  <div class="metric">
    <div class="label">PCM chunks</div>
    <div class="value" id="m-chunks">—</div>
  </div>
  <div class="metric">
    <div class="label">E2E latency</div>
    <div class="value" id="m-e2e">—</div>
  </div>
</div>

<!-- Chat -->
<div id="chat"><div class="msg system">Hold button · speak · release to send</div></div>

<!-- Controls -->
<div id="controls">
  <button id="btn">🎙 Hold to speak</button>
  <div id="status">Ready</div>
  <div id="stream-bar"><div id="stream-fill"></div></div>
</div>

<script>
const chat    = document.getElementById('chat');
const btn     = document.getElementById('btn');
const status  = document.getElementById('status');
const mTtfc   = document.getElementById('m-ttfc');
const mRtf    = document.getElementById('m-rtf');
const mChunks = document.getElementById('m-chunks');
const mE2e    = document.getElementById('m-e2e');
const fill    = document.getElementById('stream-fill');

function stage(id, state) {
  ['s-rec','s-stt','s-llm','s-tts','s-play'].forEach(s => {
    const el = document.getElementById(s);
    el.classList.remove('active','done','streaming');
    if (s === id) el.classList.add(state);
  });
}
function stagesDone(...ids) {
  ids.forEach(id => {
    document.getElementById(id).classList.remove('active','streaming');
    document.getElementById(id).classList.add('done');
  });
}
function stagesReset() {
  ['s-rec','s-stt','s-llm','s-tts','s-play'].forEach(s =>
    document.getElementById(s).classList.remove('active','done','streaming'));
}

function addMsg(text, role) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = (role==='user' ? '🎤 You: ' : role==='bot' ? '🤖 Agent: ' : '') + text;
  chat.appendChild(d);
  chat.scrollTop = chat.scrollHeight;
}

function colorMetric(el, val, goodThresh, warnThresh) {
  el.classList.remove('good','warn','bad');
  if (val <= goodThresh) el.classList.add('good');
  else if (val <= warnThresh) el.classList.add('warn');
  else el.classList.add('bad');
}

const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${proto}://${location.host}/ws`);
ws.binaryType = 'arraybuffer';

let playCtx = null;
let playQueue = Promise.resolve();
let playSR = 24000;
let chunkCount = 0;
let sendTime = 0;   // when audio was sent from browser
let firstChunkReceived = false;

ws.onmessage = async (e) => {
  if (typeof e.data === 'string') {
    const msg = JSON.parse(e.data);
    switch (msg.type) {
      case 'stt_start':
        stage('s-stt','active'); status.textContent = 'Transcribing…'; break;
      case 'transcript':
        addMsg(msg.text, 'user');
        stagesDone('s-rec','s-stt');
        stage('s-llm','active'); status.textContent = 'Thinking…'; break;
      case 'llm_response':
        addMsg(msg.text, 'bot');
        stagesDone('s-llm');
        stage('s-tts','active'); status.textContent = 'Synthesising…'; break;
      case 'audio_start':
        playSR = msg.sample_rate || 24000;
        chunkCount = 0; firstChunkReceived = false;
        fill.style.width = '0%';
        if (!playCtx || playCtx.sampleRate !== playSR) {
          if (playCtx) playCtx.close();
          playCtx = new AudioContext({ sampleRate: playSR });
        }
        break;
      case 'metrics':
        // TTFC and RTF from server
        const ttfc = msg.ttfc_ms;
        const rtf  = msg.rtf;
        mTtfc.textContent = ttfc != null ? ttfc.toFixed(0) + ' ms' : '—';
        mRtf.textContent  = rtf  != null ? rtf.toFixed(3)         : '—';
        if (ttfc != null) colorMetric(mTtfc, ttfc, 90, 500);
        if (rtf  != null) colorMetric(mRtf,  rtf,  0.3, 1.0);
        break;
      case 'done':
        stagesDone('s-tts','s-play');
        btn.disabled = false;
        status.textContent = 'Ready';
        mChunks.textContent = chunkCount;
        fill.style.width = '100%';
        setTimeout(() => { fill.style.width = '0%'; stagesReset(); }, 1200);
        break;
      case 'error':
        addMsg('Error: ' + msg.text, 'system');
        btn.disabled = false; status.textContent = 'Ready';
        stagesReset(); break;
    }
    return;
  }

  // Binary frame: PCM float32
  if (!playCtx) playCtx = new AudioContext({ sampleRate: playSR });
  const pcm = new Float32Array(e.data);
  chunkCount++;
  mChunks.textContent = chunkCount;
  fill.style.width = Math.min(99, chunkCount * 8) + '%';

  if (!firstChunkReceived) {
    firstChunkReceived = true;
    const e2e = Date.now() - sendTime;
    mE2e.textContent = e2e + ' ms';
    colorMetric(mE2e, e2e, 2000, 5000);
    stage('s-play','streaming');
    stagesDone('s-tts');
  }

  playQueue = playQueue.then(() => new Promise(resolve => {
    const buf = playCtx.createBuffer(1, pcm.length, playSR);
    buf.getChannelData(0).set(pcm);
    const src = playCtx.createBufferSource();
    src.buffer = buf; src.connect(playCtx.destination);
    src.onended = resolve; src.start();
  }));
};

// ── Recording (Web Audio API) ──────────────────────────────────────────────
let recCtx=null, recSource=null, recNode=null, recStream=null;
let pcmChunks=[], isRecording=false;

function buildWav(f32, sr) {
  const i16 = new Int16Array(f32.length);
  for (let i=0;i<f32.length;i++){const s=Math.max(-1,Math.min(1,f32[i]));i16[i]=s<0?s*32768:s*32767;}
  const data=i16.byteLength, buf=new ArrayBuffer(44+data), v=new DataView(buf);
  const ws4=(o,s)=>{for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i));};
  ws4(0,'RIFF');v.setUint32(4,36+data,true);ws4(8,'WAVE');ws4(12,'fmt ');
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);
  ws4(36,'data');v.setUint32(40,data,true);
  new Uint8Array(buf).set(new Uint8Array(i16.buffer),44);
  return buf;
}

btn.addEventListener('mousedown', startRec);
btn.addEventListener('touchstart', startRec, {passive:true});
btn.addEventListener('mouseup', stopRec);
btn.addEventListener('touchend', stopRec);

async function startRec(e) {
  e.preventDefault();
  if (btn.disabled || isRecording) return;
  stagesReset();
  mTtfc.textContent='—'; mRtf.textContent='—'; mChunks.textContent='—'; mE2e.textContent='—';
  mTtfc.className='value'; mRtf.className='value'; mE2e.className='value';

  recStream = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true},video:false});
  recCtx = new AudioContext();
  recSource = recCtx.createMediaStreamSource(recStream);
  recNode = recCtx.createScriptProcessor(4096,1,1);
  pcmChunks=[]; isRecording=true;
  recNode.onaudioprocess = ev => { if(isRecording) pcmChunks.push(new Float32Array(ev.inputBuffer.getChannelData(0))); };
  recSource.connect(recNode); recNode.connect(recCtx.destination);

  stage('s-rec','active');
  btn.textContent='🔴 Release to send';
  btn.classList.add('recording');
  status.textContent='Recording…';
}

async function stopRec() {
  if (!isRecording) return;
  isRecording=false;
  recNode.disconnect(); recSource.disconnect();
  recStream.getTracks().forEach(t=>t.stop());
  const sr=recCtx.sampleRate; await recCtx.close();

  btn.textContent='🎙 Hold to speak'; btn.classList.remove('recording');
  btn.disabled=true; status.textContent='Processing…';

  const total=pcmChunks.reduce((s,c)=>s+c.length,0);
  if (total < sr * 0.05) {
    addMsg('Too short — please hold the button while speaking.','system');
    btn.disabled=false; status.textContent='Ready'; stagesReset(); return;
  }
  const merged=new Float32Array(total);
  let off=0; for(const c of pcmChunks){merged.set(c,off);off+=c.length;}

  sendTime = Date.now();
  ws.send(JSON.stringify({type:'audio_wav',sample_rate:sr}));
  ws.send(buildWav(merged,sr));
  stage('s-stt','active'); status.textContent='Transcribing…';
}
</script>
</body>
</html>
"""


async def transcribe(wav_bytes: bytes) -> str:
    """Send client-built WAV to Deepgram."""
    log.info("Deepgram STT: %d bytes WAV", len(wav_bytes))
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "audio/wav",
            },
            content=wav_bytes,
        )
        log.info("Deepgram %d: %s", r.status_code, r.text[:200])
        r.raise_for_status()
        return r.json()["results"]["channels"][0]["alternatives"][0]["transcript"]


async def llm_reply(history: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "gpt-4o-mini", "messages": history, "max_tokens": 128},
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def stream_tts(text: str, ws: WebSocket) -> dict:
    """Stream PCM from megakernel TTS; return measured metrics."""
    import websockets

    metrics: dict = {}
    async with websockets.connect(TTS_WS_URL, max_size=2 ** 25) as tts_ws:
        t0 = time.perf_counter()
        await tts_ws.send(json.dumps({"text": text}))
        await ws.send_text(json.dumps({"type": "audio_start", "sample_rate": 24000}))

        first_chunk_t: float | None = None
        total_samples = 0
        chunk_n = 0

        while True:
            msg = await tts_ws.recv()
            if isinstance(msg, str):
                payload = json.loads(msg)
                if payload.get("event") in ("stopped", "error"):
                    break
            else:
                if first_chunk_t is None:
                    first_chunk_t = time.perf_counter()
                total_samples += len(msg) // 4  # float32
                chunk_n += 1
                await ws.send_bytes(msg)

        gen_end = time.perf_counter()

    ttfc_ms = (first_chunk_t - t0) * 1000 if first_chunk_t else None
    audio_dur = total_samples / 24000
    rtf = (gen_end - t0) / audio_dur if audio_dur > 0 else None

    log.info(
        "TTS done: chunks=%d samples=%d dur=%.2fs TTFC=%.0fms RTF=%.3f",
        chunk_n, total_samples, audio_dur,
        ttfc_ms or -1, rtf or -1,
    )
    return {"ttfc_ms": ttfc_ms, "rtf": rtf, "chunks": chunk_n}


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
                continue
            data = msg.get("bytes") or b""
            if not data:
                continue

            pending_meta = {}
            log.info("Audio received: %d bytes", len(data))

            # 1 — STT
            await ws.send_text(json.dumps({"type": "stt_start"}))
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

            # 2 — LLM
            try:
                reply = await llm_reply(history)
            except Exception as e:
                log.exception("LLM error")
                await ws.send_text(json.dumps({"type": "error", "text": f"LLM failed: {e}"}))
                continue

            history.append({"role": "assistant", "content": reply})
            await ws.send_text(json.dumps({"type": "llm_response", "text": reply}))

            # 3 — TTS (streaming frame-by-frame)
            try:
                metrics = await stream_tts(reply, ws)
                await ws.send_text(json.dumps({"type": "metrics", **metrics}))
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
