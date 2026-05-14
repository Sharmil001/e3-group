# RTX 5090 Megakernel → Qwen3-TTS → Pipecat

Wires AlpinDale's [qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) as the talker decoder for Qwen3-TTS, streaming real-time speech into a Pipecat voice pipeline.

```
mic → STT → LLM → TTS → speaker
              │
              ├─ talker (megakernel, 829 tok/s)
              ├─ code predictor (5L PyTorch)
              └─ Mimi decoder → 24 kHz PCM
```

---

## Performance (measured on RTX 5090, 2026-05-14)

| Metric | Result | Target |
|---|---|---|
| Megakernel throughput | **829 tok/s** (p95: 846) | ~1,000 |
| Talker RTF | **0.014** | < 0.15 ✅ |
| Full pipeline RTF | 1.02–1.04 | < 0.3 ❌ |
| TTFC (full pipeline) | 3.6–6.3 s | < 90 ms ❌ |

The megakernel talker comfortably hits the RTF target — 1 second of audio needs only 12 talker tokens, generated in ~14.5 ms at 829 tok/s. The bottleneck is the **code predictor + Mimi ConvNet decoder running on stock PyTorch without flash-attention**. Those two stages account for the 3–6 s TTFC and RTF > 1. See `bench_results.md` for the full breakdown.

---

## What we changed in the kernel

Three changes vs upstream `qwen_megakernel`:

1. **Configurable vocab size** — `LDG_VOCAB_SIZE` is now a compile-time `-D` macro (default 151936 for base, 3072 for TTS talker). The talker's smaller vocab shaves buffer size and improves cache utilisation.

2. **Hidden-state export** — added `Decoder.step_with_hidden(token) → (next_token, hidden[1024])`. The kernel already computes the post-final-RMSNorm hidden state internally; we just surface it. The code predictor needs this to generate codebooks 1–15.

3. **Talker weight loader** — `load_talker_weights()` handles the `talker.model.*` key prefix in the Qwen3-TTS checkpoint (vs `model.*` in base Qwen3), reads `rope_theta` and `vocab_size` from `config.json`, and correctly unties `lm_head` from `embed_tokens`.

The `.cu` kernel itself is otherwise unchanged.

---

## Architecture decisions

**Why monkey-patch instead of re-implementing the post-stack**

The upstream `qwen_tts.Qwen3TTSModel` already has working code predictor, Mimi decoder, and tokenisation. Re-implementing those would add risk without any performance benefit — they're not the bottleneck. Instead, `tts_backend/model.py` patches just `talker.model.forward` to route through the megakernel, leaving everything else stock.

**Why single-flight synthesis**

The megakernel uses global scratch buffers and is a non-cooperative single-launch kernel. Concurrent calls would corrupt each other. `TTSStreamer` holds an `asyncio.Lock` — one synthesis at a time, extras queue. Fine for a voice agent with one active speaker.

**Why `emit_every_frames=1`**

Emitting every 80 ms chunk (vs the upstream default of 640 ms) cuts TTFC for the streaming path. Higher per-chunk overhead, but latency wins for voice.

---

## Project layout

| Path | Purpose |
|---|---|
| `qwen_megakernel/` | Patched megakernel fork |
| `tts_backend/model.py` | `TTSEngine` — megakernel talker + code predictor + Mimi decoder |
| `tts_backend/streaming.py` | `TTSStreamer` — async wrapper, single-flight lock |
| `tts_backend/websocket_server.py` | FastAPI WS server: `/tts` (text → PCM), `/tokens` (raw decode) |
| `pipecat_app/service.py` | `MegakernelTTSService` — Pipecat `TTSService` subclass |
| `pipecat_app/pipeline.py` | Full pipeline: STT → LLM → TTS → audio out |
| `benchmarks/` | `throughput.py`, `ttfc.py`, `rtf.py` |
| `scripts/` | Setup, server, benchmark, and demo scripts |

---

## Running it

### On a Vast.ai RTX 5090

```bash
git clone <this repo> && cd e3-group
bash scripts/setup.sh       # ~5-10 min: deps + extension build
bash scripts/run_server.sh  # TTS WS server on :8765
```

### Pipecat demo (SmallWebRTC via SSH tunnel)

On your local machine:

```bash
# Terminal 1 — tunnel TTS port from GPU to localhost
ssh -L 8765:localhost:8765 -p <PORT> root@<HOST> -N

# Terminal 2 — run the pipeline locally
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
export TTS_WS_URL=ws://localhost:8765/tts
python -m pipecat_app.pipeline -t webrtc --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860/client/` — speak, listen, done.

### Benchmarks

```bash
bash scripts/benchmark.sh  # writes bench_results.md

# or individually:
python -m benchmarks.throughput --model talker --runs 20
python -m benchmarks.ttfc --mode ws --runs 10
python -m benchmarks.rtf --runs 5
```

---

## Credits

- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)
- [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat)
