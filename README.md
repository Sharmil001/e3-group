# RTX 5090 Megakernel → Qwen3-TTS → Pipecat

Streaming Qwen3-TTS voice synthesis powered by AlpinDale's [qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) — a ~1,200-line CUDA megakernel that decodes Qwen3-0.6B at ~1,000 tok/s on a single RTX 5090 — wired into a Pipecat voice-agent pipeline.

```
mic  ─►  STT  ─►  LLM  ─►  Megakernel-TTS  ─►  speaker
                            │
                            ├─ talker (28L Qwen3-0.6B)   <-- megakernel
                            ├─ code predictor (5L, 15 heads)
                            └─ Mimi-style ConvNet decoder (24 kHz)
```

## Why this is fast

The talker decoder inside `Qwen/Qwen3-TTS-12Hz-0.6B-Base` is **structurally identical** to `Qwen/Qwen3-0.6B` (28 layers, hidden=1024, GQA Q=16/KV=8, head_dim=128, FFN=3072). All `LDG_*` tuning carried over unchanged. At 12.5 Hz frame rate the talker produces ~12.5 frames per second of audio; at ~1 ms/step the talker portion of generating 1 s of audio costs ~12.5 ms (talker RTF ≈ 0.012). The code predictor and ConvNet decoder are run in stock PyTorch with stream-friendly chunking.

## What changed in the kernel / model

Three minimal deltas vs upstream `qwen_megakernel`:

1. **Configurable vocab size** ([qwen_megakernel/csrc/kernel.cu](qwen_megakernel/csrc/kernel.cu), [qwen_megakernel/qwen_megakernel/build.py](qwen_megakernel/qwen_megakernel/build.py)): `LDG_VOCAB_SIZE` is now a `-D` macro, default 151936 (base) or 3072 (talker). Override at build time with `LDG_VOCAB_SIZE=3072`.
2. **Hidden-state export** ([qwen_megakernel/qwen_megakernel/model.py](qwen_megakernel/qwen_megakernel/model.py)): `Decoder.step_with_hidden(token) -> (next_token, hidden_fp32[1024])`. The kernel already writes the post-final-RMSNorm hidden state into the `normalized` scratch tensor; we just clone it. Zero kernel-code changes for this one.
3. **Talker weight loader** (same file): `load_talker_weights()` auto-detects `talker.model.*` vs `model.*` prefixes in the Qwen3-TTS snapshot, reads `rope_theta` and `vocab_size` from `config.json`, and untying `lm_head` from `embed_tokens`. RoPE uses the standard 1D cos/sin tables — MRoPE [24,20,20] collapses to 1D RoPE during pure-audio autoregressive decode because all three position components advance together. See [Risks](#risks-and-honest-caveats) below.

The kernel `.cu` itself is unchanged except for making `LDG_VOCAB_SIZE` a macro. The Code Predictor (5L) and Mimi decoder run on stock PyTorch — they're small, not the bottleneck, and re-implementing them buys little.

## Layout

| Path | What it does |
| --- | --- |
| [qwen_megakernel/](qwen_megakernel/) | Patched fork of AlpinDale's megakernel (vocab macro + TTS weight loader + hidden export) |
| [tts_backend/model.py](tts_backend/model.py) | `TTSEngine` — talker (megakernel) + code predictor + ConvNet decoder |
| [tts_backend/streaming.py](tts_backend/streaming.py) | `TTSStreamer` — async streaming facade with single-flight lock |
| [tts_backend/websocket_server.py](tts_backend/websocket_server.py) | FastAPI WebSocket server: `/tokens` for `prompt token IDs → token stream`, `/tts` for `text → PCM 24kHz chunks` |
| [tts_backend/inspect_weights.py](tts_backend/inspect_weights.py) | Sanity check that the talker weight prefixes + shapes match expectations |
| [pipecat_app/service.py](pipecat_app/service.py) | `MegakernelTTSService(TTSService)` — Pipecat TTS service that streams from the WS server |
| [pipecat_app/pipeline.py](pipecat_app/pipeline.py) | Full Pipeline: `transport.input → STT → LLM → MegakernelTTSService → transport.output` |
| [benchmarks/](benchmarks/) | `throughput.py`, `ttfc.py`, `rtf.py` |
| [scripts/](scripts/) | `setup.sh`, `run_server.sh`, `run_pipecat.sh`, `benchmark.sh`, `record_demo.sh` |
| [Dockerfile](Dockerfile) | CUDA 12.8 + torch cu128 + pre-built extension |

## Run it

### On a fresh Vast.ai RTX 5090 host

```bash
git clone <this repo>
cd e3-group
bash scripts/setup.sh          # ~5-10 min: apt + pip + extension build + sanity bench
bash scripts/run_server.sh     # starts TTS WS server on :8765, warms on boot
```

Then on the operator laptop (or anywhere with a browser):

```bash
export DEEPGRAM_API_KEY=...
export OPENAI_API_KEY=...
# Daily transport (browser-friendly, recommended):
export DAILY_ROOM_URL=https://yourdomain.daily.co/yourroom
export DAILY_TOKEN=...
bash scripts/run_pipecat.sh daily
# OR local SmallWebRTC transport (point a browser at http://gpu_host:7860):
bash scripts/run_pipecat.sh local
```

### Docker (optional)

```bash
docker build -t megakernel-tts .
docker run --gpus all -p 8765:8765 megakernel-tts
```

## Benchmarks

Run all three:

```bash
bash scripts/benchmark.sh                # writes bench_results.md
```

Individually:

```bash
python -m benchmarks.throughput --model talker --runs 20    # pure megakernel
python -m benchmarks.throughput --model base   --runs 20    # sanity vs upstream
python -m benchmarks.ttfc       --mode local   --runs 10    # engine-only TTFC
python -m benchmarks.ttfc       --mode ws      --runs 10    # WS-loopback TTFC
python -m benchmarks.rtf                       --runs  5
```

### Expected ranges (RTX 5090, talker weights)

| Measurement | Target (stretch / deliverable) | Notes |
| --- | --- | --- |
| Megakernel decode tok/s | ~900-1050 | upstream Qwen3-0.6B reports 1036 tok/s |
| Code Predictor / frame | < 2 ms | 5 layers, hidden 1024, stock torch |
| Mimi decoder / 80ms chunk | < 30 ms | streaming window 16 frames, overlap-add |
| **TTFC** | < 60 ms / < 90 ms | `run_tts(text)` → first `TTSAudioRawFrame` |
| **RTF** | < 0.15 / < 0.3 | wall_time / audio_duration |

Actual numbers from a real RTX 5090 run are emitted into `bench_results.md`. Do not submit without that file populated; the scripts are present, but this local macOS workspace cannot produce valid CUDA performance numbers.

### Methodology

- ≥ 3 warmup runs (discarded) + ≥ 20 timed runs per measurement.
- `torch.cuda.synchronize()` brackets every timed region.
- TTFC measured from `synthesize(text)` entry to first `AudioChunk` yielded (local mode) or from WS send to first binary frame (ws mode — adds ~0.5 ms loopback).
- RTF computed per utterance length (~1 s, ~5 s, ~10 s) to surface chunk-decoder amortization effects.

## Architecture decisions

### Why "HF swap" instead of re-implementing the post-stack

The upstream `qwen_tts.Qwen3TTSModel` already ships a clean streaming API (`stream_generate_voice_clone`) that wires together text tokenization, code predictor invocation, and the Mimi-style decoder with overlap-add. Re-implementing the post-stack would add code volume without performance benefit — the bottleneck is the talker (replaced by megakernel) and the ConvNet decoder (unchanged either way). The [tts_backend/model.py](tts_backend/model.py) `_HFSwapTTS` strategy hijacks just the talker's forward pass on the upstream module tree, keeping everything else stock. A `_ManualTTS` fallback exists for the case where the upstream API drifts.

### Why single-flight at the server

The megakernel is a non-cooperative single-launch kernel with global scratch buffers. Concurrent requests would clobber each other. `TTSStreamer._lock` enforces one inflight synth at a time; extra requests queue. This is fine for a voice agent (one user one stream) — do not pretend to multi-tenant.

### Streaming cadence

- `emit_every_frames=1` → 80 ms PCM chunks. Lower latency than the upstream default of 8 frames (640 ms), at the cost of higher per-chunk overhead.
- `decode_window_frames=16` → the ConvNet decoder ingests up to 1.28 s of context but only emits the *new* samples per chunk. Tunable per workload.

### Pipecat transport

Default is Daily.co WebRTC because it works in any browser without local NAT/firewall setup. Local SmallWebRTC is available with `--transport local` for offline testing.

## Risks and honest caveats

- **MRoPE**: Qwen3-TTS configs declare 3D-sectioned MRoPE `[24, 20, 20]`. For pure-audio autoregressive decode (talker step k+1 depends only on step k's emit), all three position components advance together, so the implementation starts with the existing 1D RoPE table at `position=step`. Before submission, compare the first 8-16 generated talker tokens against the official PyTorch talker for the same prepared prompt. If tokens diverge, localize the fix to the RoPE apply block in [qwen_megakernel/csrc/kernel.cu](qwen_megakernel/csrc/kernel.cu) (search for `cos_pos`).
- **Talker vocab + tied embeddings**: TTS talker `lm_head` is not tied to `embed_tokens`. The loader handles this; the kernel macro `LDG_VOCAB_SIZE` must be set correctly at *build* time (rebuild after changing it).
- **HF-swap patch surface**: `_HFSwapTTS._attach_talker` probes two patch points. If the upstream module layout changes meaningfully between releases, the manual fallback path runs but may need API touch-ups (`prepare_talker_inputs`, `talker_eos_id`, etc.).
- **First request latency**: warmup-on-boot runs a tiny `Hello.` synthesis. Without it, the first user request also pays JIT compile + CUDA graph capture + cuBLAS plan cost; with it, the first user request is hot.

## Deliverables checklist

- [x] Working repo with build instructions ([scripts/setup.sh](scripts/setup.sh), [Dockerfile](Dockerfile))
- [x] README documenting architecture, kernel mods, run instructions
- [ ] Performance numbers — run [scripts/benchmark.sh](scripts/benchmark.sh) on RTX 5090 and include the generated `bench_results.md`
- [ ] Demo recording (`demo.mp4`) — run [scripts/record_demo.sh](scripts/record_demo.sh) after the RTX 5090 server and Pipecat pipeline are running. The demo should show you talking into the browser, the pipeline transcribing → LLM → megakernel-TTS → audio playback round-trip.

## Credits

- AlpinDale, [qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) (the kernel)
- Elliot Arledge, [MegaQwen](https://github.com/Infatoshi/MegaQwen) (the RTX 3090 ancestor)
- Qwen team, [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
- Pipecat AI, [pipecat](https://github.com/pipecat-ai/pipecat)
