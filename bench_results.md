# Benchmark Results

**Hardware**: RTX 5090 (Vast.ai), CUDA 12.8, PyTorch 2.7, bfloat16  
**Model**: Qwen3-TTS-12Hz-0.6B-Base  
**Date**: 2026-05-14

---

## Megakernel throughput (talker decoder only)

| Metric | Result | Target |
|---|---|---|
| Tokens / sec | **829** (median 832, p95 846) | ~1,000 |
| ms / token | 1.21 ms | — |
| Talker RTF | **0.014** | < 0.15 ✅ |

At 12 Hz frame rate, 1 s of audio = 12 tokens → 12 × 1.21 ms = 14.5 ms to generate 1 s of audio.  
The megakernel comfortably hits the talker RTF target.

---

## Full pipeline RTF (talker → code predictor → Mimi → PCM)

| Text length | RTF |
|---|---|
| Short (~1 s) | 1.04 |
| Medium (~5 s) | 1.02 |
| Long (~10 s) | 1.02 |

**Target < 0.3: not met.** The bottleneck is the code predictor (5-layer PyTorch attention) and Mimi ConvNet decoder — both running on stock PyTorch without flash-attention. The megakernel itself is fine; those two stages dominate.

---

## TTFC (time to first audio chunk)

| Text | TTFC |
|---|---|
| "Hello." | 3,665 ms |
| "The quick brown fox..." | 6,303 ms |

**Target < 90 ms: not met.** The server currently uses `non_streaming_mode=True` — a workaround for a dimension-mismatch bug in the frame-by-frame streaming path. This means all frames are generated before the first byte is sent, so TTFC equals total generation time.

With true frame-by-frame streaming re-enabled: estimated TTFC ~60–130 ms (1 talker token ~1.2 ms + 1 code predictor step ~50–100 ms + 1 Mimi frame ~10–30 ms).

---

## End-to-end round-trip (live demo)

| Stage | Observed |
|---|---|
| Speech → STT | ~300–600 ms |
| STT → LLM reply | ~400–800 ms |
| LLM → first audio chunk | 3–7 s (TTS bottleneck) |
| **Total** | **~4–9 s** |

---

## What's blocking the targets

Two things, both fixable:

1. **Streaming mode is disabled** — re-enabling frame-by-frame emit would bring TTFC to ~60–130 ms and is the most impactful single change.
2. **No flash-attention** — the code predictor uses stock PyTorch attention. With flash-attn on Blackwell, code predictor steps would be ~3–5× faster, pulling full pipeline RTF down to ~0.3–0.5.

The megakernel integration itself is correct. The kernel runs at 0.014 RTF for the talker stage — the remaining bottleneck is entirely in the PyTorch post-stack.

---

## Audio quality

Clean. No glitches or dropped frames observed across all test utterances in the live demo.
