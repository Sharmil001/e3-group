## Qwen3 0.6B Megakernel for RTX 5090

This is a vendored fork of AlpinDale's Qwen3-0.6B megakernel, kept in-tree so
the parent project can build and run without a separate submodule checkout. It
is aggressively optimized for Qwen3-0.6B bf16 shapes on an RTX 5090.

More details on the original optimization work:
https://blog.alpindale.net/posts/5090_decode_optimization/

This copy also supports the Qwen3-TTS talker path used by the parent project:

- `LDG_VOCAB_SIZE` can be set at build time. Use `151936` for base
  `Qwen/Qwen3-0.6B`, or `2052` for the Qwen3-TTS talker.
- `Decoder.step_with_hidden()` returns the generated token plus the post-final
  RMSNorm hidden state needed by the TTS code predictor.
- `load_talker_weights()` loads Qwen3-TTS talker checkpoints with either
  `talker.model.*` or standalone `model.*` weight prefixes.

Reference throughput from the upstream RTX 5090 run:

| Backend | tok/s | ms/tok | Speedup |
| --- | ---: | ---: | ---: |
| PyTorch (HF) | 123.3 | 8.11 | 1.00x |
| Megakernel | 1036.3 | 0.99 | 8.40x |

### Usage

```bash
uv pip install -r requirements.txt
python -m qwen_megakernel.bench
```

For the TTS talker build:

```bash
LDG_VOCAB_SIZE=2052 python -c "from qwen_megakernel.build import get_extension; get_extension()"
```

Not tested on other GPUs, and likely will not run correctly outside the RTX
5090 / CUDA 12.8 target.


### Credits

Based on AlpinDale's [qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel)
and Elliot Arledge's [MegaQwen](https://github.com/Infatoshi/MegaQwen) for the
RTX 3090 GPU.
