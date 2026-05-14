"""Pure megakernel decode throughput — no code predictor, no audio decoder.

Usage:
    python -m benchmarks.throughput --model talker --runs 20
    python -m benchmarks.throughput --model base   --runs 20
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import torch

from benchmarks._common import print_table, summarize


def bench_run(decoder, prompt_tokens: int, gen_tokens: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    decoder.reset()
    seed = 0
    decoder.prefill([seed] * (prompt_tokens - 1))
    last = seed
    for _ in range(gen_tokens):
        last = decoder.step(last)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("base", "talker"), default="talker")
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--prompt-tokens", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--model-name", default=os.getenv("BENCH_MODEL_NAME"))
    args = p.parse_args()

    from qwen_megakernel.model import Decoder

    if args.model == "base":
        name = args.model_name or "Qwen/Qwen3-0.6B"
        is_talker = False
    else:
        name = args.model_name or "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        is_talker = True

    print(f"Loading {name} (is_talker={is_talker})...")
    decoder = Decoder(model_name=name, is_talker=is_talker, verbose=False)
    print(f"vocab_size={decoder.vocab_size}, rope_theta={decoder.rope_theta}")

    for _ in range(args.warmup):
        bench_run(decoder, args.prompt_tokens, args.tokens)
    gc.collect()
    torch.cuda.empty_cache()

    samples_total_ms = []
    samples_per_tok_ms = []
    for _ in range(args.runs):
        dt = bench_run(decoder, args.prompt_tokens, args.tokens)
        samples_total_ms.append(dt * 1000.0)
        samples_per_tok_ms.append(dt * 1000.0 / args.tokens)

    rows = [
        summarize(f"{args.model}: total wall time", samples_total_ms),
        summarize(f"{args.model}: ms per token", samples_per_tok_ms),
    ]
    print_table(rows)

    tps_samples = [args.tokens / (ms / 1000.0) for ms in samples_total_ms]
    print()
    print(summarize(f"{args.model}: tokens / sec", tps_samples).format())


if __name__ == "__main__":
    main()
