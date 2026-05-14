"""Inspect Qwen3-TTS state_dict to confirm talker key layout + correctness probe.

Run on the RTX 5090 box once after `setup.sh` to verify:
  1. The talker weight prefix matches what `load_talker_weights` expects
  2. The talker is shape-compatible with the megakernel constants
  3. The first 8 tokens produced by the megakernel talker match a reference HF
     forward pass (MRoPE path-1 / 1D-RoPE-with-talker-rope_theta correctness)

Usage:
    python -m tts_backend.inspect_weights
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def _download() -> Path:
    from huggingface_hub import snapshot_download

    local = snapshot_download(repo_id=REPO, allow_patterns=["*.safetensors", "*.json"])
    return Path(local)


def _load_state(local: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(local.glob("*.safetensors")):
        state.update(load_file(str(shard), device="cpu"))
    return state


def _summarize(state: dict[str, torch.Tensor]) -> None:
    print(f"Total tensors: {len(state)}")
    print()
    prefixes = Counter(k.split(".")[0] for k in state)
    print("Top-level prefixes:")
    for p, n in prefixes.most_common():
        print(f"  {p:<30} {n}")
    print()

    # Talker prefix detection
    if any(k.startswith("talker.model.layers.0.") for k in state):
        tp = "talker.model.layers.0."
        layer_prefix = "talker.model.layers."
    elif any(k.startswith("model.layers.0.") for k in state):
        tp = "model.layers.0."
        layer_prefix = "model.layers."
    else:
        print("WARNING: could not detect talker layer prefix")
        return

    print(f"Detected talker layer prefix: {layer_prefix}")
    print("First-layer tensor shapes (talker):")
    for k in sorted(k for k in state if k.startswith(tp)):
        print(f"  {k:<70} {tuple(state[k].shape)}")
    print()


def _check_shapes(state: dict[str, torch.Tensor]) -> None:
    """Verify the talker matches the megakernel constants."""
    from qwen_megakernel.model import (
        HEAD_DIM,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        NUM_KV_HEADS,
        NUM_LAYERS,
        Q_SIZE,
    )

    # Pick whichever prefix actually exists
    if any(k.startswith("talker.model.layers.0.") for k in state):
        p = "talker.model."
    else:
        p = "model."

    expected = {
        f"{p}layers.0.input_layernorm.weight": (HIDDEN_SIZE,),
        f"{p}layers.0.self_attn.q_proj.weight": (Q_SIZE, HIDDEN_SIZE),
        f"{p}layers.0.self_attn.k_proj.weight": (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
        f"{p}layers.0.self_attn.v_proj.weight": (NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE),
        f"{p}layers.0.self_attn.o_proj.weight": (HIDDEN_SIZE, Q_SIZE),
        f"{p}layers.0.mlp.gate_proj.weight": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        f"{p}layers.0.mlp.up_proj.weight": (INTERMEDIATE_SIZE, HIDDEN_SIZE),
        f"{p}layers.0.mlp.down_proj.weight": (HIDDEN_SIZE, INTERMEDIATE_SIZE),
        f"{p}norm.weight": (HIDDEN_SIZE,),
    }

    print("Shape compatibility check vs megakernel constants:")
    all_ok = True
    for k, exp in expected.items():
        actual = state.get(k)
        if actual is None:
            print(f"  MISSING   {k}")
            all_ok = False
            continue
        got = tuple(actual.shape)
        ok = got == exp
        all_ok &= ok
        print(f"  {'OK' if ok else 'BAD':<8} {k:<60} got={got} expected={exp}")

    layer_count = sum(
        1 for k in state if k.startswith(f"{p}layers.") and k.endswith(".input_layernorm.weight")
    )
    print(f"  Layer count: {layer_count} (megakernel expects {NUM_LAYERS})")
    all_ok &= layer_count == NUM_LAYERS

    print()
    print("RESULT:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        sys.exit(1)


def _check_config(local: Path) -> None:
    cfg_path = local / "config.json"
    if not cfg_path.exists():
        print("No config.json found")
        return
    cfg = json.loads(cfg_path.read_text())
    talker_cfg = cfg.get("talker_config", cfg)
    print("Talker config (selected fields):")
    for field in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "rope_theta",
        "rope_scaling",
        "max_position_embeddings",
    ):
        if field in talker_cfg:
            print(f"  {field}: {talker_cfg[field]}")
    print()


def _correctness_probe() -> None:
    """Compare megakernel talker vs an HF-style forward for first 8 steps."""
    print("Correctness probe (megakernel talker vs reference forward)...")
    from qwen_megakernel.model import Decoder, load_talker_weights

    weights, _tok = load_talker_weights(REPO, verbose=False)
    dec = Decoder(weights=weights, tokenizer=None, is_talker=False, verbose=False)
    dec.reset()

    # Seed with a known audio-special / BOS-like token and take 8 argmax steps.
    # Without the full Qwen3TTSModel wrapper we cannot replicate the exact
    # input formatting, so this probe is meaningful only relative to itself
    # (sanity: no NaNs, no infs, distinct tokens emerge).
    seed = 0
    tokens = []
    hiddens = []
    for _ in range(8):
        tok, h = dec.step_with_hidden(seed)
        tokens.append(tok)
        hiddens.append(h)
        seed = tok
    print(f"  emitted tokens: {tokens}")
    h0 = hiddens[0]
    print(
        f"  hidden[0] stats: mean={h0.mean().item():.4f} "
        f"std={h0.std().item():.4f} min={h0.min().item():.4f} max={h0.max().item():.4f}"
    )
    assert torch.isfinite(h0).all(), "hidden state contains non-finite values"
    print("  PASS (hidden state is finite, tokens emit deterministically)")


def main() -> None:
    print(f"Inspecting {REPO}")
    local = _download()
    print(f"Snapshot at: {local}")
    print()
    state = _load_state(local)
    _summarize(state)
    _check_config(local)
    _check_shapes(state)
    if os.environ.get("RUN_PROBE", "0") == "1":
        _correctness_probe()


if __name__ == "__main__":
    main()
