"""Weight loading and high-level decode API for Qwen3-0.6B / Qwen3-TTS talker.

The talker decoder inside `Qwen/Qwen3-TTS-12Hz-0.6B-Base` is structurally
identical to `Qwen/Qwen3-0.6B` (28L, hidden=1024, Q=16/KV=8 GQA, head_dim=128,
intermediate=3072). The only deltas for TTS are:
    1. weights live under a `talker.*` prefix
    2. the lm_head projects to the audio codebook-0 vocab (~2048+specials),
       not the 151936-token text vocab
    3. `rope_theta` may differ from the base 1_000_000

This file gracefully handles all three via constructor kwargs.
"""

import math
import struct

import torch

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM  # 2048
KV_SIZE = 8 * HEAD_DIM  # 1024
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 151936  # default; overridable for TTS talker

_decode = torch.ops.qwen_megakernel_C.decode


def build_rope_tables(
    rope_theta: float = 1_000_000.0, max_seq_len: int = MAX_SEQ_LEN
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin RoPE tables matching the kernel's `cos_table[pos]` layout.

    Kernel expects shape [max_seq_len, HEAD_DIM] bf16, where each row is
    `[cos(freq_0*pos), ..., cos(freq_{H/2-1}*pos)]` repeated twice along dim.
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_seq_len, HEAD_DIM/2]
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    return cos_table, sin_table


def _quiet_hf(verbose: bool) -> None:
    if verbose:
        return
    import os

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except AttributeError:
        pass
    try:
        from huggingface_hub import logging as hf_hub_logging

        hf_hub_logging.set_verbosity_error()
    except Exception:
        pass


def _extract_layer_blob(state: dict, prefix: str) -> list[torch.Tensor]:
    """Extract the 11-tensor-per-layer flat list from a state dict.

    `prefix` is `"model.layers."` for base Qwen3, or `"talker.model.layers."`
    for the Qwen3-TTS talker (key names match the HF Qwen3 module tree).
    """
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"{prefix}{i}."
        layer_weights.extend(
            [
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state[p + "self_attn.q_norm.weight"].contiguous(),
                state[p + "self_attn.k_norm.weight"].contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ]
        )
    return layer_weights


def load_weights(
    model_name: str = "Qwen/Qwen3-0.6B",
    verbose: bool = True,
    rope_theta: float | None = None,
):
    """Load Qwen3-0.6B weights from HuggingFace into GPU tensors.

    For the TTS talker, prefer `load_talker_weights()` instead.
    """
    _quiet_hf(verbose)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if verbose:
        print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()
    cfg = model.config

    theta = (
        rope_theta
        if rope_theta is not None
        else float(getattr(cfg, "rope_theta", 1_000_000.0))
    )
    cos_table, sin_table = build_rope_tables(theta)

    layer_weights = _extract_layer_blob(state, "model.layers.")
    embed_weight = state["model.embed_tokens.weight"].contiguous()
    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=state["model.norm.weight"].contiguous(),
        lm_head_weight=embed_weight,  # tied embeddings for Qwen3-0.6B base
        cos_table=cos_table,
        sin_table=sin_table,
        vocab_size=int(getattr(cfg, "vocab_size", VOCAB_SIZE)),
        rope_theta=theta,
    )

    del model
    torch.cuda.empty_cache()
    return weights, tokenizer


def load_talker_weights(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    verbose: bool = True,
):
    """Load the Qwen3-TTS talker (audio LLM) weights into GPU tensors.

    Returns a dict suitable for `Decoder(weights=..., vocab_size=...)`. The
    talker shares architecture with Qwen3-0.6B but emits codebook-0 audio
    tokens, so vocab_size differs and lm_head is **not** tied to embeddings.

    The exact prefix used in the HF snapshot depends on whether the package
    wraps the modules under `talker.model.*` (full TTS model) or exposes the
    talker as a standalone `model.*` (talker-only checkpoint). This function
    auto-detects either layout.
    """
    _quiet_hf(verbose)

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    if verbose:
        print(f"Loading talker weights from {model_name}...")

    local_dir = snapshot_download(
        repo_id=model_name, allow_patterns=["*.safetensors", "*.json"]
    )
    import glob
    import json
    import os

    # Merge all shards into one state dict (talker is small, fits in memory)
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
        state.update(load_file(shard, device="cuda"))

    # Auto-detect prefix
    if any(k.startswith("talker.model.layers.") for k in state):
        layer_prefix = "talker.model.layers."
        embed_key = "talker.model.embed_tokens.weight"
        norm_key = "talker.model.norm.weight"
        lm_head_key = "talker.lm_head.weight"
    elif any(k.startswith("model.layers.") for k in state):
        layer_prefix = "model.layers."
        embed_key = "model.embed_tokens.weight"
        norm_key = "model.norm.weight"
        lm_head_key = "lm_head.weight"
    else:
        raise RuntimeError(
            "Could not find Qwen3 layer weights in checkpoint. "
            f"Top-level keys: {list(state.keys())[:8]}..."
        )

    layer_weights = _extract_layer_blob(state, layer_prefix)
    embed_weight = state[embed_key].to(torch.bfloat16).contiguous()
    final_norm_weight = state[norm_key].to(torch.bfloat16).contiguous()
    lm_head_weight = state[lm_head_key].to(torch.bfloat16).contiguous()

    # Resolve rope_theta + vocab from config.json
    cfg_path = os.path.join(local_dir, "config.json")
    rope_theta = 1_000_000.0
    vocab_size = int(lm_head_weight.shape[0])
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        # Talker config may be nested under "talker_config" or top-level
        talker_cfg = cfg.get("talker_config", cfg)
        rope_theta = float(talker_cfg.get("rope_theta", rope_theta))
        if "vocab_size" in talker_cfg:
            vocab_size = int(talker_cfg["vocab_size"])

    cos_table, sin_table = build_rope_tables(rope_theta)

    # Cast layer weights to bf16 (they should already be, but enforce)
    layer_weights = [t.to(torch.bfloat16).contiguous() for t in layer_weights]

    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=final_norm_weight,
        lm_head_weight=lm_head_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        vocab_size=vocab_size,
        rope_theta=rope_theta,
    )

    # Tokenizer-equivalent for the talker: the audio codebook is a plain int
    # range, no BPE. The text tokenizer (used for the *text* side of the talker
    # input) lives on the full repo's tokenizer.json.
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        tokenizer = None

    if verbose:
        print(
            f"  vocab_size={vocab_size}, rope_theta={rope_theta}, "
            f"lm_head shape={tuple(lm_head_weight.shape)}"
        )
    return weights, tokenizer


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8  # 64-bit pointers
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8).cuda()
    return t


class Decoder:
    """Stateful decoder wrapping the Qwen Megakernel torch ops.

    Works for both Qwen3-0.6B base and the Qwen3-TTS talker. The vocab size
    is taken from the loaded weights and must match the kernel's compile-time
    `LDG_VOCAB_SIZE` (set via the `LDG_VOCAB_SIZE` env var before first build).
    """

    def __init__(
        self,
        weights=None,
        tokenizer=None,
        model_name="Qwen/Qwen3-0.6B",
        verbose: bool = True,
        is_talker: bool = False,
    ):
        if weights is None:
            if is_talker:
                weights, tokenizer = load_talker_weights(model_name, verbose=verbose)
            else:
                weights, tokenizer = load_weights(model_name, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0
        self.vocab_size = int(weights.get("vocab_size", VOCAB_SIZE))
        self.rope_theta = float(weights.get("rope_theta", 1_000_000.0))

        # Keep references so tensors stay alive (prevents GC of weight memory).
        self._weights = weights

        # Model weights (read-only, shared across calls)
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache
        self._k_cache = torch.zeros(
            NUM_LAYERS,
            NUM_KV_HEADS,
            MAX_SEQ_LEN,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers (single-token decode)
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def step(self, token_id: int) -> int:
        """Decode one token. Returns the next token id."""
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def step_with_hidden(self, token_id: int) -> tuple[int, torch.Tensor]:
        """Decode one token; return `(next_token, hidden_state)`.

        `hidden_state` is the post-final-RMSNorm activation `[HIDDEN_SIZE]`
        in fp32 on the GPU — exactly what the Code Predictor wants as input.
        The returned tensor is a CLONE of the kernel's scratch buffer, so it
        survives the next step.
        """
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        hidden = self._norm_out.clone()
        return self._out_token.item(), hidden

    def prefill(self, token_ids: list[int]) -> None:
        """Step through a prompt without keeping outputs (warm KV cache)."""
        for tid in token_ids:
            self.step(int(tid))

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        _gen = torch.ops.qwen_megakernel_C.generate_nosync
        output_ids = _gen(
            ids[-1],
            max_tokens,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += max_tokens
        out = output_ids.cpu().tolist()
        eos = self.tokenizer.eos_token_id
        if eos in out:
            out = out[: out.index(eos)]
        return self.tokenizer.decode(out, skip_special_tokens=True)


def generate(prompt: str, max_tokens: int = 100, verbose: bool = True) -> str:
    """One-shot convenience: load model, generate, return text."""
    return Decoder(verbose=verbose).generate(prompt, max_tokens)
