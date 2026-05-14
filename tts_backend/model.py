"""Qwen3-TTS inference with the AlpinDale megakernel as the talker decoder.

The full Qwen3-TTS-12Hz-0.6B-Base stack is:
    text + audio_prompt
        -> Talker (28-layer Qwen3, 1024 dim)        -- THIS IS THE MEGAKERNEL
        -> hidden state per step
        -> CodePredictor (5-layer transformer, 15 LM heads, parallel)
        -> 16 codebook tokens per frame (12.5 Hz, i.e. 80 ms per frame)
        -> Mimi-style ConvNet decoder (480x upsample, causal, streaming)
        -> 24 kHz PCM waveform

We keep CodePredictor + ConvNet decoder on stock PyTorch (small, not the
bottleneck) and replace just the Talker forward with the megakernel Decoder.

Two integration strategies are supported, selected via `TTSConfig.strategy`:

  - "hf_swap"  (preferred) :: load the official `Qwen3TTSModel` via
        `transformers` `trust_remote_code=True`, then monkey-patch its talker
        forward to call our megakernel Decoder. This reuses the upstream
        Code Predictor + ConvNet decoder + tokenization, minimizing surface
        we have to re-implement.

  - "manual"  (fallback)   :: load weights ourselves and run a hand-rolled
        forward pass for the code predictor and decoder. Used only if the
        upstream class layout changes upstream.

Run `python -m tts_backend.inspect_weights` first to confirm the talker
weight layout before relying on `hf_swap`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import torch


SAMPLE_RATE = 24000
FRAME_RATE_HZ = 12.5
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE_HZ)  # 1920 samples = 80 ms
N_CODEBOOKS = 16


@dataclass
class TTSConfig:
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    strategy: str = "hf_swap"  # or "manual"
    emit_every_frames: int = 1  # 1 frame = 80 ms; emit ASAP for low TTFC
    decode_window_frames: int = 16  # ConvNet streaming window (1.28 s)
    overlap_samples: int = 0
    voice_clone_audio: Optional[str] = None  # path to reference WAV
    voice_clone_text: Optional[str] = None
    verbose: bool = True


class MegakernelTalker:
    """Adapter that exposes a `step(token) -> (token, hidden)` API used by the
    code predictor. Wraps `qwen_megakernel.model.Decoder` for the talker."""

    def __init__(self, model_name: str, verbose: bool = True):
        from qwen_megakernel.model import Decoder

        self._dec = Decoder(model_name=model_name, is_talker=True, verbose=verbose)
        self.vocab_size = self._dec.vocab_size

    def reset(self) -> None:
        self._dec.reset()

    def prefill(self, token_ids: list[int]) -> None:
        self._dec.prefill(token_ids)

    def step(self, token_id: int) -> tuple[int, torch.Tensor]:
        """Returns `(next_token, hidden_fp32[HIDDEN_SIZE])`."""
        return self._dec.step_with_hidden(token_id)

    @property
    def position(self) -> int:
        return self._dec.position


class _HFSwapTTS:
    """Loads the upstream `Qwen3TTSModel` and replaces its talker forward.

    Implementation strategy:
        - `model.talker.lm_head` and `model.talker.model` (the Qwen3 backbone)
          are the slots we hijack. Our `MegakernelTalker.step` produces both
          the next codebook-0 token (replaces sampling from `lm_head`) and the
          post-final-norm hidden state (feeds the code predictor).
        - The official streaming loop in `model.stream_generate_voice_clone`
          (or equivalent) handles: text tokenization, voice-clone prompt
          encoding, code predictor calls, ConvNet decoder, chunk emission.
        - We override only the talker step inside that loop.

    NOTE: The exact monkey-patch surface depends on the upstream class.
    Two patch points are tried (in order). If both fail, fall back to
    `_ManualTTS`. See the `_attach_talker` method.
    """

    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.talker = MegakernelTalker(cfg.model_name, verbose=cfg.verbose)
        self._load_upstream()
        self._attach_talker()

    def _load_upstream(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        if self.cfg.verbose:
            print(f"Loading upstream {self.cfg.model_name} (for code predictor + decoder)...")

        # Prefer the official/space Qwen3TTSModel wrapper when available. The
        # streaming helpers (`create_voice_clone_prompt`, `stream_generate_voice_clone`)
        # live on that wrapper in the public reference implementation; `AutoModel`
        # only returns the underlying `Qwen3TTSForConditionalGeneration`.
        self.wrapper = None
        try:
            try:
                from qwen_tts import Qwen3TTSModel
            except ImportError:
                from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

            self.wrapper = Qwen3TTSModel.from_pretrained(
                self.cfg.model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="cuda",
            )
            self.model = self.wrapper.model
            self.processor = self.wrapper.processor
        except Exception as e:
            if self.cfg.verbose:
                print(f"Qwen3TTSModel wrapper unavailable ({e}); using AutoModel")
            self.model = AutoModel.from_pretrained(
                self.cfg.model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="cuda",
            )
            self.processor = None
        self.model.eval()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name, trust_remote_code=True
            )
        except Exception:
            self.tokenizer = None

    def _attach_talker(self) -> None:
        """Wire our `MegakernelTalker` into the upstream streaming loop.

        Two patch points are supported:

          1. The upstream exposes a `talker.decode_step(token) -> (token, hidden)`
             — we replace that method with ours directly.
          2. The upstream has a `_talker_forward_one_step(input_ids, past_kv)`
             — we wrap it so the megakernel produces token + hidden and we
             return a tuple shaped like the upstream output.

        If neither exists, we leave the upstream talker in place and let the
        caller fall back to `_ManualTTS`.
        """
        talker = self.model.talker if hasattr(self.model, "talker") else self.model

        # Patch point 1
        if hasattr(talker, "decode_step"):
            self._orig_step = talker.decode_step

            def patched_step(token_id, *args, **kwargs):
                return self.talker.step(int(token_id))

            talker.decode_step = patched_step
            self._patched = "decode_step"
            return

        # Patch point 2 — wrap the underlying Qwen3 backbone's forward to
        # surface the post-final-norm hidden state directly. We monkey-patch
        # talker.model.forward so that calls during the streaming loop return
        # a `BaseModelOutputWithPast`-shaped namespace whose `last_hidden_state`
        # is our megakernel hidden; we also stash the next token id and let
        # the upstream sampling logic read it via a side channel.
        if hasattr(talker, "model") and hasattr(talker.model, "forward"):
            self._patch_backbone(talker)
            self._patched = "backbone_forward"
            return

        self._patched = None
        raise RuntimeError(
            "Could not find a patch point on the upstream Qwen3TTSModel."
            " Either the upstream class has changed, or `--strategy manual`"
            " is required."
        )

    def _patch_backbone(self, talker) -> None:
        from types import SimpleNamespace

        megakernel = self.talker
        state = {"new_stream": True, "next_token": 0}

        def patched_forward(input_ids=None, past_key_values=None, **kwargs):
            ids = input_ids.flatten().tolist() if input_ids is not None else []
            if state["new_stream"]:
                megakernel.reset()
                if len(ids) > 1:
                    megakernel.prefill(ids[:-1])
                last = int(ids[-1]) if ids else 0
                state["new_stream"] = False
            else:
                last = int(ids[-1]) if ids else 0
            tok, hidden = megakernel.step(last)
            state["next_token"] = int(tok)
            # Code predictor expects shape [B, T, H] in bf16.
            hidden_btH = hidden.to(torch.bfloat16).view(1, 1, -1)
            return SimpleNamespace(
                last_hidden_state=hidden_btH,
                # qwen_tts accesses outputs.hidden_states (not last_hidden_state)
                # in its talker forward to feed the code predictor.
                hidden_states=hidden_btH,
                past_key_values=past_key_values,
                _megakernel_next_token=tok,
            )

        talker.model.forward = patched_forward
        # Also bypass the upstream lm_head by overriding it to a passthrough
        # that returns the megakernel's already-sampled token id as logits
        # with a one-hot peak. Cheap and correct for greedy sampling, which
        # the megakernel already does.
        if hasattr(talker, "lm_head"):
            vocab = megakernel.vocab_size

            def passthrough_lm_head(hidden):
                # Upstream greedy/sampling code expects logits. The megakernel
                # already computed the argmax, so return logits with a single
                # large peak at that token. This preserves upstream generation
                # plumbing while ensuring the selected token comes from CUDA.
                B, T, _ = hidden.shape
                logits = torch.full(
                    (B, T, vocab), -1e4, dtype=torch.float32, device=hidden.device
                )
                logits[:, -1, state["next_token"]] = 1e4
                return logits

            talker.lm_head.forward = passthrough_lm_head
        self._patch_state = state

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Generate 24 kHz mono PCM chunks for `text` and yield as np.float32."""
        cfg = self.cfg
        if hasattr(self, "_patch_state"):
            self._patch_state["new_stream"] = True

        stream_owner = self.wrapper or self.model

        # qwen_tts >=0.1.1: generate_voice_clone requires ref_audio or
        # voice_clone_prompt. The base model does not support
        # generate_custom_voice (speaker-ID synthesis). When no user reference
        # is supplied, pass a short silence array so the model uses its own
        # default voice distribution.
        if hasattr(stream_owner, "generate_voice_clone"):
            if cfg.voice_clone_audio is not None:
                ref = cfg.voice_clone_audio
                ref_text = cfg.voice_clone_text or ""
            else:
                ref = (np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
                ref_text = " "
            kw: dict = dict(text=text, ref_audio=ref, ref_text=ref_text,
                            non_streaming_mode=False)
            gen = stream_owner.generate_voice_clone(**kw)
        elif hasattr(stream_owner, "generate_custom_voice"):
            speakers = list(stream_owner.get_supported_speakers() or [])
            kw = dict(text=text, speaker=speakers[0] if speakers else "Chelsie",
                      non_streaming_mode=False)
            gen = stream_owner.generate_custom_voice(**kw)
        elif hasattr(stream_owner, "stream_generate_voice_clone"):
            # Legacy upstream API
            kw = dict(
                text=text, language="Auto",
                emit_every_frames=cfg.emit_every_frames,
                decode_window_frames=cfg.decode_window_frames,
                overlap_samples=cfg.overlap_samples,
            )
            if cfg.voice_clone_audio is not None:
                kw["voice_clone_prompt"] = stream_owner.create_voice_clone_prompt(
                    ref_audio=cfg.voice_clone_audio, ref_text=cfg.voice_clone_text or ""
                )
            else:
                kw["voice_clone_prompt"] = _default_voice_prompt(stream_owner)
            for chunk, _sr in stream_owner.stream_generate_voice_clone(**kw):
                yield np.asarray(chunk, dtype=np.float32).reshape(-1)
            return
        else:
            raise RuntimeError(
                "Loaded Qwen3-TTS object does not expose a generation API. "
                f"Available: {[m for m in dir(stream_owner) if not m.startswith('_')]}"
            )

        for audio_batch, _sr in gen:
            batch = audio_batch if isinstance(audio_batch, (list, tuple)) else [audio_batch]
            for chunk in batch:
                yield np.asarray(chunk, dtype=np.float32).reshape(-1)


def _default_voice_prompt(model):
    """Synthesize a tiny 0.5 s silence reference so we can call the voice-clone
    streaming method without a user-supplied speaker. The output sounds like
    the model's prior over the talker's voice distribution (acceptable for
    a demo)."""
    import tempfile

    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
        return model.create_voice_clone_prompt(ref_audio=f.name, ref_text=" ")


class _ManualTTS:
    """Fallback: hand-rolled post-stack if the HF-swap patch points fail.

    Loads the upstream model but does NOT rely on its streaming API. Instead,
    we directly call:
        1. text+prompt encoder        -> initial tokens
        2. megakernel talker          -> codebook-0 + hidden per step
        3. code_predictor             -> codebooks 1..15
        4. decoder (Mimi)             -> 24 kHz PCM chunks

    This path is intentionally conservative and should be validated against
    the official wrapper on the RTX 5090 box before using for submission.
    """

    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.talker = MegakernelTalker(cfg.model_name, verbose=cfg.verbose)
        self._load_upstream()

    def _load_upstream(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        # Importing qwen_tts registers the qwen3_tts model type with
        # transformers' AutoModel registry. Without this, AutoModel.from_pretrained
        # raises ValueError("unknown architecture: qwen3_tts") even with
        # trust_remote_code=True on transformers < 5.x.
        try:
            import qwen_tts  # noqa: F401
        except Exception:
            pass

        self.model = AutoModel.from_pretrained(
            self.cfg.model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.model.eval()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name, trust_remote_code=True
            )
        except Exception:
            self.tokenizer = None

        # Locate code_predictor + decoder on the upstream module tree
        self.code_predictor = (
            getattr(self.model, "code_predictor", None)
            or getattr(self.model, "code_pred", None)
        )
        self.audio_decoder = (
            getattr(self.model, "audio_decoder", None)
            or getattr(self.model, "decoder", None)
            or getattr(self.model, "speech_tokenizer", None)
        )
        if self.code_predictor is None or self.audio_decoder is None:
            raise RuntimeError(
                "Manual strategy could not locate code_predictor / audio_decoder."
            )

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Run the full pipeline manually.

        IMPORTANT: This path makes assumptions about the upstream APIs that
        SHOULD be validated on the GPU box via inspect_weights. If it
        breaks at runtime, use `strategy="hf_swap"`.
        """
        cfg = self.cfg
        # 1. Text encoding -> initial talker token sequence (prompt)
        prompt_ids = _encode_text_for_talker(self.model, self.tokenizer, text)
        self.talker.reset()
        if len(prompt_ids) > 1:
            self.talker.prefill(prompt_ids[:-1])
        last = prompt_ids[-1] if prompt_ids else 0

        # Streaming buffers
        codes_buf: list[torch.Tensor] = []  # list of [N_CODEBOOKS] long tensors
        hidden_buf: list[torch.Tensor] = []

        eos_id = _talker_eos_id(self.model)
        max_frames = int(cfg.decode_window_frames * 64)  # safety cap

        emitted_frames = 0
        with torch.inference_mode():
            for _ in range(max_frames):
                tok, hidden = self.talker.step(last)
                last = tok
                if tok == eos_id:
                    break

                # Code predictor: produce codebooks 1..15 in parallel from hidden
                hidden_bf16 = hidden.to(torch.bfloat16).view(1, 1, -1)
                cb_rest = self.code_predictor(hidden_bf16)  # [1, 1, 15] or list
                cb_rest = _flatten_cb_rest(cb_rest).view(-1)  # [15]
                frame_codes = torch.empty(N_CODEBOOKS, dtype=torch.long, device="cuda")
                frame_codes[0] = tok
                frame_codes[1:] = cb_rest
                codes_buf.append(frame_codes)
                hidden_buf.append(hidden)

                # Emit cadence: every `emit_every_frames`, push a PCM chunk
                if (len(codes_buf) - emitted_frames) >= cfg.emit_every_frames:
                    window_start = max(0, len(codes_buf) - cfg.decode_window_frames)
                    window = torch.stack(codes_buf[window_start:], dim=0)
                    # window shape: [T, N_CODEBOOKS]
                    audio = self.audio_decoder(window.unsqueeze(0))  # [1, samples]
                    audio = audio.float().cpu().numpy().reshape(-1)
                    new_samples = audio[-cfg.emit_every_frames * SAMPLES_PER_FRAME :]
                    emitted_frames = len(codes_buf)
                    yield new_samples


def _encode_text_for_talker(model, tokenizer, text: str) -> list[int]:
    """Produce the talker's initial token sequence from raw text.

    Upstream Qwen3-TTS prepends special tokens for language/voice/etc. Prefer
    the model's own `prepare_talker_inputs` helper. Plain tokenizer encoding is
    only a last-resort dev fallback and should not be used for final numbers.
    """
    if hasattr(model, "prepare_talker_inputs"):
        return list(model.prepare_talker_inputs(text=text))
    if tokenizer is not None:
        return tokenizer.encode(text, add_special_tokens=True)
    raise RuntimeError("No way to encode text for talker — provide a tokenizer.")


def _talker_eos_id(model) -> int:
    if hasattr(model, "talker_eos_id"):
        return int(model.talker_eos_id)
    if hasattr(model, "config") and hasattr(model.config, "talker_eos_token_id"):
        return int(model.config.talker_eos_token_id)
    return -1  # sentinel: never matches


def _flatten_cb_rest(cb) -> torch.Tensor:
    """Normalize code predictor output to a [15] long tensor."""
    if isinstance(cb, (list, tuple)):
        return torch.stack([t.argmax(-1).view(-1) for t in cb])
    if cb.ndim >= 2:
        return cb.argmax(-1).view(-1)
    return cb.view(-1)


# =============================================================================
# Public API
# =============================================================================


@dataclass
class TTSEngine:
    """Top-level facade. Constructs the right backend strategy."""

    cfg: TTSConfig = field(default_factory=TTSConfig)
    _impl: object = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.cfg.strategy == "hf_swap":
            try:
                self._impl = _HFSwapTTS(self.cfg)
                return
            except Exception as e:
                if self.cfg.verbose:
                    print(f"HF-swap strategy failed ({e}); falling back to manual")
        self._impl = _ManualTTS(self.cfg)

    def stream(self, text: str) -> Iterator[np.ndarray]:
        return self._impl.stream(text)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE


__all__ = [
    "FRAME_RATE_HZ",
    "MegakernelTalker",
    "N_CODEBOOKS",
    "SAMPLE_RATE",
    "SAMPLES_PER_FRAME",
    "TTSConfig",
    "TTSEngine",
]
