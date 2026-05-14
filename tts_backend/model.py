from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 24000
FRAME_RATE_HZ = 12.5
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE_HZ)  # 1920 samples = 80 ms
N_CODEBOOKS = 16


@dataclass
class TTSConfig:
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    strategy: str = "hf_swap"
    emit_every_frames: int = 1
    decode_window_frames: int = 16
    overlap_samples: int = 0
    voice_clone_audio: Optional[str] = None
    voice_clone_text: Optional[str] = None
    verbose: bool = True


class MegakernelTalker:
    """Wraps qwen_megakernel.Decoder to expose the step-with-hidden API."""

    def __init__(self, model_name: str, verbose: bool = True):
        from qwen_megakernel.model import Decoder

        self._dec = Decoder(model_name=model_name, is_talker=True, verbose=verbose)
        self.vocab_size = self._dec.vocab_size

    def reset(self) -> None:
        self._dec.reset()

    def prefill(self, token_ids: list[int]) -> None:
        self._dec.prefill(token_ids)

    def step(self, token_id: int) -> tuple[int, torch.Tensor]:
        return self._dec.step_with_hidden(token_id)

    def prefill_step(self, token_id: int) -> tuple[int, torch.Tensor]:
        return self._dec.prefill_step(token_id)

    @property
    def position(self) -> int:
        return self._dec.position


class _HFSwapTTS:
    """Loads Qwen3TTSModel and replaces its talker forward with the megakernel."""

    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.talker = MegakernelTalker(cfg.model_name, verbose=cfg.verbose)
        self._load_upstream()
        self._attach_talker()

    def _load_upstream(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        if self.cfg.verbose:
            print(f"Loading upstream {self.cfg.model_name}...")

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
                print(f"Qwen3TTSModel unavailable ({e}); using AutoModel")
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
        talker = self.model.talker if hasattr(self.model, "talker") else self.model

        if hasattr(talker, "decode_step"):
            self._orig_step = talker.decode_step

            def patched_step(token_id, *args, **kwargs):
                return self.talker.step(int(token_id))

            talker.decode_step = patched_step
            self._patched = "decode_step"
            return

        if hasattr(talker, "model") and hasattr(talker.model, "forward"):
            self._patch_backbone(talker)
            self._patched = "backbone_forward"
            return

        raise RuntimeError(
            "Could not find a patch point on the upstream Qwen3TTSModel. "
            "Try --strategy manual."
        )

    def _patch_backbone(self, talker) -> None:
        megakernel = self.talker
        state = {"new_stream": True, "next_token": 0}
        _orig_forward = talker.model.forward

        def patched_forward(
            input_ids=None, past_key_values=None, inputs_embeds=None, **kwargs
        ):
            # qwen_tts always passes inputs_embeds to the backbone; the megakernel
            # needs integer token IDs, so delegate those calls to the original forward.
            if inputs_embeds is not None:
                return _orig_forward(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    **kwargs,
                )

            ids = input_ids.flatten().tolist() if input_ids is not None else []
            if state["new_stream"]:
                megakernel.reset()
                if len(ids) > 1:
                    megakernel.prefill(ids[:-1])
                last = int(ids[-1]) if ids else 0
                state["new_stream"] = False
                tok, hidden = megakernel.prefill_step(last)
            else:
                last = int(ids[-1]) if ids else 0
                tok, hidden = megakernel.step(last)

            state["next_token"] = int(tok)
            hidden_btH = hidden.to(torch.bfloat16).view(1, 1, -1)
            return SimpleNamespace(
                last_hidden_state=hidden_btH,
                hidden_states=hidden_btH,
                past_key_values=past_key_values,
                attentions=None,
                cross_attentions=None,
                _megakernel_next_token=tok,
            )

        talker.model.forward = patched_forward

        if hasattr(talker, "lm_head"):
            vocab = megakernel.vocab_size

            def passthrough_lm_head(hidden):
                B, T, _ = hidden.shape
                logits = torch.full(
                    (B, T, vocab), -1e4, dtype=torch.float32, device=hidden.device
                )
                logits[:, -1, state["next_token"]] = 1e4
                return logits

            talker.lm_head.forward = passthrough_lm_head

        self._patch_state = state

    def stream(self, text: str) -> Iterator[np.ndarray]:
        cfg = self.cfg
        if hasattr(self, "_patch_state"):
            self._patch_state["new_stream"] = True

        stream_owner = self.wrapper or self.model

        if hasattr(stream_owner, "generate_voice_clone"):
            if cfg.voice_clone_audio is not None:
                ref = cfg.voice_clone_audio
                ref_text = cfg.voice_clone_text or ""
            else:
                ref = (np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
                ref_text = " "
            audio_list, _sr = stream_owner.generate_voice_clone(
                text=text, ref_audio=ref, ref_text=ref_text, non_streaming_mode=True
            )
            chunks = (
                audio_list if isinstance(audio_list, (list, tuple)) else [audio_list]
            )
            for chunk in chunks:
                yield np.asarray(chunk, dtype=np.float32).reshape(-1)

        elif hasattr(stream_owner, "generate_custom_voice"):
            speakers = list(stream_owner.get_supported_speakers() or [])
            audio_list, _sr = stream_owner.generate_custom_voice(
                text=text,
                speaker=speakers[0] if speakers else "Chelsie",
                non_streaming_mode=True,
            )
            chunks = (
                audio_list if isinstance(audio_list, (list, tuple)) else [audio_list]
            )
            for chunk in chunks:
                yield np.asarray(chunk, dtype=np.float32).reshape(-1)

        elif hasattr(stream_owner, "stream_generate_voice_clone"):
            kw: dict = dict(
                text=text,
                language="Auto",
                emit_every_frames=cfg.emit_every_frames,
                decode_window_frames=cfg.decode_window_frames,
                overlap_samples=cfg.overlap_samples,
            )
            if cfg.voice_clone_audio is not None:
                kw["voice_clone_prompt"] = stream_owner.create_voice_clone_prompt(
                    ref_audio=cfg.voice_clone_audio, ref_text=cfg.voice_clone_text or ""
                )
            else:
                kw["voice_clone_prompt"] = _silence_voice_prompt(stream_owner)
            for chunk, _sr in stream_owner.stream_generate_voice_clone(**kw):
                yield np.asarray(chunk, dtype=np.float32).reshape(-1)

        else:
            raise RuntimeError(
                f"Qwen3-TTS object has no generation API. "
                f"Available: {[m for m in dir(stream_owner) if not m.startswith('_')]}"
            )


def _silence_voice_prompt(model):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(
            f.name, np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE
        )
        return model.create_voice_clone_prompt(ref_audio=f.name, ref_text=" ")


class _ManualTTS:
    """Fallback when hf_swap patch points fail — hand-rolls the full post-stack."""

    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.talker = MegakernelTalker(cfg.model_name, verbose=cfg.verbose)
        self._load_upstream()

    def _load_upstream(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        try:
            import qwen_tts  # noqa: F401 — registers qwen3_tts with AutoModel
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

        self.code_predictor = getattr(self.model, "code_predictor", None) or getattr(
            self.model, "code_pred", None
        )
        self.audio_decoder = (
            getattr(self.model, "audio_decoder", None)
            or getattr(self.model, "decoder", None)
            or getattr(self.model, "speech_tokenizer", None)
        )
        if self.code_predictor is None or self.audio_decoder is None:
            raise RuntimeError(
                "Manual strategy: could not locate code_predictor / audio_decoder."
            )

    def stream(self, text: str) -> Iterator[np.ndarray]:
        cfg = self.cfg
        prompt_ids = _encode_text_for_talker(self.model, self.tokenizer, text)
        self.talker.reset()
        if len(prompt_ids) > 1:
            self.talker.prefill(prompt_ids[:-1])
        last = prompt_ids[-1] if prompt_ids else 0

        codes_buf: list[torch.Tensor] = []
        eos_id = _talker_eos_id(self.model)
        max_frames = int(cfg.decode_window_frames * 64)
        emitted_frames = 0

        with torch.inference_mode():
            for _ in range(max_frames):
                tok, hidden = self.talker.step(last)
                last = tok
                if tok == eos_id:
                    break

                hidden_bf16 = hidden.to(torch.bfloat16).view(1, 1, -1)
                cb_rest = _flatten_cb_rest(self.code_predictor(hidden_bf16)).view(-1)
                frame_codes = torch.empty(N_CODEBOOKS, dtype=torch.long, device="cuda")
                frame_codes[0] = tok
                frame_codes[1:] = cb_rest
                codes_buf.append(frame_codes)

                if (len(codes_buf) - emitted_frames) >= cfg.emit_every_frames:
                    window_start = max(0, len(codes_buf) - cfg.decode_window_frames)
                    window = torch.stack(codes_buf[window_start:], dim=0)
                    audio = (
                        self.audio_decoder(window.unsqueeze(0))
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    yield audio[-cfg.emit_every_frames * SAMPLES_PER_FRAME :]
                    emitted_frames = len(codes_buf)


def _encode_text_for_talker(model, tokenizer, text: str) -> list[int]:
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
    return -1


def _flatten_cb_rest(cb) -> torch.Tensor:
    if isinstance(cb, (list, tuple)):
        return torch.stack([t.argmax(-1).view(-1) for t in cb])
    if cb.ndim >= 2:
        return cb.argmax(-1).view(-1)
    return cb.view(-1)


@dataclass
class TTSEngine:
    cfg: TTSConfig = field(default_factory=TTSConfig)
    _impl: object = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.cfg.strategy == "hf_swap":
            try:
                self._impl = _HFSwapTTS(self.cfg)
                return
            except Exception as e:
                if self.cfg.verbose:
                    print(f"HF-swap failed ({e}); falling back to manual")
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
