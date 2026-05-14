"""Qwen Megakernel - single-kernel Qwen3-0.6B/TTS talker decode for RTX 5090."""

from qwen_megakernel.build import get_extension as _get_ext

_get_ext()

from qwen_megakernel.model import (  # noqa: E402
    Decoder,
    generate,
    load_talker_weights,
    load_weights,
)

__all__ = ["Decoder", "generate", "load_talker_weights", "load_weights"]
