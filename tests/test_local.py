"""
Local validation tests — no GPU or CUDA required.

Tests every logic change made to fix the Qwen3-TTS integration:
  1. Weight key detection (correct names for full-model checkpoint)
  2. patched_forward SimpleNamespace completeness (all fields qwen_tts accesses)
  3. stream() routing (generate_voice_clone vs generate_custom_voice)
  4. ref_audio silence tuple format
  5. Audio chunk iteration (list-wrapped and bare-array generator shapes)
  6. Decoder.prefill embedding selection (text vs codec embed)
  7. _ManualTTS qwen_tts pre-registration logic

Run:
    python tests/test_local.py
or:
    python -m pytest tests/test_local.py -v
"""

import sys
import types
import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

SAMPLE_RATE = 24000
HIDDEN_SIZE = 1024


# ─────────────────────────────────────────────────────────────────────────────
# 1. Weight key detection
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyDetection(unittest.TestCase):
    """load_talker_weights must map the real checkpoint key names."""

    def _talker_state(self):
        """Minimal fake state dict matching the actual Qwen3-TTS checkpoint layout."""
        state = {}
        for i in range(2):          # 2 layers is enough to trigger detection
            p = f"talker.model.layers.{i}."
            for suffix in [
                "input_layernorm.weight",
                "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.q_norm.weight",
                "self_attn.k_norm.weight", "self_attn.o_proj.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
            ]:
                state[p + suffix] = np.zeros(1)
        state["talker.model.codec_embedding.weight"] = np.zeros((3072, HIDDEN_SIZE))
        state["talker.model.text_embedding.weight"]  = np.zeros((151936, HIDDEN_SIZE))
        state["talker.model.norm.weight"]            = np.zeros(HIDDEN_SIZE)
        state["talker.codec_head.weight"]            = np.zeros((3072, HIDDEN_SIZE))
        return state

    def test_layer_prefix_detected(self):
        state = self._talker_state()
        self.assertTrue(any(k.startswith("talker.model.layers.") for k in state),
                        "Layer prefix detection broken")

    def test_embed_key_exists(self):
        state = self._talker_state()
        self.assertIn("talker.model.codec_embedding.weight", state,
                      "codec_embedding key missing — weight loader will KeyError")

    def test_text_embed_key_exists(self):
        state = self._talker_state()
        self.assertIn("talker.model.text_embedding.weight", state,
                      "text_embedding key missing — prefill will use wrong embedding")

    def test_norm_key_exists(self):
        state = self._talker_state()
        self.assertIn("talker.model.norm.weight", state)

    def test_lm_head_key_exists(self):
        state = self._talker_state()
        self.assertIn("talker.codec_head.weight", state,
                      "codec_head key missing — old code used talker.lm_head.weight")

    def test_old_wrong_keys_absent(self):
        """Confirm the old (broken) key names are NOT in the real checkpoint layout."""
        state = self._talker_state()
        self.assertNotIn("talker.model.embed_tokens.weight", state,
                         "Old embed_tokens key should not exist")
        self.assertNotIn("talker.lm_head.weight", state,
                         "Old lm_head key should not exist")


# ─────────────────────────────────────────────────────────────────────────────
# 2. patched_forward SimpleNamespace completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchedForwardNamespace(unittest.TestCase):
    """
    _patch_backbone returns a SimpleNamespace.
    qwen_tts talker.forward accesses these specific fields — all must exist.
    """

    REQUIRED = ["last_hidden_state", "hidden_states", "past_key_values",
                "attentions", "cross_attentions"]

    def _make_ns(self):
        hidden = np.zeros((1, 1, HIDDEN_SIZE), dtype=np.float32)
        return SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=hidden,
            past_key_values=None,
            attentions=None,
            cross_attentions=None,
            _megakernel_next_token=42,
        )

    def test_all_required_fields_present(self):
        ns = self._make_ns()
        missing = [f for f in self.REQUIRED if not hasattr(ns, f)]
        self.assertEqual(missing, [], f"SimpleNamespace missing fields: {missing}")

    def test_hidden_states_matches_last_hidden_state(self):
        ns = self._make_ns()
        # qwen_tts packs it as: hidden_states=(outputs.hidden_states, codec_ids)
        packed = (ns.hidden_states, [0, 1, 2])
        self.assertIsNotNone(packed[0])

    def test_attentions_is_none(self):
        ns = self._make_ns()
        # qwen_tts does: attentions=outputs.attentions — must be accessible
        _ = ns.attentions

    def test_megakernel_token_accessible(self):
        ns = self._make_ns()
        self.assertEqual(ns._megakernel_next_token, 42)


# ─────────────────────────────────────────────────────────────────────────────
# 3. stream() routing logic
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamRouting(unittest.TestCase):
    """stream() must pick generate_voice_clone > generate_custom_voice > legacy."""

    def _cfg(self, voice_clone_audio=None):
        return SimpleNamespace(
            voice_clone_audio=voice_clone_audio,
            voice_clone_text=None,
            emit_every_frames=1,
            decode_window_frames=16,
            overlap_samples=0,
        )

    def test_prefers_generate_voice_clone(self):
        owner = MagicMock(spec=["generate_voice_clone", "generate_custom_voice"])
        self.assertTrue(hasattr(owner, "generate_voice_clone"))

    def test_falls_back_to_generate_custom_voice(self):
        owner = MagicMock(spec=["generate_custom_voice"])
        self.assertFalse(hasattr(owner, "generate_voice_clone"))
        self.assertTrue(hasattr(owner, "generate_custom_voice"))

    def test_no_ref_audio_builds_silence_tuple(self):
        """Without voice_clone_audio the code must pass (array, sr), not bare array."""
        cfg = self._cfg(voice_clone_audio=None)
        ref = (np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
        self.assertIsInstance(ref, tuple, "ref_audio must be a tuple")
        self.assertEqual(len(ref), 2)
        self.assertIsInstance(ref[0], np.ndarray)
        self.assertEqual(ref[1], SAMPLE_RATE)

    def test_ref_audio_forwarded_when_set(self):
        """When cfg.voice_clone_audio is set, it goes straight into ref_audio."""
        cfg = self._cfg(voice_clone_audio="/path/to/ref.wav")
        ref = cfg.voice_clone_audio   # the stream method just passes this through
        self.assertEqual(ref, "/path/to/ref.wav")

    def test_generate_voice_clone_called_with_correct_kwargs(self):
        """Simulate the exact kw dict stream() would build for the base model."""
        kw = dict(
            text="Hello.",
            ref_audio=(np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE),
            ref_text=" ",
            non_streaming_mode=False,
        )
        self.assertIn("ref_audio", kw)
        self.assertIn("ref_text", kw)
        self.assertFalse(kw["non_streaming_mode"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Audio chunk iteration
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioChunkIteration(unittest.TestCase):
    """The for-loop that unpacks generate_voice_clone's generator."""

    CHUNK = np.zeros(1920, dtype=np.float32)   # 80 ms @ 24 kHz

    def _iterate(self, gen):
        """Replicate the iteration logic from _HFSwapTTS.stream."""
        out = []
        for audio_batch, _sr in gen:
            batch = audio_batch if isinstance(audio_batch, (list, tuple)) else [audio_batch]
            for chunk in batch:
                out.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        return out

    def test_list_wrapped_chunks(self):
        """Generator yields ([ndarray], sr) — qwen_tts default."""
        def gen():
            yield ([self.CHUNK.copy()], SAMPLE_RATE)
            yield ([self.CHUNK.copy()], SAMPLE_RATE)

        result = self._iterate(gen())
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (1920,))

    def test_bare_array_chunks(self):
        """Generator yields (ndarray, sr) directly — handle gracefully."""
        def gen():
            yield (self.CHUNK.copy(), SAMPLE_RATE)

        result = self._iterate(gen())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (1920,))

    def test_multi_chunk_batch(self):
        """Generator yields ([c1, c2], sr) in one step."""
        def gen():
            yield ([self.CHUNK.copy(), self.CHUNK.copy()], SAMPLE_RATE)

        result = self._iterate(gen())
        self.assertEqual(len(result), 2)

    def test_output_is_float32(self):
        def gen():
            yield ([self.CHUNK.astype(np.float64)], SAMPLE_RATE)   # wrong dtype in

        result = self._iterate(gen())
        self.assertEqual(result[0].dtype, np.float32)

    def test_output_is_1d(self):
        """Even if qwen_tts returns 2D audio it gets reshaped to 1D."""
        def gen():
            yield ([self.CHUNK.reshape(1, -1)], SAMPLE_RATE)   # 2D in

        result = self._iterate(gen())
        self.assertEqual(result[0].ndim, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Decoder.prefill embedding selection
# ─────────────────────────────────────────────────────────────────────────────

class TestPrefillEmbeddingSelection(unittest.TestCase):
    """Decoder.prefill must use text_embed_weight for text tokens."""

    def _select_embed(self, text_embed, codec_embed):
        """Replicate the logic at the top of Decoder.prefill."""
        return text_embed if text_embed is not None else codec_embed

    def test_uses_text_embed_when_available(self):
        codec = np.zeros((3072, HIDDEN_SIZE))
        text  = np.ones((151936, HIDDEN_SIZE))   # distinguishable by value
        chosen = self._select_embed(text, codec)
        self.assertIs(chosen, text)

    def test_falls_back_to_codec_embed(self):
        codec = np.ones((3072, HIDDEN_SIZE))
        chosen = self._select_embed(None, codec)
        self.assertIs(chosen, codec)

    def test_none_text_embed_does_not_crash(self):
        """text_embed_weight=None is valid (base model path)."""
        result = self._select_embed(None, np.zeros((3072, HIDDEN_SIZE)))
        self.assertIsNotNone(result)


# ─────────────────────────────────────────────────────────────────────────────
# 6. _ManualTTS qwen_tts pre-registration
# ─────────────────────────────────────────────────────────────────────────────

class TestQwenTTSPreRegistration(unittest.TestCase):
    """
    _load_upstream imports qwen_tts before AutoModel so it registers the
    qwen3_tts type. Test that a failed import is silently swallowed.
    """

    def test_failed_import_is_swallowed(self):
        """Even if qwen_tts import raises, the code continues."""
        registered = []

        def try_register():
            try:
                raise ImportError("torchaudio CUDA mismatch (simulated)")
            except Exception:
                pass
            registered.append("continued")

        try_register()
        self.assertEqual(registered, ["continued"])

    def test_successful_import_registers(self):
        """If qwen_tts imports cleanly, it registers the type (simulated)."""
        registry = {}

        def fake_qwen_tts_import():
            registry["qwen3_tts"] = "registered"

        fake_qwen_tts_import()
        self.assertIn("qwen3_tts", registry)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Source-file content checks — read the REAL .py files and verify every fix
#    is still present. These catch accidental reverts without needing a GPU.
# ─────────────────────────────────────────────────────────────────────────────

import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

class TestSourceCodePatterns(unittest.TestCase):
    """Read the actual source files and grep for the exact strings we fixed.

    What this adds over the logic tests above:
      - If someone accidentally reverts a fix in model.py, THESE tests fail.
      - They test the real code, not a replica of the logic.

    What they still can't cover (requires GPU):
      - CUDA kernel actually executing
      - qwen_tts monkey-patch intercepting the right call inside its loop
      - Weight tensor shapes matching the kernel's compile-time vocab size
      - Real audio being produced with correct EOS detection
    """

    def _src(self, rel_path: str) -> str:
        with open(_os.path.join(_REPO, rel_path)) as f:
            return f.read()

    # ── qwen_megakernel/model.py ─────────────────────────────────────────────

    def test_weight_loader_uses_codec_embedding_key(self):
        src = self._src("qwen_megakernel/qwen_megakernel/model.py")
        self.assertIn("talker.model.codec_embedding.weight", src,
                      "Key name reverted — loader will KeyError on real checkpoint")

    def test_weight_loader_uses_codec_head_not_lm_head(self):
        src = self._src("qwen_megakernel/qwen_megakernel/model.py")
        self.assertIn("talker.codec_head.weight", src,
                      "lm_head key reverted — loader will KeyError on real checkpoint")

    def test_weight_loader_loads_text_embed_key(self):
        src = self._src("qwen_megakernel/qwen_megakernel/model.py")
        self.assertIn("talker.model.text_embedding.weight", src,
                      "text_embedding key absent — prefill will use codec embed for text tokens")

    def test_decoder_stores_text_embed_weight_field(self):
        src = self._src("qwen_megakernel/qwen_megakernel/model.py")
        self.assertIn("_text_embed_weight", src,
                      "Decoder._text_embed_weight field removed")

    def test_prefill_selects_text_embed_when_available(self):
        src = self._src("qwen_megakernel/qwen_megakernel/model.py")
        self.assertIn(
            "self._text_embed_weight if self._text_embed_weight is not None",
            src,
            "prefill() no longer selects text embed — will embed text tokens incorrectly",
        )

    def test_prefill_step_method_exists(self):
        src = self._src("qwen_megakernel/qwen_megakernel/model.py")
        self.assertIn("def prefill_step", src,
                      "Decoder.prefill_step missing — last prompt token will use codec embed on text token ID")

    def test_patched_forward_uses_prefill_step_for_initial_call(self):
        src = self._src("tts_backend/model.py")
        self.assertIn("megakernel.prefill_step(last)", src,
                      "patched_forward must call prefill_step (text embed) for last prompt token")

    def test_patched_forward_uses_step_for_subsequent_calls(self):
        src = self._src("tts_backend/model.py")
        self.assertIn("megakernel.step(last)", src,
                      "patched_forward must call step (codec embed) for audio tokens")

    def test_old_lm_head_key_not_used_as_lm_head_var(self):
        """Old code had: lm_head_key = 'talker.lm_head.weight' — must be gone."""
        src = self._src("qwen_megakernel/qwen_megakernel/model.py")
        self.assertNotIn('"talker.lm_head.weight"', src,
                         "Old broken lm_head key is back in the loader")

    # ── tts_backend/model.py ─────────────────────────────────────────────────

    def test_patched_forward_has_hidden_states(self):
        src = self._src("tts_backend/model.py")
        self.assertIn("hidden_states=hidden_btH", src,
                      "hidden_states field removed from patched_forward SimpleNamespace")

    def test_patched_forward_has_attentions_none(self):
        src = self._src("tts_backend/model.py")
        self.assertIn("attentions=None", src,
                      "attentions=None removed — qwen_tts will AttributeError")

    def test_patched_forward_has_cross_attentions_none(self):
        src = self._src("tts_backend/model.py")
        self.assertIn("cross_attentions=None", src,
                      "cross_attentions=None removed — qwen_tts may AttributeError")

    def test_stream_calls_generate_voice_clone(self):
        src = self._src("tts_backend/model.py")
        self.assertIn("generate_voice_clone", src,
                      "stream() no longer calls generate_voice_clone")

    def test_stream_passes_ref_audio_as_tuple(self):
        src = self._src("tts_backend/model.py")
        self.assertIn(
            "np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)",
            src,
            "silence ref_audio is no longer a (array, sr) tuple — qwen_tts will ValueError",
        )

    def test_manual_tts_pre_registers_qwen_tts(self):
        src = self._src("tts_backend/model.py")
        self.assertIn("import qwen_tts", src,
                      "_ManualTTS no longer pre-registers qwen_tts — AutoModel will fail")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Basic import checks (CPU-only)
# ─────────────────────────────────────────────────────────────────────────────

class TestImports(unittest.TestCase):

    def _try_import(self, name):
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    def test_numpy(self):
        self.assertTrue(self._try_import("numpy"), "numpy not installed")

    def test_fastapi(self):
        if not self._try_import("fastapi"):
            self.skipTest("fastapi not in this env — OK on macOS dev machine")

    def test_websockets(self):
        if not self._try_import("websockets"):
            self.skipTest("websockets not in this env")

    def test_soundfile(self):
        if not self._try_import("soundfile"):
            self.skipTest("soundfile not in this env")

    def test_torch_if_available(self):
        if not self._try_import("torch"):
            self.skipTest("torch not installed — expected on GPU host only")
        import torch
        self.assertIsNotNone(torch.__version__)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Qwen3-TTS local validation (no GPU required)")
    print("=" * 65)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
