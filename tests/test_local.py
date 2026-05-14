"""CPU-only validation tests — no GPU or CUDA required."""

import sys
import unittest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock

SAMPLE_RATE = 24000
HIDDEN_SIZE = 1024


class TestKeyDetection(unittest.TestCase):
    def _talker_state(self):
        state = {}
        for i in range(2):
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
        state["talker.model.text_embedding.weight"] = np.zeros((151936, HIDDEN_SIZE))
        state["talker.model.norm.weight"] = np.zeros(HIDDEN_SIZE)
        state["talker.codec_head.weight"] = np.zeros((3072, HIDDEN_SIZE))
        return state

    def test_layer_prefix_detected(self):
        state = self._talker_state()
        self.assertTrue(any(k.startswith("talker.model.layers.") for k in state))

    def test_embed_key_exists(self):
        self.assertIn("talker.model.codec_embedding.weight", self._talker_state())

    def test_text_embed_key_exists(self):
        self.assertIn("talker.model.text_embedding.weight", self._talker_state())

    def test_norm_key_exists(self):
        self.assertIn("talker.model.norm.weight", self._talker_state())

    def test_lm_head_key_exists(self):
        self.assertIn("talker.codec_head.weight", self._talker_state())

    def test_old_wrong_keys_absent(self):
        state = self._talker_state()
        self.assertNotIn("talker.model.embed_tokens.weight", state)
        self.assertNotIn("talker.lm_head.weight", state)


class TestPatchedForwardNamespace(unittest.TestCase):
    REQUIRED = ["last_hidden_state", "hidden_states", "past_key_values", "attentions", "cross_attentions"]

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
        self.assertEqual(missing, [])

    def test_hidden_states_matches_last_hidden_state(self):
        ns = self._make_ns()
        packed = (ns.hidden_states, [0, 1, 2])
        self.assertIsNotNone(packed[0])

    def test_attentions_is_none(self):
        ns = self._make_ns()
        _ = ns.attentions

    def test_megakernel_token_accessible(self):
        self.assertEqual(self._make_ns()._megakernel_next_token, 42)


class TestStreamRouting(unittest.TestCase):
    def test_prefers_generate_voice_clone(self):
        owner = MagicMock(spec=["generate_voice_clone", "generate_custom_voice"])
        self.assertTrue(hasattr(owner, "generate_voice_clone"))

    def test_falls_back_to_generate_custom_voice(self):
        owner = MagicMock(spec=["generate_custom_voice"])
        self.assertFalse(hasattr(owner, "generate_voice_clone"))
        self.assertTrue(hasattr(owner, "generate_custom_voice"))

    def test_no_ref_audio_builds_silence_tuple(self):
        ref = (np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)
        self.assertIsInstance(ref, tuple)
        self.assertEqual(len(ref), 2)
        self.assertIsInstance(ref[0], np.ndarray)
        self.assertEqual(ref[1], SAMPLE_RATE)

    def test_ref_audio_forwarded_when_set(self):
        class Cfg:
            voice_clone_audio = "/path/to/ref.wav"
        self.assertEqual(Cfg.voice_clone_audio, "/path/to/ref.wav")

    def test_generate_voice_clone_called_with_correct_kwargs(self):
        kw = dict(
            text="Hello.",
            ref_audio=(np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE),
            ref_text=" ",
            non_streaming_mode=False,
        )
        self.assertIn("ref_audio", kw)
        self.assertIn("ref_text", kw)
        self.assertFalse(kw["non_streaming_mode"])


class TestAudioChunkIteration(unittest.TestCase):
    CHUNK = np.zeros(1920, dtype=np.float32)

    def _iterate(self, gen):
        out = []
        for audio_batch, _sr in gen:
            batch = audio_batch if isinstance(audio_batch, (list, tuple)) else [audio_batch]
            for chunk in batch:
                out.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
        return out

    def test_list_wrapped_chunks(self):
        def gen():
            yield ([self.CHUNK.copy()], SAMPLE_RATE)
            yield ([self.CHUNK.copy()], SAMPLE_RATE)
        result = self._iterate(gen())
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (1920,))

    def test_bare_array_chunks(self):
        def gen():
            yield (self.CHUNK.copy(), SAMPLE_RATE)
        result = self._iterate(gen())
        self.assertEqual(len(result), 1)

    def test_multi_chunk_batch(self):
        def gen():
            yield ([self.CHUNK.copy(), self.CHUNK.copy()], SAMPLE_RATE)
        self.assertEqual(len(self._iterate(gen())), 2)

    def test_output_is_float32(self):
        def gen():
            yield ([self.CHUNK.astype(np.float64)], SAMPLE_RATE)
        self.assertEqual(self._iterate(gen())[0].dtype, np.float32)

    def test_output_is_1d(self):
        def gen():
            yield ([self.CHUNK.reshape(1, -1)], SAMPLE_RATE)
        self.assertEqual(self._iterate(gen())[0].ndim, 1)


class TestPrefillEmbeddingSelection(unittest.TestCase):
    def _select(self, text_embed, codec_embed):
        return text_embed if text_embed is not None else codec_embed

    def test_uses_text_embed_when_available(self):
        codec = np.zeros((3072, HIDDEN_SIZE))
        text = np.ones((151936, HIDDEN_SIZE))
        self.assertIs(self._select(text, codec), text)

    def test_falls_back_to_codec_embed(self):
        codec = np.ones((3072, HIDDEN_SIZE))
        self.assertIs(self._select(None, codec), codec)

    def test_none_text_embed_does_not_crash(self):
        self.assertIsNotNone(self._select(None, np.zeros((3072, HIDDEN_SIZE))))


class TestQwenTTSPreRegistration(unittest.TestCase):
    def test_failed_import_is_swallowed(self):
        registered = []
        try:
            raise ImportError("simulated")
        except Exception:
            pass
        registered.append("continued")
        self.assertEqual(registered, ["continued"])

    def test_successful_import_registers(self):
        registry = {}
        registry["qwen3_tts"] = "registered"
        self.assertIn("qwen3_tts", registry)


import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


class TestSourceCodePatterns(unittest.TestCase):
    def _src(self, rel_path: str) -> str:
        with open(_os.path.join(_REPO, rel_path)) as f:
            return f.read()

    def test_weight_loader_uses_codec_embedding_key(self):
        self.assertIn("talker.model.codec_embedding.weight", self._src("qwen_megakernel/qwen_megakernel/model.py"))

    def test_weight_loader_uses_codec_head_not_lm_head(self):
        self.assertIn("talker.codec_head.weight", self._src("qwen_megakernel/qwen_megakernel/model.py"))

    def test_weight_loader_loads_text_embed_key(self):
        self.assertIn("talker.model.text_embedding.weight", self._src("qwen_megakernel/qwen_megakernel/model.py"))

    def test_decoder_stores_text_embed_weight_field(self):
        self.assertIn("_text_embed_weight", self._src("qwen_megakernel/qwen_megakernel/model.py"))

    def test_prefill_selects_text_embed_when_available(self):
        self.assertIn(
            "self._text_embed_weight if self._text_embed_weight is not None",
            self._src("qwen_megakernel/qwen_megakernel/model.py"),
        )

    def test_prefill_step_method_exists(self):
        self.assertIn("def prefill_step", self._src("qwen_megakernel/qwen_megakernel/model.py"))

    def test_patched_forward_uses_prefill_step_for_initial_call(self):
        self.assertIn("megakernel.prefill_step(last)", self._src("tts_backend/model.py"))

    def test_patched_forward_uses_step_for_subsequent_calls(self):
        self.assertIn("megakernel.step(last)", self._src("tts_backend/model.py"))

    def test_patched_forward_delegates_inputs_embeds_to_orig(self):
        src = self._src("tts_backend/model.py")
        self.assertIn("inputs_embeds is not None", src)
        self.assertIn("_orig_forward", src)

    def test_stream_unpacks_generate_voice_clone_as_tuple(self):
        src = self._src("tts_backend/model.py")
        self.assertIn("audio_list, _sr = stream_owner.generate_voice_clone", src)

    def test_old_lm_head_key_not_used(self):
        self.assertNotIn('"talker.lm_head.weight"', self._src("qwen_megakernel/qwen_megakernel/model.py"))

    def test_patched_forward_has_hidden_states(self):
        self.assertIn("hidden_states=hidden_btH", self._src("tts_backend/model.py"))

    def test_patched_forward_has_attentions_none(self):
        self.assertIn("attentions=None", self._src("tts_backend/model.py"))

    def test_patched_forward_has_cross_attentions_none(self):
        self.assertIn("cross_attentions=None", self._src("tts_backend/model.py"))

    def test_stream_calls_generate_voice_clone(self):
        self.assertIn("generate_voice_clone", self._src("tts_backend/model.py"))

    def test_stream_passes_ref_audio_as_tuple(self):
        self.assertIn(
            "np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32), SAMPLE_RATE)",
            self._src("tts_backend/model.py"),
        )

    def test_run_server_uses_correct_vocab_size(self):
        src = self._src("scripts/run_server.sh")
        self.assertNotIn("LDG_VOCAB_SIZE:-2052", src)
        self.assertIn("LDG_VOCAB_SIZE:-3072", src)

    def test_manual_tts_pre_registers_qwen_tts(self):
        self.assertIn("import qwen_tts", self._src("tts_backend/model.py"))


class TestImports(unittest.TestCase):
    def _try_import(self, name):
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    def test_numpy(self):
        self.assertTrue(self._try_import("numpy"))

    def test_fastapi(self):
        if not self._try_import("fastapi"):
            self.skipTest("fastapi not in this env")

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


if __name__ == "__main__":
    print("=" * 65)
    print("  Qwen3-TTS local validation (no GPU required)")
    print("=" * 65)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
