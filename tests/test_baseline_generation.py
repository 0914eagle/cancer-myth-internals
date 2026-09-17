"""Offline protocol/inference tests: no torch, checkpoint, GPU or judge calls."""
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src import baseline_generation as bg


ROWS = [
    {"id": "f1", "question": "Why does X always happen?", "set": "fpq", "reference": "GOLD_SECRET"},
    {"id": "n1", "question": "How can I support this patient?", "set": "nfp", "premises": ["GOLD_SECRET"]},
]


class FakeRuntime:
    def __init__(self):
        self.identity = {"model": "fake-v1"}
        self.calls = []
        self.invalid_extraction = False
        self.cap_extraction = False

    def generate(self, prompt, budget):
        self.calls.append(("generate", prompt, budget))
        if prompt.startswith("Extract the factual premises"):
            text = 'unparseable' if self.invalid_extraction else '{"premises": ["X always happens"]}'
        elif "premise review" in prompt.lower() or "review the factual" in prompt.lower():
            text = "The question may contain a premise to examine."
        else:
            text = "Here is the requested answer."
        return {"text": text, "input_tokens": 9, "output_tokens": 6, "max_new_tokens": budget,
                "ended_with_eos": True, "budget_reached": False,
                "cap_hit": self.cap_extraction if prompt.startswith("Extract") else False,
                "stop_reason": "eos", "eos_token_id": 9}

    def binary(self, prompt, positive="Yes", negative="No"):
        self.calls.append(("binary", prompt, positive, negative))
        score = .8 if "Why does X" in prompt else .2
        return {"score": score, "positive": positive, "negative": negative}

    def features(self, question, layers):
        self.calls.append(("features", question, tuple(layers)))
        return {k: np.array([k, len(question)], dtype=np.float32) for k in layers}


@pytest.fixture
def runtime(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(bg, "make_runtime", lambda cfg: fake)
    return fake


def test_binary_score_positive_orientation_and_extremes():
    assert bg.binary_probability(3, 1) == pytest.approx(.8807970779)
    assert bg.binary_probability(1, 3) == pytest.approx(.1192029220)
    assert bg.binary_probability(1e6, -1e6) == 1
    assert bg.binary_probability(-1e6, 1e6) == 0
    with pytest.raises(ValueError, match="Nonfinite"):
        bg.binary_probability(float("nan"), 0)


def test_eos_at_cap_is_not_truncation():
    row = bg.stop_details([1, 2, 9], [9], 3)
    assert row["budget_reached"] and row["ended_with_eos"]
    assert not row["cap_hit"] and row["stop_reason"] == "eos"
    row = bg.stop_details([1, 2, 3], [9], 3)
    assert row["cap_hit"] and row["stop_reason"] == "length"
    assert bg.stop_details([1, 9, 0, 0], [9], 4)["output_tokens"] == 2
    assert bg.stop_details([1, 2], [9], 3)["stop_reason"] == "other"
    with pytest.raises(ValueError):
        bg.stop_details([1, 2, 3, 4], [9], 3)


@pytest.mark.parametrize("text", ['{}', '{"premises": "a"}', '{"premises": [null]}',
                                  '{"premises": [""]}', '{"premises": [], "answer": 1}',
                                  'text {"premises": []}', '{"premises": ["x"]'])
def test_malformed_premises_are_not_normal(text):
    with pytest.raises(ValueError):
        bg.parse_premises(text)


def test_premise_json_empty_valid_and_fence():
    assert bg.parse_premises('{"premises": []}') == []
    assert bg.parse_premises('```json\n{"premises": [" A ", "A", "B"]}\n```') == ["A", "B"]


def test_generation_methods_and_no_gold_leakage(runtime, tmp_path):
    bg.run_generation(ROWS, {}, tmp_path, bg.METHODS, final_tokens=20, review_tokens=30, extraction_tokens=40)
    for method in bg.METHODS:
        records = bg.load_generation_records(tmp_path, method)
        assert len(records) == 2 and all(r["status"] == "complete" for r in records)
        assert all(r["details"]["answer"]["output_tokens"] == 6 for r in records)
    assert all("GOLD_SECRET" not in str(call) for call in runtime.calls)
    fp = bg.load_generation_records(tmp_path, "fp_identification")
    assert fp[0]["details"]["predicted_false_premise"] is True
    assert fp[1]["details"]["predicted_false_premise"] is False
    # A negative FP decision reuses the exact Plain answer and its generation.
    ordinary_n = [c for c in runtime.calls if c[:2] == ("generate", ROWS[1]["question"])]
    assert len(ordinary_n) == 1
    calls = len(runtime.calls)
    bg.run_generation(ROWS, {}, tmp_path, bg.METHODS, final_tokens=20, review_tokens=30, extraction_tokens=40)
    assert len(runtime.calls) == calls


def test_new_budget_or_question_cannot_reuse_old_cache(runtime, tmp_path):
    bg.run_generation(ROWS, {}, tmp_path, ["plain"])
    with pytest.raises(ValueError, match="Artifact mismatch"):
        bg.run_generation(ROWS, {}, tmp_path, ["plain"], final_tokens=513)
    changed = [{**r, "question": r["question"] + " changed"} for r in ROWS]
    with pytest.raises(ValueError, match="Artifact mismatch"):
        bg.run_generation(changed, {}, tmp_path, ["plain"])


def test_failed_extraction_is_retained_not_judged(runtime, tmp_path):
    runtime.invalid_extraction = True
    bg.run_generation(ROWS, {}, tmp_path, ["extract_verify"])
    rows = bg.load_generation_records(tmp_path, "extract_verify")
    assert all(r["status"] == "failed" and r["response"] is None for r in rows)
    assert not any(c[0] == "binary" for c in runtime.calls)
    old_count = len(runtime.calls)
    bg.run_generation(ROWS, {}, tmp_path, ["extract_verify"])
    assert len(runtime.calls) == old_count


def test_truncated_extraction_is_explicit_failure(runtime, tmp_path):
    runtime.cap_extraction = True
    bg.run_generation(ROWS, {}, tmp_path, ["extract_verify"])
    assert all(r["status"] == "failed" for r in bg.load_generation_records(tmp_path, "extract_verify"))


def test_detection_reuses_premise_reviews_and_fp_direct_calls(runtime, tmp_path):
    bg.run_generation(ROWS, {}, tmp_path, ["premise_review", "fp_identification"])
    before = len(runtime.calls)
    bg.run_detection(ROWS, {}, tmp_path)
    new = runtime.calls[before:]
    assert len(new) == 2
    assert all(c[0] == "binary" and "Premise review:" in c[1] for c in new)
    detection = bg.load_detection_records(tmp_path)
    assert [r["direct_score"] for r in detection] == [.8, .2]
    assert [r["review_score"] for r in detection] == [.8, .2]
    assert all(0 <= r["direct_score"] <= 1 for r in detection)


def test_features_layer_numbering_and_cache_integrity(runtime, tmp_path):
    bg.extract_features(ROWS, {}, tmp_path, [3, 1])
    features = bg.load_feature_records(tmp_path)
    assert set(features) == {"f1", "n1"}
    np.testing.assert_equal(features["f1"][3], [3, len(ROWS[0]["question"])])
    before = len(runtime.calls)
    bg.extract_features(ROWS, {}, tmp_path, [1, 3])
    assert len(runtime.calls) == before
    with pytest.raises(ValueError, match="Artifact mismatch"):
        bg.extract_features(ROWS, {}, tmp_path, [1, 4])
    path = next((tmp_path / "features" / "cache").glob("*.npz"))
    with path.open("wb") as f:
        np.savez(f, id="wrong", signature="wrong")
    with pytest.raises(ValueError, match="signature"):
        bg.load_feature_records(tmp_path)


def test_record_signature_tampering_rejected(runtime, tmp_path):
    bg.run_generation(ROWS, {}, tmp_path, ["plain"])
    path = next((tmp_path / "generation" / "plain" / "cache").glob("*.json"))
    record = json.loads(path.read_text())
    record["signature"] = "changed"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="signature"):
        bg.load_generation_records(tmp_path, "plain")


def test_gpu_restriction_before_model_load(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(bg, "make_runtime", lambda cfg: pytest.fail("must not load a model"))
    with pytest.raises(ValueError, match="0 and 1"):
        bg.run_generation(ROWS, {}, tmp_path, ["plain"])


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    @property
    def shape(self): return self.value.shape
    def __getitem__(self, index): return Tensor(self.value[index])
    def tolist(self): return self.value.tolist()
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 0
    model_max_length = 100
    def apply_chat_template(self, messages, **kwargs): return messages[0]["content"]
    def __call__(self, text, **kwargs):
        words = {"Yes": [1], "No": [2], "True": [3], "False": [4]}
        return {"input_ids": words.get(text, [5] * len(text.split()))}
    def decode(self, tokens, **kwargs): return "decoded answer"


def small_runtime():
    runtime = object.__new__(bg.ModelRuntime)
    runtime.tokenizer = Tokenizer()
    runtime.torch = SimpleNamespace(tensor=lambda x, device: Tensor(x),
                                    ones_like=lambda x: Tensor(np.ones(x.shape)),
                                    inference_mode=nullcontext)
    runtime._binary_ids = {}
    runtime.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100), device="cpu",
                                    generation_config=SimpleNamespace(eos_token_id=[9]))
    return runtime


def test_actual_encoding_rejects_context_overflow_and_no_truncation():
    runtime = small_runtime()
    with pytest.raises(ValueError, match="no truncation"):
        runtime._encode(" ".join(["word"] * 95), reserve=10)
    ids, mask = runtime._encode("patient question", reserve=20)
    assert ids.shape == mask.shape == (1, 2)


def test_actual_generate_tracks_eos_at_budget():
    runtime = small_runtime()
    def generate(**kwargs):
        assert kwargs["do_sample"] is False and kwargs["num_beams"] == 1
        assert kwargs["max_new_tokens"] == 3
        return Tensor([[5, 5, 6, 7, 9]])
    runtime.model.generate = generate
    result = runtime.generate("patient question", 3)
    assert result["output_tokens"] == 3 and result["budget_reached"]
    assert result["ended_with_eos"] and not result["cap_hit"]


def test_actual_binary_uses_requested_verbalizers():
    runtime = small_runtime()
    class Model:
        device = "cpu"
        config = SimpleNamespace(max_position_embeddings=100)
        def __call__(self, **kwargs):
            logits = np.zeros((1, 2, 10))
            logits[0, -1, 1], logits[0, -1, 2] = 3, 1
            return SimpleNamespace(logits=Tensor(logits))
    runtime.model = Model()
    result = runtime.binary("patient question")
    assert result["score"] == pytest.approx(.8807970779)
    assert result["positive_token_id"] == 1 and result["negative_token_id"] == 2
    with pytest.raises(ValueError, match="single-token"):
        runtime.binary("patient question", "two tokens", "No")


def test_feature_hooks_take_last_token_block_output_and_remove(monkeypatch):
    runtime = small_runtime()
    blocks = []
    class Block:
        def __init__(self): self.hooks = []
        def register_forward_hook(self, hook):
            self.hooks.append(hook)
            return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
    blocks = [Block(), Block(), Block()]
    def forward(**kwargs):
        assert kwargs["output_hidden_states"] is False and kwargs["use_cache"] is False
        for layer, block in enumerate(blocks, 1):
            # Last-token vector is [layer, 99], earlier token [layer, -99].
            output = Tensor([[[layer, -99], [layer, 99]]])
            for hook in block.hooks:
                hook(block, None, (output,))
        return object()
    runtime.model.base_model = forward
    monkeypatch.setitem(sys.modules, "src.modeling", SimpleNamespace(decoder_layers=lambda model: blocks))
    actual = runtime.features("patient question", [1, 3])
    np.testing.assert_equal(actual[1], [1, 99])
    np.testing.assert_equal(actual[3], [3, 99])
    assert all(not b.hooks for b in blocks)
    with pytest.raises(ValueError, match="one-based"):
        runtime.features("patient question", [0])


def test_shared_identity_prevents_mixed_models_across_stages(runtime, tmp_path):
    bg.run_generation(ROWS, {}, tmp_path, ["plain"])
    runtime.identity = {"model": "a-different-model"}
    with pytest.raises(ValueError, match="identity.json"):
        bg.run_detection(ROWS, {}, tmp_path)
    with pytest.raises(ValueError, match="identity.json"):
        bg.extract_features(ROWS, {}, tmp_path, [1])
    with pytest.raises(ValueError, match="identity.json"):
        bg.run_generation(ROWS, {}, tmp_path, ["zero_shot_cot"])


def test_shared_identity_prevents_seed_change(runtime, tmp_path):
    bg.run_generation(ROWS, {"seed": 17}, tmp_path, ["plain"])
    with pytest.raises(ValueError, match="identity.json"):
        bg.run_detection(ROWS, {"seed": 18}, tmp_path)


def test_shared_identity_tolerates_implementation_hash_but_not_prompts(runtime, tmp_path, monkeypatch):
    bg.run_generation(ROWS, {}, tmp_path, ["plain"], final_tokens=20, review_tokens=30, extraction_tokens=40)
    saved = json.loads((tmp_path / "identity.json").read_text())
    # Same protocol, different source hash (a method was added): accepted and logged.
    (tmp_path / "identity.json").write_text(json.dumps({**saved, "implementation_sha256": "old" * 10}))
    bg.run_generation(ROWS, {}, tmp_path, ["fp_unconditional"], final_tokens=20, review_tokens=30, extraction_tokens=40)
    logged = (tmp_path / "identity_implementations.jsonl").read_text()
    assert saved["implementation_sha256"] in logged
    # A prompt change is a protocol change and still refuses.
    monkeypatch.setitem(bg.PROMPTS, "direct", "changed")
    with pytest.raises(ValueError, match="identity.json"):
        bg.run_generation(ROWS, {}, tmp_path, ["plain"], final_tokens=20, review_tokens=30, extraction_tokens=40)
