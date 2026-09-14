from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.overnight_c_gpu import DiagnosticsRuntime, _Intervention


class Tokenizer:
    chat_template = "toy-char-template"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "U:" + messages[0]["content"] + "|A:"
        if len(messages) == 2:
            text += messages[1]["content"] + "~"
        return text

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) % 64 for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + i % 26) for i in ids)


class Block(torch.nn.Module):
    def forward(self, h):
        return (h,)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(31)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block(), Block(), Block()])
        self.emb = torch.nn.Embedding(64, 4)
        self.head = torch.nn.Linear(4, 64, bias=False)
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(_commit_hash="toy", hidden_size=4)
        self.generation_config = SimpleNamespace(eos_token_id=63)
        self.forward_calls = 0
        self.last_generate_ids = None

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        self.forward_calls += 1
        h = self.emb(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        return SimpleNamespace(logits=self.head(h))

    def generate(self, input_ids, max_new_tokens, **kwargs):
        self.last_generate_ids = input_ids.clone()
        result = input_ids
        current = input_ids
        for _ in range(max_new_tokens):
            scores = self(current).logits[:, -1]
            current = scores.argmax(-1)[:, None]
            result = torch.cat([result, current], 1)
        return result


@pytest.fixture
def runtime(tmp_path):
    path = tmp_path / "direction.npz"
    np.savez(path, c=np.array([1, 0, 0, 0], dtype=np.float32), s=np.asarray(3.0))
    rt = DiagnosticsRuntime(ToyModel(), Tokenizer(), {"source_model": {"model_id": "toy"}})
    direction = {"path": str(path), "key": "c", "scale_key": "s", "layer": 1}
    return rt, direction


def test_teacher_forcing_uses_previous_position_including_first(runtime):
    rt, direction = runtime
    task = dict(kind="dose", question="Q", positive="abc", negative="xy", direction=direction, alpha=0)
    result = rt.execute(task)
    ids, start, end = rt._answer("Q", "abc", task)
    logits = rt.model(ids).logits[0].detach().float()
    expected = torch.log_softmax(logits[start - 1:end - 1], -1).gather(1, ids[0, start:end, None]).squeeze(1)
    wrong = torch.log_softmax(logits[start:end], -1).gather(1, ids[0, start:end, None]).squeeze(1)
    assert result["positive"]["mean_logp"] == pytest.approx(float(expected.mean()), abs=1e-6)
    assert result["positive"]["mean_logp"] != pytest.approx(float(wrong.mean()), abs=1e-4)
    assert result["positive"]["token_count"] == 3
    assert result["positive"]["rest"]["mean_logp"] is None
    assert result["steering_audit"]["positive"]["modified_predictor_positions"] == 0


@pytest.mark.parametrize("policy,positions", [("all", [3, 4, 5, 6]), ("prefill", [3]), ("first_k", [3, 4])])
def test_policies_modify_predictors_and_match_cached_generation(policy, positions):
    vec = torch.tensor([1., 0.])
    original = torch.ones(1, 8, 2)
    hook = _Intervention(vec, 2, 3, 6, policy, first_k=2)
    full = hook(None, None, (original,))[0]
    modified = ((full != original).any(-1)[0]).nonzero().flatten().tolist()
    assert modified == positions
    cached_hook = _Intervention(vec, 2, 3, 6, policy, first_k=2)
    chunks = [cached_hook(None, None, (original[:, :4],))[0]]
    for i in range(4, 8):
        chunks.append(cached_hook(None, None, (original[:, i:i + 1],))[0])
    assert torch.equal(torch.cat(chunks, dim=1), full)
    assert hook.audit()["modified_predictor_positions"] == len(positions)
    assert hook.audit()["mean_actual_delta_norm"] == pytest.approx(2.)


def test_ablation_removes_axis_with_sign_symmetry():
    vec = torch.tensor([.6, .8])
    original = torch.tensor([[[2., 1.], [3., 4.]]])
    plus = _Intervention(vec, 1, 0, 1, "ablate")(None, None, original)
    minus = _Intervention(-vec, 1, 0, 1, "ablate")(None, None, original)
    assert torch.allclose(plus, minus)
    assert torch.allclose(plus @ vec, torch.zeros(1, 2), atol=1e-6)


def test_nonunit_and_badscale_directions_rejected(runtime, tmp_path):
    rt, spec = runtime
    p = tmp_path / "bad.npz"
    np.savez(p, c=np.array([2., 0, 0, 0]), s=np.asarray(1.))
    with pytest.raises(ValueError, match="unit"):
        rt._direction(dict(spec, path=str(p)))
    np.savez(p, c=np.array([1., 0, 0, 0]), s=np.asarray(float("nan")))
    with pytest.raises(ValueError, match="scale"):
        rt._direction(dict(spec, path=str(p)))
    np.savez(p, c=np.array([1., 0]), s=np.asarray(1.))
    with pytest.raises(ValueError, match="dimension"):
        rt._direction(dict(spec, path=str(p)))
    with pytest.raises(ValueError, match="interior"):
        rt._direction(dict(spec, layer=3))


def test_random_direction_reproducible_and_external_scale(runtime, tmp_path):
    rt, spec = runtime
    scale_file = tmp_path / "scale.npz"
    np.savez(scale_file, source_scale=np.asarray(7.))
    random = dict(spec, random_seed=20, scale_path=str(scale_file), scale_key="source_scale")
    a, scale = rt._direction(random)
    b, _ = rt._direction(random)
    assert torch.equal(a, b)
    assert a.norm() == pytest.approx(1., abs=1e-6)
    assert scale == 7
    assert not torch.equal(a, torch.tensor([1., 0, 0, 0]))


def test_project_uses_answer_positions_not_predictors(runtime):
    rt, spec = runtime
    task = dict(kind="project", question="Q", response="abc", direction=spec)
    result = rt.execute(task)
    ids, start, end = rt._answer("Q", "abc", task)
    h = rt.model.emb(ids)[0, start:end].detach().mean(0)
    assert result["full"]["dot"] == pytest.approx(float(h[0]))
    assert result["full"]["cosine"] == pytest.approx(float(h[0] / h.norm()))
    assert result["answer_tokens"] == 3


def test_dose_changes_probability_and_always_removes_hook(runtime):
    rt, spec = runtime
    task = dict(kind="dose", question="Q", positive="abc", negative="xy", direction=spec)
    zero, pushed = rt.execute(dict(task, alpha=0)), rt.execute(dict(task, alpha=.2))
    assert zero["positive"]["mean_logp"] != pushed["positive"]["mean_logp"]
    assert pushed["steering_audit"]["positive"]["modified_predictor_positions"] == 3
    assert not rt.layers[0]._forward_hooks
    original = rt.model.forward
    def broken(**kwargs):
        raise RuntimeError("failure")
    rt.model.forward = broken
    with pytest.raises(RuntimeError, match="failure"):
        rt.execute(dict(task, alpha=.2))
    assert not rt.layers[0]._forward_hooks
    rt.model.forward = original


def test_generate_uses_override_and_position_budget(runtime):
    rt, spec = runtime
    result = rt.execute(dict(kind="generate", question="Q", prompt="OTHER", direction=spec,
                             alpha=.1, policy="first_k", first_k=2, max_new_tokens=4))
    assert result["output_tokens"] == 4
    assert result["steering_audit"]["modified_predictor_positions"] == 2
    assert rt.model.last_generate_ids.shape[1] == len("U:OTHER|A:")
    assert result["response"]


def test_extract_unit_directions_atomic_cache_and_resume(runtime, tmp_path):
    rt, _ = runtime
    task = dict(kind="extract", layers=[1, 2], pairs=[dict(id="a", question="Q", positive="abc", negative="xy")],
                output_path=str(tmp_path / "new.npz"), cache_dir=str(tmp_path / "cache"))
    result = rt.execute(task)
    calls = rt.model.forward_calls
    again = rt.execute(task)
    assert rt.model.forward_calls == calls
    assert result == again
    with np.load(task["output_path"]) as arrays:
        assert set(arrays.files) == {"L1_prefix32", "L1_full", "L2_prefix32", "L2_full"}
        assert all(np.linalg.norm(arrays[k]) == pytest.approx(1.) for k in arrays.files)
    with pytest.raises(ValueError, match="Duplicate"):
        rt.execute(dict(task, pairs=task["pairs"] * 2))


def test_long_prompt_fails_without_silent_truncation(runtime):
    rt, spec = runtime
    with pytest.raises(ValueError, match="no question truncation"):
        rt.execute(dict(kind="dose", question="Q" * 100, positive="a", negative="b", direction=spec, max_sequence_tokens=20))
    with pytest.raises(ValueError, match="sequence budget"):
        rt.execute(dict(kind="generate", question="Q", max_new_tokens=100, max_sequence_tokens=20))
