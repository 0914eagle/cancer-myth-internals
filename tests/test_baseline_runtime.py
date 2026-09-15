"""Real tiny CPU Qwen/Gemma forwards; no pretrained weights or downloads."""
import numpy as np
import pytest

from src import baseline_generation as bg


@pytest.mark.parametrize("architecture", ["qwen", "gemma"])
def test_real_block_features_binary_readout_and_generation(architecture, monkeypatch):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
                              Gemma2Config, Gemma2ForCausalLM)
    from src import pilot_model

    torch.manual_seed(17)
    vocab = {w: i for i, w in enumerate(["<pad>", "<unk>", "<end>", "<user>",
                                        "<assistant>", "Yes", "No", "True", "False",
                                        "patient", "question", "answer"])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>",
                                 pad_token="<pad>", eos_token="<end>")
    tok.chat_template = "{% for m in messages %}{{ '<user> ' + m['content'] + ' <end> ' }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    settings = dict(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=256, pad_token_id=0, eos_token_id=2,
                    bos_token_id=3, attn_implementation="eager")
    if architecture == "qwen":
        model = Qwen2ForCausalLM(Qwen2Config(**settings)).eval()
    else:
        model = Gemma2ForCausalLM(Gemma2Config(**settings, head_dim=8, sliding_window=128)).eval()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(pilot_model, "load_model", lambda cfg: (model, tok))
    runtime = bg.ModelRuntime({"source_model": {"model_id": "offline-" + architecture}})
    prompt = "patient question"
    features = runtime.features(prompt, [1, 2])
    captured = {}
    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured[2] = h[0, -1].detach().cpu().numpy().copy()
    handle = model.model.layers[1].register_forward_hook(hook)
    ids, mask = runtime._encode(prompt)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[0, -1]
    handle.remove()
    np.testing.assert_allclose(features[2], captured[2], atol=1e-6)
    assert all(v.shape == (32,) for v in features.values())
    measured = runtime.binary(prompt)
    expected = torch.softmax(logits[[vocab["Yes"], vocab["No"]]], dim=0)[0].item()
    assert measured["score"] == pytest.approx(expected, abs=1e-6)
    answer = runtime.generate(prompt, 4)
    assert 1 <= answer["output_tokens"] <= 4
    assert answer["stop_reason"] in {"eos", "length"}
    assert not any(block._forward_hooks for block in model.model.layers)
