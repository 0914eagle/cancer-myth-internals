"""resolve_positions on a fake whitespace tokenizer, so the span logic is
tested without a model download."""

import re

from src.extract_activations import (
    encode_messages,
    encode_row,
    resolve_positions,
    substring_char_span,
    token_span_for_char_span,
)


class FakeTokenizer:
    """Whitespace tokens with offsets, and a chat template that wraps turns."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "<bos>"
        for m in messages:
            text += f"<{m['role']}> {m['content']} <end>\n"
        if add_generation_prompt:
            text += "<assistant> "
        return text

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False):
        ids, offsets = [], []
        for match in re.finditer(r"\S+", text):
            ids.append(hash(match.group(0)) % 1000)
            offsets.append((match.start(), match.end()))
        return {"input_ids": ids, "offset_mapping": offsets}


def test_target_text_resolves_to_the_tokens_overlapping_the_substring():
    tok = FakeTokenizer()
    row = {"id": "x", "prompt": "I think surgery is the only way. Is that right?", "position_mode": "target_text", "target_text": "surgery is the only way"}
    enc = encode_row(tok, row)
    span, selections = resolve_positions(row, enc, {})
    assert selections == []
    words = enc["text"].split()
    assert words[span[0]] == "surgery" and words[span[1] - 1].startswith("way")
    assert row["target_char_span"][1] - row["target_char_span"][0] == len("surgery is the only way")


def test_last_token_is_the_generation_prompt_token():
    tok = FakeTokenizer()
    row = {"id": "x", "prompt": "hello there", "position_mode": "last_token"}
    enc = encode_row(tok, row)
    span, selections = resolve_positions(row, enc, {})
    assert selections == ["last_token"] and span == (enc["n_tokens"] - 1, enc["n_tokens"])


def test_assistant_prefix_starts_at_the_first_response_token():
    tok = FakeTokenizer()
    messages = [{"role": "user", "content": "hello there"}, {"role": "assistant", "content": "Surgery is indeed the standard of care here"}]
    row = {"id": "x", "chat_messages": messages, "position_mode": "assistant_prefix", "prefix_tokens": 3}
    enc = encode_messages(tok, messages)
    span, selections = resolve_positions(row, enc, {})
    words = enc["text"].split()
    assert words[span[0] : span[1]] == ["Surgery", "is", "indeed"]
    assert selections == []


def test_char_span_helpers():
    text = "a b c b"
    assert substring_char_span(text, "b", occurrence=1) == (6, 7)
    assert token_span_for_char_span([(0, 1), (2, 3), (4, 5), (6, 7)], 2, 5) == (1, 3)
