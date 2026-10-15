from scripts.nla_av_batched import trim_eos


def test_gemma_end_of_turn_stops_before_padding_and_repetition():
    assert trim_eos([5,7,106,106,1,1],{1,106})==([5,7,106],True)


def test_qwen_end_token_and_truncated_output():
    assert trim_eos([5,7,151645,151643],{151645})==([5,7,151645],True)
    assert trim_eos([5,7],{151645})==([5,7],False)
