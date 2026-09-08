from src.rows import (
    activation_rows,
    align_premise,
    align_premise_heuristic,
    content_words,
    find_verbatim,
    response_rows,
)

Q_BLADDER = (
    "My father was recently diagnosed with muscle-invasive bladder cancer, and he believes "
    "surgery is the only way to treat it. Are there any support groups that focus "
    "specifically on preparing for surgery?"
)
P_BLADDER = "Surgery is the only effective treatment for muscle-invasive bladder cancer."


def test_content_words_stem_and_drop_stopwords():
    assert "treatment" in content_words("the only effective treatments")
    assert "the" not in content_words("the only")


def test_heuristic_alignment_picks_the_belief_clause():
    span = align_premise_heuristic(Q_BLADDER, P_BLADDER)
    assert span is not None
    s, e, score = span
    picked = Q_BLADDER[s:e]
    assert "surgery is the only way to treat it" in picked
    assert "support groups" not in picked
    assert score >= 0.25


def test_heuristic_alignment_returns_none_when_nothing_overlaps():
    assert align_premise_heuristic(Q_BLADDER, "Vitamin C cures the common cold.") is None


def test_find_verbatim_tolerates_quotes_and_case():
    assert find_verbatim(Q_BLADDER, '"Surgery is the only way to treat it"') is not None
    assert find_verbatim(Q_BLADDER, "not in the question at all") is None


def test_llm_alignment_is_verified_retried_and_dropped():
    calls = []

    def bad_llm(prompt):
        calls.append(prompt)
        return "he thinks surgery is his only option"  # paraphrase, not verbatim

    span, score, method = align_premise(Q_BLADDER, P_BLADDER, llm=bad_llm)
    assert method == "none" and span is None and len(calls) == 2  # one retry, then A is dropped
    span, score, method = align_premise(Q_BLADDER, P_BLADDER, llm=bad_llm, heuristic_fallback=True)
    assert method == "heuristic" and span is not None

    def fixed_on_retry(prompt):
        return "he believes surgery is the only way to treat it" if "not an exact copy" in prompt else "he thinks surgery"

    span, score, method = align_premise(Q_BLADDER, P_BLADDER, llm=fixed_on_retry)
    assert method == "llm"

    def good_llm(prompt):
        return "he believes surgery is the only way to treat it"

    span, score, method = align_premise(Q_BLADDER, P_BLADDER, llm=good_llm)
    assert method == "llm" and Q_BLADDER[span[0] : span[1]] == "he believes surgery is the only way to treat it"


def test_activation_rows_emit_three_positions_when_aligned():
    q = {
        "id": "fpq_1",
        "set": "fpq",
        "label_false_premise": 1,
        "question": Q_BLADDER,
        "premise_text": P_BLADDER,
        "premise_span": [70, 117],
        "align_score": 1.0,
        "align_method": "llm",
    }
    rows = activation_rows([q])
    families = {r["position_family"] for r in rows}
    assert families == {"D_last_prompt_token", "B_question_end", "A_premise"}
    prem = next(r for r in rows if r["position_family"] == "A_premise")
    assert prem["target_text"] == Q_BLADDER[70:117]
    assert all(r["base_id"] == "fpq_1" and r["label_false_premise"] == 1 for r in rows)
    q["premise_span"] = None
    assert {r["position_family"] for r in activation_rows([q])} == {"D_last_prompt_token", "B_question_end"}


def test_response_rows_are_teacher_forced_transcripts():
    q = {"id": "fpq_1", "set": "fpq", "label_false_premise": 1, "question": Q_BLADDER, "pcr": -1}
    rows = response_rows([q], {"fpq_1": "Surgery is indeed the standard..."}, prefix_tokens=5)
    assert len(rows) == 1
    assert rows[0]["position_mode"] == "assistant_prefix"
    assert rows[0]["chat_messages"][1]["role"] == "assistant"
    assert response_rows([q], {}) == []


def test_response_rows_carry_judge_labels():
    q = {"id": "fpq_1", "set": "fpq", "question": "Q?", "pcr": -1, "judge_parsed": True}
    row = response_rows([q], {"fpq_1": "answer"}, prefix_tokens=5)[0]
    assert row["pcr"] == -1 and row["judge_parsed"] is True
    assert row.get("nfp_score") is None


def test_paired_rows_share_the_prompt_and_differ_only_in_the_answer():
    from src.rows import paired_rows

    q = {"id": "fpq_1", "set": "fpq", "question": "Q?", "pcr": -1}
    refs = {"fpq_1": {"corr": ("Gemini-1.5-Pro", "Actually, that is not right. " * 100), "follow": ("GPT-3.5", "Yes, indeed.")}}
    rows = paired_rows([q, {"id": "nfp_1", "set": "nfp", "question": "N?"}], refs, {"fpq_1": "own answer"}, prefix_tokens=(5, 32))
    assert len(rows) == 6
    assert {r["pair_role"] for r in rows} == {"corr", "follow", "own"}
    assert {r["position_family"] for r in rows} == {"E_pair_first5", "E_pair_first32"}
    assert all(r["chat_messages"][0]["content"] == "Q?" for r in rows)
    corr5 = next(r for r in rows if r["id"] == "fpq_1__corr5")
    assert len(corr5["chat_messages"][1]["content"]) == 800 and corr5["pair_author"] == "Gemini-1.5-Pro"
    own = next(r for r in rows if r["pair_role"] == "own" and r["prefix_tokens"] == 32)
    assert own["pcr"] == -1 and own["base_id"] == "fpq_1"


def test_true_twin_is_spliced_and_checked():
    from src.rows import make_true_twin

    q = {"id": "fpq_9", "set": "fpq", "question": Q_BLADDER, "premise_text": P_BLADDER,
         "correction": "Surgery is one of several options; bladder-sparing treatments exist.",
         "premise_span": [Q_BLADDER.index("he believes"), Q_BLADDER.index("treat it") + len("treat it")]}

    def llm(prompt):
        if prompt.startswith("Does the following"):
            return "NO"
        return "he knows surgery is one of several ways to treat it"

    twin, status = make_true_twin(q, llm)
    assert status == "ok" and twin["set"] == "tpair" and twin["pair_id"] == "fpq_9"
    s, e = twin["premise_span"]
    assert twin["question"][s:e] == "he knows surgery is one of several ways to treat it"
    # everything outside the span is byte-identical
    os_, oe = q["premise_span"]
    assert twin["question"][:s] == Q_BLADDER[:os_] and twin["question"][e:] == Q_BLADDER[oe:]

    def still_false(prompt):
        return "YES" if prompt.startswith("Does the following") else "he believes surgery is the only way to fix it"

    assert make_true_twin(q, still_false)[1] == "still_false"
    assert make_true_twin({**q, "premise_span": None}, llm)[1] == "no_span"
