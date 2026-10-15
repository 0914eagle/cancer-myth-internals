import pytest

from scripts.premise_replacement_full import arms_for, parse_edit, score_rule

Q = "My brother was diagnosed. He's heard that surgery is the only option, and he’s worried. Are there alternatives?"


def test_parse_edit_accepts_json_with_prose_and_curly_quotes():
    reply = ('Here you go:\n{"keep": [2, "1"], "add": [{"premise": " Surgery is the only option. ", '
             '"evidence": "he\'s worried"}], "note": "split hearsay"}')
    edit = parse_edit(reply, 3, Q)
    assert edit["keep"] == [1, 2]
    assert edit["add"][0]["premise"] == "Surgery is the only option."
    assert edit["add"][0]["evidence_exact"] is False  # straight vs curly apostrophe, accepted after normalisation
    assert edit["note"] == "split hearsay"


def test_parse_edit_rejects_bad_indices_and_missing_evidence():
    with pytest.raises(ValueError, match="1..2"):
        parse_edit('{"keep": [3], "add": []}', 2, Q)
    with pytest.raises(ValueError, match="repeated"):
        parse_edit('{"keep": [1, 1], "add": []}', 2, Q)
    with pytest.raises(ValueError, match="not a substring"):
        parse_edit('{"keep": [], "add": [{"premise": "x", "evidence": "not in the question"}]}', 2, Q)
    with pytest.raises(ValueError, match="no JSON"):
        parse_edit("I cannot help", 2, Q)


def test_arms_and_rule():
    original = ["a", "b", "c"]
    arms = arms_for(original, {"keep": [1, 3], "add": [{"premise": "d"}, {"premise": "a"}]})
    assert arms == {"original": ["a", "b", "c"], "retained_only": ["a", "c"], "reviewed": ["a", "c", "d"]}
    assert score_rule([0.9, 0.2]) == {"score": 0.8, "gate": True}
    assert score_rule([]) == {"score": 0, "gate": False}
