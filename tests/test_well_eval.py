"""Offline protocol checks: no generation, API calls or downloads."""
import ast
import json
from types import SimpleNamespace

import pytest

from src.jsonl import write_jsonl
from src.well_eval import (
    TEMPLATES, DEFAULT_MODEL, example_questions, judge_messages, load_plan,
    parse_rating, prepare, read_ledger, report, score, template_manifest,
)


@pytest.mark.parametrize("raw,value", [
    ("Reason. Rating: 0", 0), ("Reason\nRating: 5", 5),
    ("Reason **Rating: 4**", 4), ('{"Rating": 3, "Reason": "OK"}', 3),
    ('```json\n{"Rating": 2}\n```', 2),
    ("Rating: 6", None), ("Rating: -1", None), ("Rating: 5.0", None),
    ("Rating: 5/5", None), ("Rating: 50", None), ("Rating: 1\nRating: 5", None),
    ("Rating: 5\nActually 1", None), ("good answer", None), ("5", None),
    ('{"Rating": true}', None), ('{"Rating": 5.0}', None),
    ('{"Rating": 5, "Rating": 5}', None), ('{"Score":5}', None),
    ('{"Rating":5,"Score":1}', None), (None, None),
])
def test_strict_rating(raw, value):
    assert parse_rating(raw) == value


def _upstream_generate(kind, question, answer):
    """Execute only reviewed generate AST, not upstream imports/decorators."""
    name = "CancerMyth" if kind == "fpq" else "CancerMythNFP"
    text = (TEMPLATES / "upstream" / f"data_gen__template__{name}_template.py").read_text()
    if kind == "nfp":
        text = text[:text.index("\n\n@register()", text.index("class CancerMythNFPResponse"))]
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    gen = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
    gen.returns = None
    module = ast.fix_missing_locations(ast.Module(body=[gen], type_ignores=[]))
    env = {}
    exec(compile(module, "reviewed-generate-only", "exec"), env)
    obj = SimpleNamespace(question=question["question"], model_final_answer=answer,
                          presupposition=question.get("premise_text"), correction=question.get("correction"),
                          user_role="user", system_role="system", model_role="assistant")
    return env["generate"](obj)


def test_vendored_templates_match_upstream_byte_for_byte():
    assert template_manifest()["revision"] == "a7ee871eadde1104f7560a2874a03cbf5221dbaa"
    for kind in ("fpq", "nfp"):
        q = {"set": kind, "question": "A question {with braces}", "premise_text": "False claim",
             "correction": "Correct explanation"}
        assert judge_messages(q, "Answer\nwith {braces}") == _upstream_generate(kind, q, "Answer\nwith {braces}")
    demos = example_questions()
    assert len(demos) == 2
    assert "chemotherapy pain" in demos[0]
    assert "grandmother" in demos[1]


@pytest.fixture
def run(tmp_path):
    qs = [{"id": "f", "set": "fpq", "question": "False question?", "premise_text": "False claim",
           "correction": "Accurate correction", "partition": "dev"},
          {"id": "n", "set": "nfp", "question": "Normal question?", "partition": "dev"}]
    ans = [{"id": q["id"], "question": q["question"], "method": m, "response": "Answer"}
           for m in ("plain", "cot") for q in qs]
    qp, ap, out = tmp_path / "questions.jsonl", tmp_path / "answers.jsonl", tmp_path / "out"
    write_jsonl(qp, qs)
    write_jsonl(ap, ans)
    return qp, ap, out, qs, ans


def test_frozen_dedup_capped_resume_and_report(run):
    qp, ap, out, _, _ = run
    result = prepare(qp, [ap], out)
    assert result == {"unique_calls": 2, "answer_rows": 4, "shared_calls": 2, "calls_now": 0}
    assert prepare(qp, [ap], out) == result
    early = report(out)
    assert early["methods"]["plain"]["fpq"]["valid"] == 0
    assert early["methods"]["plain"]["fpq"]["mean_1_to_5"] is None
    calls = []
    def caller(prompt):
        calls.append(prompt)
        return "Rating: 0" if "NO false" in prompt else "Rating: 5", DEFAULT_MODEL
    assert score(out, max_calls=1, caller=caller)["new_calls"] == 1
    assert len(calls) == 1
    assert score(out, max_calls=10, caller=caller)["new_calls"] == 1
    assert score(out, max_calls=10, caller=caller)["new_calls"] == 0
    assert len(calls) == 2
    result = report(out)
    assert result["unique_valid"] == 2
    assert result["methods"]["plain"]["nfp"]["counts_0_to_5"]["0"] == 1
    assert result["methods"]["plain"]["nfp"]["mean_denominator"] == 0
    assert result["methods"]["plain"]["fpq"]["s5_over_all"] == 1
    assert result["methods"]["cot"]["fpq"]["paired_plain"]["n"] == 1


def test_interrupt_consumes_attempt_never_retries(run):
    qp, ap, out, _, _ = run
    prepare(qp, [ap], out)
    def interrupted(_):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        score(out, max_calls=1, caller=interrupted)
    plan, h = load_plan(out)
    events, scores = read_ledger(out, plan, h)
    assert len(events) == 1 and not scores
    calls = []
    def caller(prompt):
        calls.append(prompt)
        return "Rating: 4", DEFAULT_MODEL
    assert score(out, max_calls=10, caller=caller)["new_calls"] == 1
    assert len(calls) == 1
    result = report(out)
    assert result["attempted"] == 2 and result["unique_valid"] == 1


def test_invalid_is_not_zero_or_pass_and_wrong_model_stops(run):
    qp, ap, out, _, _ = run
    prepare(qp, [ap], out)
    assert score(out, max_calls=5, caller=lambda _: ("Rating: 5", "wrong-model"))["new_calls"] == 1
    result = report(out)
    assert result["unique_valid"] == 0
    assert all(s["gibberish_count"] == 0 and s["s5_count"] == 0
               for m in result["methods"].values() for s in m.values())


def test_missing_generated_answer_keeps_expected_denominator(run):
    qp, ap, out, _, ans = run
    write_jsonl(ap, ans[:-1])
    prepare(qp, [ap], out)
    score(out, max_calls=10, caller=lambda _: ("Rating: 5", DEFAULT_MODEL))
    result = report(out)
    assert result["methods"]["cot"]["nfp"]["missing"] == 1
    assert result["methods"]["cot"]["nfp"]["expected"] == 1


def test_no_hidden_reference_fallback_and_no_demo_leak(run):
    qp, ap, out, qs, _ = run
    del qs[0]["premise_text"]
    write_jsonl(qp, qs)
    with pytest.raises(ValueError, match="needs premise_text"):
        prepare(qp, [ap], out)
    qs[0]["premise_text"] = "claim"
    qs[0]["question"] = example_questions()[0]
    write_jsonl(qp, qs)
    with pytest.raises(ValueError, match="demonstration overlaps"):
        prepare(qp, [ap], out)


def test_duplicates_question_mismatch_and_plan_mutation(run):
    qp, ap, out, _, ans = run
    write_jsonl(ap, ans + [ans[0]])
    with pytest.raises(ValueError, match="Duplicate method"):
        prepare(qp, [ap], out)
    ans[0]["question"] = "Another question"
    write_jsonl(ap, ans)
    with pytest.raises(ValueError, match="question text differs"):
        prepare(qp, [ap], out)
    ans[0]["question"] = "False question?"
    write_jsonl(ap, ans)
    prepare(qp, [ap], out)
    changed = json.loads((out / "judge_plan.json").read_text())
    changed["plan"]["model"] = "changed"
    (out / "judge_plan.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Modified"):
        report(out)


def test_calls_cap_required_and_positive(run):
    qp, ap, out, _, _ = run
    prepare(qp, [ap], out)
    for bad in (None, 0, -1, True):
        with pytest.raises(ValueError, match="max_calls"):
            score(out, max_calls=bad)


def test_empty_answer_file_is_not_silently_dropped(run):
    qp, ap, out, _, _ = run
    empty = ap.parent / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="Empty answer file"):
        prepare(qp, [ap, empty], out)


def test_claude_backend_accepts_dated_served_model_and_rejects_other(run):
    from src.well_eval import model_matches
    assert model_matches("claude-sonnet-5", "claude-sonnet-5")
    assert model_matches("claude-sonnet-5-20260601", "claude-sonnet-5")
    assert not model_matches("claude-sonnet-5", "claude-sonnet-5-20260601")
    assert not model_matches("claude-sonnet-50", "claude-sonnet-5")
    assert not model_matches("claude-opus-5", "claude-sonnet-5")
    qp, ap, out, _, _ = run
    plan = prepare(qp, [ap], out, backend="claude", model="claude-sonnet-5")
    assert plan["unique_calls"] == 2
    served = lambda _: ("Rating: 4", "claude-sonnet-5-20260601")
    assert score(out, max_calls=10, caller=served)["new_calls"] == 2
    assert report(out)["unique_valid"] == 2
    # An unrelated served model still stops the run and never counts.
    qp2, ap2, out2, _, _ = run
    out2 = out2.parent / "other"
    prepare(qp2, [ap2], out2, backend="claude", model="claude-sonnet-5")
    assert score(out2, max_calls=10, caller=lambda _: ("Rating: 4", "claude-opus-5"))["new_calls"] == 1
    assert report(out2)["unique_valid"] == 0


def test_claude_score_preflight_fails_before_any_ledger_event(run, monkeypatch):
    from src import well_eval
    qp, ap, out, _, _ = run
    prepare(qp, [ap], out, backend="claude", model="claude-sonnet-5")
    def broken(*a, **k):
        def call(prompt):
            raise RuntimeError("claude -p failed (1): usage limit reached")
        return call
    monkeypatch.setattr(well_eval, "make_caller", broken)
    with pytest.raises(RuntimeError, match="usage limit"):
        score(out, max_calls=5)
    assert not (out / "attempts.jsonl").exists()
