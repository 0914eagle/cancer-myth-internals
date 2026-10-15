"""Offline equivalence tests against pinned Well, not copied expectations alone."""
import json
from pathlib import Path
import pytest

from src.fpqa_cli_backend import envelope
from src.fpqa_well_upstream import load_upstream, make_args, UncachedCaller, FatalTransportError

ROOT = Path(__file__).resolve().parents[1]
WELL = ROOT / "external/fpqa_20261002/Well"


def config():
    return {f"{role}_model": {"backend": "codex", "model": "gpt-6-luna", "effort": "medium", "timeout_seconds": 240}
            for role in ("task", "judge", "reflection")}


@pytest.fixture(scope="module")
def upstream():
    if not WELL.exists():
        pytest.skip("Pinned Well checkout required")
    calls = []
    def call(role, messages):
        calls.append((role, messages))
        if messages[0]["content"] == "You are a helpful assistant.":
            return "Explanation. Rating: 4\nRating: 4"
        return "ASSISTANT\nA MiXeD Case Answer"
    return (*load_upstream(WELL, call), calls)


@pytest.mark.parametrize("family,kind", [("CREPE", "fpq"), ("CREPE", "tpq"), ("CancerMyth", "fpq"), ("CancerMyth", "tpq")])
def test_upstream_evaluation_identical_messages_score_feedback(upstream, tmp_path, family, kind):
    balanced, direct, original, calls = upstream
    args = make_args(balanced, family, tmp_path, tmp_path, config())
    row = {"id": "test", "question": "Test question?", "presuppositions": ["GOLD PREMISE"],
           "answer": "REFERENCE", "few_shot_data": [], "passages": [], "_gepa_question_type": kind}
    candidate = {"system_prompt": direct.load_seed_prompt(args)}
    calls.clear()
    expected = original(args, "gemini")(candidate, row)
    expected_messages = [m for _, m in calls]
    calls.clear()
    observed = balanced.BalancedDirectQAGEPAEvaluator(args, "gemini")(candidate, row)
    assert observed == expected
    assert [m for _, m in calls] == expected_messages
    assert [r for r, _ in calls] == ["task", "judge"]
    assert observed[0] == .8
    assert observed[1]["answer"] == "a mixed case answer"
    assert set(observed[1]) == {"question", "answer", "raw_score", "normalized_score", "score_explanation", "question_type", "score_template"}
    assert "GOLD" not in json.dumps(calls[0])
    for _, messages in calls:
        assert envelope(messages, exact=True) == (messages[0]["content"], messages[1]["content"])


def test_exact_transport_adds_no_text_and_rejects_serialization():
    messages = [{"role": "system", "content": "  Exact instruction\n"}, {"role": "user", "content": "Question?\n"}]
    assert envelope(messages, exact=True) == ("  Exact instruction\n", "Question?\n")
    with pytest.raises(ValueError, match=r"system \+ user"):
        envelope(messages + [{"role": "assistant", "content": "x"}], exact=True)


def test_original_optimizer_arguments_and_interleaving(upstream, tmp_path, monkeypatch):
    balanced, direct, _, _ = upstream
    for split in ("train", "val"):
        for kind in ("fpq", "tpq"):
            rows = [{"id": f"{split}_{kind}_{i}", "question": f"{kind}{i}", "few_shot_data": ["REMOVE"]} for i in range(2)]
            (tmp_path / f"{kind}_{split}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    args = make_args(balanced, "CREPE", tmp_path, tmp_path, config())
    captured = {}
    monkeypatch.setattr(balanced, "optimize_anything", lambda **kw: captured.update(kw))
    monkeypatch.setattr(balanced, "save_result", lambda *a: None)
    balanced.run_gepa(args, "gemini")
    engine = captured["config"].engine
    assert engine.parallel is False and engine.cache_evaluation is False
    assert engine.max_metric_calls == 500 and engine.seed == 42
    assert engine.max_candidate_proposals is None
    assert captured["objective"] == args.objective
    assert captured["background"] == args.background
    assert captured["seed_candidate"] == {"system_prompt": direct.DEFAULT_SEED_PROMPT}
    assert [r["_gepa_question_type"] for r in captured["dataset"]] == ["fpq", "tpq", "fpq", "tpq"]
    assert all(r["few_shot_data"] == [] for r in captured["dataset"])


def test_calls_are_recorded_but_never_memoized(tmp_path):
    call = UncachedCaller(config(), tmp_path)
    observed = []
    call.clients["task"] = lambda messages: (observed.append(messages) or "answer", {})
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    call("task", messages)
    call("task", messages)
    assert len(observed) == 2
    assert len(list((tmp_path / "calls").glob('*_task.json'))) == 2
    (tmp_path / "STOP").touch()
    with pytest.raises(FatalTransportError, match="STOP"):
        call("task", messages)


def test_original_rating_parser_behavior(upstream):
    from response import ResponseLevelScoreResponse
    assert ResponseLevelScoreResponse.model_validate_plain_text("**Rating:** 4\nRating: 5").score == 4
    assert ResponseLevelScoreResponse.model_validate_plain_text("missing rating").score == 0


def test_original_run_gepa_with_real_engine_mock_models(upstream, tmp_path, monkeypatch):
    from types import SimpleNamespace
    balanced, direct, _, _ = upstream
    for split in ("train", "val"):
        for kind in ("fpq", "tpq"):
            rows = [{"id": f"{split}_{kind}", "question": f"{split} {kind}?",
                     "presuppositions": ["premise"], "answer": "reference", "few_shot_data": []}]
            (tmp_path / f"{kind}_{split}.jsonl").write_text("\n".join(map(json.dumps, rows)))
    args = make_args(balanced, "CREPE", tmp_path, tmp_path, config())
    args.max_metric_calls = 14  # Test fixture only; production uses original 500.
    def transport(messages, **kwargs):
        if messages[0]["content"] == "You are a helpful assistant.":
            text = "Rating: 5" if "improved answer marker" in messages[1]["content"] else "Rating: 1"
        else:
            text = "improved answer marker" if "IMPROVED" in messages[0]["content"] else "base answer"
        return SimpleNamespace(text=text)
    monkeypatch.setattr(direct, "call_gemini_one_by_one_api", transport)
    monkeypatch.setattr(balanced, "create_reflection_lm", lambda **kw: lambda prompt: '```\nIMPROVED answer instruction\n```')
    balanced.run_gepa(args, "gemini")
    saved = json.loads((tmp_path / "gepa_result_summary.json").read_text())
    assert saved["best_score"] == 1.0
    assert "IMPROVED" in saved["best_system_prompt"]


def test_empty_system_uses_catalog_without_mutating_other_context(tmp_path, monkeypatch):
    from src.fpqa_cli_backend import codex_instruction_options
    cache = tmp_path / 'cache'
    cache.mkdir()
    original = {'models': [{'slug': 'gpt-6-luna', 'tool_mode': 'unchanged', 'model_messages': {
        'instructions_template': 'default task instructions', 'persistent_instructions': 'retain',
        'permissions': {'retain': True}}}]}
    source = cache / 'models_cache.json'
    source.write_text(json.dumps(original))
    monkeypatch.setenv('CODEX_HOME', str(cache))
    options = codex_instruction_options(tmp_path, '', 'gpt-6-luna')
    assert 'model_instructions_file' not in options
    output = json.loads(Path(options['model_catalog_json']).read_text())
    expected = json.loads(json.dumps(original))
    expected['models'][0]['model_messages']['instructions_template'] = ''
    assert output == expected
    assert json.loads(source.read_text()) == original


def test_reflection_failure_cannot_be_swallowed_as_exception(tmp_path):
    caller = UncachedCaller(config(), tmp_path)
    def fail(messages):
        raise RuntimeError('CLI transport failure')
    caller.clients['reflection'] = fail
    with pytest.raises(FatalTransportError):
        try:
            caller('reflection', [{'role': 'system', 'content': None}, {'role': 'user', 'content': 'improve'}])
        except Exception:
            pytest.fail('GEPA would swallow this failure and continue spending calls')
    assert json.loads((tmp_path / 'FAILED.json').read_text())['role'] == 'reflection'
