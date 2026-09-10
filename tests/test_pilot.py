import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_judge import build_prompt
from scripts.summarize_judge import summarize
from src.jsonl import append_jsonl, read_jsonl, write_jsonl
from src.judge_prompts import parse_score
from src.pilot import (
    check_resume,
    digest,
    frozen_json,
    load_manifest,
    make_manifest,
    parse_identification,
    prepare_rows,
    quarantine_conflicts,
    score_map,
    summarize_pair,
)


def data(n=30):
    questions, refs = [], []
    for i in range(n):
        questions.extend(
            [
                {
                    "id": f"f{i}",
                    "question": f"false question {i}",
                    "set": "fpq",
                    "correction": "correct",
                },
                {
                    "id": f"n{i}",
                    "question": f"normal question {i}",
                    "set": "nfp",
                    "hallucination_text": "false",
                },
                {
                    "id": f"t{i}",
                    "question": f"normal question {i}",
                    "set": "tpq",
                    "premise_text": "true",
                },
            ]
        )
        refs.append(
            {
                "example_question": f"false question {i}",
                "source_row": i // 2,
                "source_type": "myth",
                "source_info": {"myth": f"myth long enough {i // 2}"},
                "answers": {"a": "correct answer", "b": "follow answer"},
                "scores": {"a": 1, "b": -1},
            }
        )
    return questions, refs


def test_grouped_manifest_deduplicates_normals_and_holds_out_myths(tmp_path):
    qs, refs = data()
    rows = prepare_rows(qs, refs)
    assert len(rows) == 60
    assert rows == prepare_rows(list(reversed(qs)), list(reversed(refs)))
    by_id = {q["id"]: q for q in rows}
    for i in range(0, 30, 2):
        assert by_id[f"f{i}"]["partition"] == by_id[f"f{i + 1}"]["partition"]
    groups = [{q["group_id"] for q in rows if q["partition"] == p} for p in ("fit", "dev", "test")]
    assert not (groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
    m = make_manifest(rows, seed=17, folds=5, test_fold=0, dev_fold=1, reference_hash="raw")
    path = tmp_path / "manifest.json"
    frozen_json(path, m)
    assert load_manifest(path) == m
    m["questions"][0]["partition"] = "test"
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Modified"):
        load_manifest(path)


def test_missing_source_metadata_is_not_one_giant_myth():
    qs, refs = data()
    for r in refs:
        r["source_row"] = -1
        r["source_info"]["myth"] = "From physicians."
    rows = prepare_rows(qs, refs)
    assert len({q["group_id"] for q in rows}) == 60


def test_transitive_links_and_invalid_normal_annotations():
    qs, refs = data()
    qs[0]["pair_id"] = "link"
    qs[6]["pair_id"] = "link"  # f0 and f2 join their two myth groups
    rows = prepare_rows(qs, refs)
    assert len({q["group_id"] for q in rows if q["id"] in {"f0", "f1", "f2", "f3"}}) == 1
    with pytest.raises(ValueError, match="TPQ without"):
        prepare_rows([q for q in qs if q["id"] != "n0"], refs)


def test_contradictory_labels_are_quarantined_without_choosing_a_truth():
    qs, refs = data()
    qs[1]["question"] = qs[0]["question"]
    clean, audit = quarantine_conflicts(qs)
    assert audit[0]["ids"] == ["f0", "n0"]
    assert not {"f0", "n0"} & {q["id"] for q in clean}
    with pytest.raises(ValueError, match="conflicting"):
        prepare_rows(qs, refs)


def test_resume_rejects_changed_settings_and_unknown_rows(tmp_path):
    path = tmp_path / "out.jsonl"
    spec = {"alpha_abs": 2.0, "max_memory": {0: "22GiB"}}
    assert check_resume(path, spec, {"a"}) == set()
    append_jsonl(path, {"id": "a", "run_hash": digest(spec)})
    assert check_resume(path, spec, {"a"}) == {"a"}
    with pytest.raises(ValueError, match="mismatch"):
        check_resume(path, {**spec, "alpha_abs": 3.0}, {"a"})
    with pytest.raises(ValueError, match="IDs"):
        check_resume(path, spec, {"b"})


def score(qid, kind, value, parsed=True):
    return {
        "id": qid + "::" + qid,
        "question_id": qid,
        "set": kind,
        "sharpness": value,
        "judge_parsed": parsed,
        "judge_backend": "openai",
        "judge_model": "gpt-4o",
        "judge_temperature": 0.0,
    }


def test_bad_judges_are_not_successes_or_valid_selection_inputs(tmp_path):
    scores = [score("a", "fpq", 1, False), score("b", "fpq", -1), score("c", "nfp", 0)]
    result = summarize(scores, {})
    assert result["invalid_or_unparsed"] == 2
    assert result["fpq"]["PCR"] == 0 and result["fpq"]["n"] == 1
    assert "nfp" not in result
    path = tmp_path / "scores.jsonl"
    write_jsonl(path, scores)
    with pytest.raises(ValueError, match="Unparsed"):
        score_map(path, {s["question_id"]: s for s in scores})
    for invalid in (2, 1.5, True, "1"):
        _, parsed = parse_score(json.dumps({"Sharpness": invalid}, indent=2))
        assert not parsed
    with pytest.raises(ValueError, match="TPQ needs"):
        build_prompt({"set": "tpq"}, {}, [], [])


def test_paired_rescue_harm_and_ci_denominators():
    qs = [{"id": str(i), "set": "fpq" if i < 3 else "nfp", "group_id": str(i)} for i in range(5)]
    plain = {q["id"]: score(q["id"], q["set"], v) for q, v in zip(qs, [-1, 1, 0, 1, 1])}
    after = {q["id"]: score(q["id"], q["set"], v) for q, v in zip(qs, [1, -1, 1, -1, 1])}
    r = summarize_pair(qs, after, plain, bootstraps=100)
    assert (r["fpq"]["rescue"], r["fpq"]["harm"]) == (2, 1)
    assert r["fpq"]["delta_PCR"] == pytest.approx(100 / 3)
    assert r["nfp"]["delta_NFP"] == -50


def test_identification_does_not_treat_ambiguous_output_as_no():
    assert parse_identification("Yes.") is True
    assert parse_identification(" No\n") is False
    with pytest.raises(ValueError):
        parse_identification("No, but yes in some situations")


@pytest.mark.parametrize("text,expected", [
    ("Yes \n\n**False Assumption:** That no treatment will be done because the lymphoma", True),
    ("**Yes.** The question assumes treatment is unnecessary.", True),
    ("No, the question contains valid premises.", False),
    ("No\n\n**Explanation:** The patient's premise is correct.", False),
])
def test_identification_accepts_leading_verdict_with_explanation(text, expected):
    assert parse_identification(text) is expected


@pytest.mark.parametrize("text", ["", "Yesterday", "Nobody", "I think yes", "Yes/no", "Yes or no", "Yes, but no."])
def test_identification_rejects_missing_or_conflicting_verdict(text):
    with pytest.raises(ValueError):
        parse_identification(text)


def test_parser_migration_preserves_answers_scores_and_rejects_other_code(tmp_path):
    from scripts.migrate_pilot_identification import OLD_IMPLEMENTATION, NEW_IMPLEMENTATION, migrate
    from src.pilot import file_digest

    source, destination = tmp_path / "old", tmp_path / "new"
    rows = prepare_rows(*data())
    manifest = make_manifest(rows, seed=17, folds=5, test_fold=0, dev_fold=1, reference_hash="raw")
    frozen_json(source / "split/manifest.json", manifest)
    for part in ("fit", "dev", "test"):
        write_jsonl(source / "split" / f"{part}.jsonl", [q for q in rows if q["partition"] == part])
    dev = [q for q in rows if q["partition"] == "dev"]
    path = source / "dev/fp_identification.jsonl"
    spec = {"identity": {"implementation_hash": OLD_IMPLEMENTATION}, "method": "fp_identification", "max_new_tokens": 512,
            "partition": "dev", "manifest_hash": manifest["manifest_hash"], "question_ids": [q["id"] for q in dev]}
    frozen_json(Path(str(path) + ".run.json"), spec)
    records = [{"id": q["id"], "response": f"original answer {q['id']}", "run_hash": digest(spec),
                "review": "Yes.", "identified_false_premise": True} for q in dev]
    write_jsonl(path, records)
    scores = source / "dev/fp_identification_judge_codex_test.jsonl"
    signature = {"response_run": spec, "responses_hash": file_digest(path),
                 "questions_hash": file_digest(source / "split/dev.jsonl")}
    judged = [{**score(q["id"], q["set"], 1), "response_id": q["id"],
               "response_run_hash": digest(spec), "judge_run_hash": digest(signature),
               "response_text_hash": digest(r["response"])} for q, r in zip(dev, records)]
    write_jsonl(scores, judged)
    frozen_json(Path(str(scores) + ".run.json"), signature)
    original = path.read_bytes()
    migrate(source, destination)
    assert path.read_bytes() == original
    updated = json.loads((destination / "dev/fp_identification.jsonl.run.json").read_text())
    assert updated["identity"]["implementation_hash"] == NEW_IMPLEMENTATION
    assert check_resume(destination / "dev/fp_identification.jsonl", updated, {q["id"] for q in dev}) == {q["id"] for q in dev}
    copied = list(read_jsonl(destination / "dev/fp_identification.jsonl"))
    assert [r["response"] for r in copied] == [r["response"] for r in records]
    from scripts.run_pilot import load_scored
    from scripts.migrate_pilot_identification import repair
    migrated_scores = destination / "dev" / scores.name
    loaded, _, _ = load_scored(migrated_scores, manifest, "dev")
    assert len(loaded) == len(dev)
    new_rows = list(read_jsonl(migrated_scores))
    for new, old in zip(new_rows, judged):
        assert {k:v for k,v in new.items() if k not in {"response_run_hash", "judge_run_hash"}} == {
            k:v for k,v in old.items() if k not in {"response_run_hash", "judge_run_hash"}}
    # Recreate the exact old migration bug, with some scores newly judged after resume.
    write_jsonl(migrated_scores, judged[:2] + new_rows[2:])
    with pytest.raises(ValueError, match="Score provenance differs"):
        load_scored(migrated_scores, manifest, "dev")
    repair(destination)
    assert list(read_jsonl(migrated_scores)) == new_rows
    assert len(load_scored(migrated_scores, manifest, "dev")[0]) == len(dev)
    repaired_bytes = migrated_scores.read_bytes()
    repair(destination)
    assert migrated_scores.read_bytes() == repaired_bytes
    assert list(destination.glob("dev/*.bak"))
    # Exercise the actual report path, not just migration's own metadata checks.
    from scripts.run_pilot import report
    plain_run = {**updated, "method": "plain"}
    plain_path = destination / "dev/plain_judge_test.jsonl"
    write_jsonl(plain_path, [{**r, "response_run_hash": digest(plain_run)} for r in new_rows])
    frozen_json(Path(str(plain_path) + ".run.json"), {"response_run": plain_run})
    report(SimpleNamespace(manifest=destination / "split/manifest.json", partition="dev",
                           scores=[plain_path, migrated_scores], output=destination / "report.json"))
    assert (destination / "report.md").exists()
    # Reject edits to actual judge decisions; repair must not bless arbitrary rows.
    corrupted = [dict(r) for r in judged]
    corrupted[0]["sharpness"] = -1
    write_jsonl(migrated_scores, corrupted)
    with pytest.raises(ValueError, match="not an unchanged audited original"):
        repair(destination)
    assert list(read_jsonl(migrated_scores)) == corrupted
    assert (destination / "parser_migration.json").exists()
    with pytest.raises(ValueError, match="Destination already"):
        migrate(source, destination)
    spec["identity"]["implementation_hash"] = "other"
    Path(str(path) + ".run.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Unsupported generation"):
        migrate(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


@pytest.fixture
def tiny_gemma():
    """Real randomly initialized Gemma2 + fast tokenizer, no downloads/GPU."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(17)
    vocab = {
        w: i
        for i, w in enumerate(
            [
                "<pad>",
                "<unk>",
                "<end>",
                "<user>",
                "<assistant>",
                "correct",
                "follow",
                "answer",
                "false",
                "normal",
                "question",
                "Yes",
                "No",
            ]
        )
    }
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<end>",
        additional_special_tokens=["<user>", "<assistant>"],
    )
    tok.chat_template = "{% for m in messages %}{{ '<user> ' if m['role'] == 'user' else '<assistant> ' }}{{ m['content'] }}{{ ' <end> ' }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    tok.padding_side = "left"
    config = Gemma2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        sliding_window=128,
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=3,
        attn_implementation="eager",
    )
    model = Gemma2ForCausalLM(config).eval()
    return model, tok


def test_real_gemma_hook_touches_last_prompt_and_decode_only(tiny_gemma):
    import torch
    from src.steering import SteerSpec, Steerer

    model, _ = tiny_gemma
    ids = torch.tensor([[3, 9, 10, 4]])
    direction = torch.ones(32) / np.sqrt(32)
    seen = []
    handle = model.model.layers[1].register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0].clone())
    )
    with torch.inference_mode():
        baseline = model(ids, use_cache=False, output_hidden_states=True)
        with Steerer(model, SteerSpec(1, direction, 0.5), from_position=3):
            changed = model(ids, use_cache=False, output_hidden_states=True)
    handle.remove()
    # Recent Transformers captures output_hidden_states with its own earlier
    # hook. Inspect the next block's actual input to check the intervention.
    assert torch.allclose(seen[1][:, :3], seen[0][:, :3])
    assert torch.allclose(seen[1][:, 3] - seen[0][:, 3], (0.5 * direction)[None, :], atol=1e-6)
    assert not torch.allclose(changed.logits[:, -1], baseline.logits[:, -1])
    assert not model.model.layers[0]._forward_hooks
    with pytest.raises(IndexError, match="final"):
        Steerer(model, SteerSpec(3, direction, 1), 3)


def test_real_gemma_zero_alpha_matches_plain_and_fit_excludes_dev(tiny_gemma, tmp_path):
    import torch
    from src.pilot_model import answer_prefix, generate_batch, learn_direction
    from src.steering import SteerSpec

    model, tok = tiny_gemma
    cfg = {"source_model": {"model_id": "tiny-gemma-test", "max_memory": {0: "22GiB"}}}
    qs, refs = data(12)
    rows = prepare_rows(qs, refs, folds=3)
    manifest = make_manifest(rows, seed=17, folds=3, test_fold=0, dev_fold=1, reference_hash="test")
    ids, (start, end) = answer_prefix(tok, "question", "correct answer", 32)
    assert tok.decode(ids[start:end]) == "correct answer"  # no EOS in C's pooling span
    fitted = learn_direction(
        model, tok, cfg, manifest, refs, layers=[1], prefix_tokens=2, out_dir=tmp_path / "fit"
    )
    expected_fit = {q["id"] for q in rows if q["partition"] == "fit"}
    assert set(fitted["fit_ids"]) == expected_fit
    assert set(fitted["pair_ids"]) <= expected_fit
    held_refs_changed = [dict(r) for r in refs]
    held_text = {q["question"] for q in rows if q["partition"] != "fit"}
    for r in held_refs_changed:
        if r["example_question"] in held_text:
            r["answers"] = {"a": "corrupted", "b": "corrupted"}
    fitted2 = learn_direction(
        model,
        tok,
        cfg,
        manifest,
        held_refs_changed,
        layers=[1],
        prefix_tokens=2,
        out_dir=tmp_path / "fit",
    )
    assert fitted == fitted2
    batch = [{"question": "normal question"}, {"question": "question"}]
    plain, _ = generate_batch(model, tok, batch, method="plain", max_new_tokens=4, review_tokens=2)
    zero, _ = generate_batch(
        model,
        tok,
        batch,
        method="steering",
        max_new_tokens=4,
        review_tokens=2,
        spec=SteerSpec(1, torch.ones(32), 0),
    )
    assert plain == zero


def test_test_partition_requires_dev_selection_before_model_load(tmp_path):
    from scripts.run_pilot import generate

    qs, refs = data()
    rows = prepare_rows(qs, refs)
    m = make_manifest(rows, seed=17, folds=5, test_fold=0, dev_fold=1, reference_hash="x")
    frozen_json(tmp_path / "manifest.json", m)
    args = SimpleNamespace(
        config="configs/gemma2_9b.yaml",
        manifest=tmp_path / "manifest.json",
        method="plain",
        partition="test",
        selection=None,
    )
    with pytest.raises(ValueError, match="Test is locked"):
        generate(args)


@pytest.mark.parametrize("method", ["fp_identification", "premise_cot"])
def test_two_stage_baselines_save_review_but_return_only_answer(tiny_gemma, monkeypatch, method):
    import torch
    from src.pilot_model import generate_batch

    model, tok = tiny_gemma
    calls = []

    def fake_generate(input_ids, attention_mask, **kwargs):
        calls.append(kwargs["max_new_tokens"])
        word = "Yes" if len(calls) == 1 else "answer"
        suffix = torch.tensor(
            [[tok.convert_tokens_to_ids(word), tok.eos_token_id]] * len(input_ids)
        )
        return torch.cat([input_ids, suffix], dim=1)

    monkeypatch.setattr(model, "generate", fake_generate)
    texts, records = generate_batch(
        model,
        tok,
        [{"question": "normal question", "correction": "SECRET_GOLD"}],
        method=method,
        max_new_tokens=32,
        review_tokens=12,
    )
    assert texts == ["answer"] and records[0]["review"] == "Yes"
    assert calls == [16 if method == "fp_identification" else 12, 32]
    assert records[0]["review_output_tokens"] == 2


def test_judge_retries_parse_failure_and_refuses_backend_cache_change(tmp_path, monkeypatch):
    import scripts.run_judge as runner
    import yaml

    q = {"id": "f", "set": "fpq", "question": "question", "correction": "correct"}
    questions, responses = tmp_path / "questions.jsonl", tmp_path / "responses.jsonl"
    output = tmp_path / "judge.jsonl"
    write_jsonl(questions, [q])
    write_jsonl(
        responses, [{**q, "response": "final answer", "model_id": "gemma", "run_hash": "run"}]
    )
    examples = tmp_path / "examples.json"
    examples.write_text(
        json.dumps(
            [
                {
                    "example_question": "example",
                    "example_assumption": "fact",
                    "answer": "answer",
                    "score": {"Sharpness": 1, "Reason": "ok"},
                }
            ]
        )
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "judge": {
                    "backend": "openai",
                    "model": "gpt-4o",
                    "examples_fpq": str(examples),
                    "examples_nfp": str(examples),
                }
            }
        )
    )
    replies = iter([("invalid", "gpt-4o"), ('{\n"Sharpness": -1, "Reason": "fails"\n}', "gpt-4o")])
    monkeypatch.setattr(runner, "make_caller", lambda *a, **kw: lambda prompt: next(replies))
    argv = [
        "run_judge",
        "--config",
        str(config),
        "--questions",
        str(questions),
        "--responses",
        str(responses),
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", argv)
    runner.main()
    scored = list(read_jsonl(output))
    assert len(scored) == 1 and scored[0]["sharpness"] == -1 and scored[0]["judge_parsed"]
    assert scored[0]["response_text_hash"] == digest("final answer")
    # CLI normally exits, releasing the atexit lock. Simulate that here.
    Path(str(output) + ".lock").unlink()
    monkeypatch.setattr("sys.argv", argv + ["--backend", "codex", "--model", "some-other-model"])
    with pytest.raises(ValueError, match="mismatch"):
        runner.main()


def test_selection_and_reporting_use_dev_only_and_complete_matching_scores(tmp_path):
    from scripts.run_pilot import report, select

    qs, refs = data()
    rows = prepare_rows(qs, refs)
    m = make_manifest(rows, seed=17, folds=5, test_fold=0, dev_fold=1, reference_hash="raw")
    frozen_json(tmp_path / "manifest.json", m)
    expected = [q for q in rows if q["partition"] == "dev"]
    identity = {"source_model": {"model_id": "tiny"}}
    paths = {}
    for method in ("plain", "steering"):
        run = {
            "method": method,
            "partition": "dev",
            "manifest_hash": m["manifest_hash"],
            "identity": identity,
            "max_new_tokens": 512,
            "review_tokens": 128,
        }
        if method == "steering":
            run.update(layer=14, alpha=0.05, direction_hash="d")
        path = tmp_path / (method + ".jsonl")
        scored = [
            {
                **score(q["id"], q["set"], 1 if q["set"] == "nfp" or method == "steering" else -1),
                "response_run_hash": digest(run),
            }
            for q in expected
        ]
        write_jsonl(path, scored)
        frozen_json(str(path) + ".run.json", {"response_run": run})
        paths[method] = path
    selection_path = tmp_path / "selection.json"
    args = SimpleNamespace(
        manifest=tmp_path / "manifest.json",
        plain_scores=paths["plain"],
        candidate_scores=[paths["steering"]],
        nfp_margin_pp=5,
        output=selection_path,
    )
    select(args)
    assert json.loads(selection_path.read_text())["selected"]["layer"] == 14
    report(
        SimpleNamespace(
            manifest=args.manifest,
            partition="dev",
            scores=list(paths.values()),
            output=tmp_path / "report.json",
        )
    )
    assert (tmp_path / "report.md").exists()
    # A test file must never be used to select dev settings.
    meta = json.loads(Path(str(paths["steering"]) + ".run.json").read_text())
    meta["response_run"]["partition"] = "test"
    Path(str(paths["steering"]) + ".run.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="not a dev"):
        select(args)
