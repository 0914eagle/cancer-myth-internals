import json
from pathlib import Path

import numpy as np
import pytest

from scripts import pilot_gate as mod
from src.pilot import digest, make_manifest, file_digest
from src.jsonl import load_json, write_jsonl


def test_threshold_respects_ties_and_small_calibration_set():
    s = np.array([0.1] * 18 + [0.9] * 2)
    t = mod.threshold_at_fpr(s)
    assert sum(s > t) == 0  # cannot split tied high scores to hit 1/20
    s = np.arange(30)
    assert sum(s > mod.threshold_at_fpr(s)) == 1
    s = np.arange(18)
    assert sum(s > mod.threshold_at_fpr(s)) == 0
    with pytest.raises(ValueError):
        mod.threshold_at_fpr([])


@pytest.fixture
def data(tmp_path):
    pilot, out = tmp_path / "pilot", tmp_path / "gate"
    (pilot / "split").mkdir(parents=True)
    (pilot / "dev").mkdir()
    (out / "cache").mkdir(parents=True)
    qs = []
    for part, n in [("fit", 120), ("dev", 60), ("test", 12)]:
        for i in range(n):
            kind = "fpq" if i % 2 else "nfp"
            qs.append(
                {
                    "id": f"{part}{i:03}",
                    "set": kind,
                    "partition": part,
                    "question": f"{'false' if kind == 'fpq' else 'normal'} question {part} {i}",
                    "correction": "correct fact",
                    "hallucination_text": "invented objection",
                    "group_id": f"{part}-g{i // 2}",
                }
            )
    manifest = make_manifest(qs, seed=17, folds=5, test_fold=0, dev_fold=1, reference_hash="ref")
    (pilot / "split/manifest.json").write_text(json.dumps(manifest))
    dev = [q for q in qs if q["partition"] == "dev"]
    write_jsonl(pilot / "split/dev.jsonl", dev)
    rows, split = mod.split_fit(manifest)
    spec = {
        "version": mod.VERSION,
        "identity": {},
        "manifest_hash": manifest["manifest_hash"],
        "question_hash": digest(rows),
        "split": split,
    }
    (out / "features.json").write_text(json.dumps(spec))
    rng = np.random.default_rng(19)
    for q in rows:
        x = rng.normal(size=6).astype("float32")
        x[0] += 5 if q["set"] == "fpq" else -5
        np.savez(out / "cache" / f"{digest(q['id'])}.npz", x=x, qid=q["id"], plan_hash=digest(spec))
    return pilot, out, manifest


def test_fit_never_uses_dev_to_train_or_calibrate(data, tmp_path):
    pilot, out, manifest = data
    rows, split = mod.split_fit(manifest)
    groups = {q["id"]: q["group_id"] for q in rows}
    sets = [{groups[i] for i in split[key]} for key in ("train_ids", "calibration_ids", "dev_ids")]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    assert not any("test" in q["id"] for q in rows)
    mod.fit(out, pilot / "split/manifest.json")
    a = load_json(out / "gate.json")
    parameters = load_json(out / "parameters.json")
    import shutil

    other = tmp_path / "changed-dev"
    shutil.copytree(out, other)
    (other / "gate.json").unlink()
    (other / "parameters.json").unlink()
    for qid in split["dev_ids"]:
        p = other / "cache" / f"{digest(qid)}.npz"
        with np.load(p) as z:
            np.savez(p.with_suffix(".new.npz"), x=z["x"] * -100, qid=qid, plan_hash=z["plan_hash"])
        p.with_suffix(".new.npz").replace(p)
    mod.fit(other, pilot / "split/manifest.json")
    b = load_json(other / "gate.json")
    assert load_json(other / "parameters.json") == parameters
    for name in ("hidden", "text"):
        assert a["gates"][name]["threshold"] == b["gates"][name]["threshold"]
    assert any(
        a["gates"]["hidden"]["scores"][i] != b["gates"]["hidden"]["scores"][i]
        for i in split["dev_ids"]
    )


def test_report_replays_scores_without_model_calls(data, monkeypatch):
    from scripts import reevaluate_pilot_nfp as reeval
    from scripts.check_terra_judge import MODEL, PROTOCOL_V2

    pilot, out, manifest = data
    mod.fit(out, pilot / "split/manifest.json")
    dev = [q for q in manifest["questions"] if q["partition"] == "dev"]
    qpath = pilot / "split/dev.jsonl"
    for method in reeval.METHODS:
        run = {
            "method": method,
            "partition": "dev",
            "manifest_hash": manifest["manifest_hash"],
            "identity": {},
        }
        path = pilot / "dev" / f"{method}.jsonl"
        answers = [
            {
                "id": q["id"],
                "set": q["set"],
                "question": q["question"],
                "run_hash": digest(run),
                "response": "objection" if method == "fp_identification" else "same answer",
            }
            for q in dev
        ]
        write_jsonl(path, answers)
        Path(str(path) + ".run.json").write_text(json.dumps(run))
        signature = {
            "requested_model": "gpt-5.6-sol",
            "response_run": run,
            "responses_hash": file_digest(path),
            "questions_hash": file_digest(qpath),
            "backend": "codex",
            "temperature": None,
        }
        sp = pilot / "dev" / f"{method}_judge_codex_gpt-5.6-sol.jsonl"
        scored = []
        for q, ans in zip(dev, answers):
            scored.append(
                {
                    "id": q["id"] + "::" + q["id"],
                    "question_id": q["id"],
                    "response_id": q["id"],
                    "set": q["set"],
                    "sharpness": 1 if method != "plain" else -1,
                    "judge_parsed": True,
                    "judge_run_hash": digest(signature),
                    "response_run_hash": digest(run),
                    "judge_model": "gpt-5.6-sol",
                    "judge_backend": "codex",
                    "judge_temperature": None,
                    "rubric": q["set"],
                    "response_text_hash": digest(ans["response"]),
                }
            )
        write_jsonl(sp, scored)
        Path(str(sp) + ".run.json").write_text(json.dumps(signature))
    source = out / "protocol-source"
    source.mkdir()
    template = (mod.ROOT / "prompts/nfp_role_clarified_v2.txt").read_text()
    base = {
        "protocol": PROTOCOL_V2,
        "model": MODEL,
        "backend": "codex",
        "cases": [],
        "prompt_template": template,
        "prompt_template_hash": digest(template),
        "transport_hash": file_digest(mod.ROOT / "src/llm_backend.py"),
    }
    (source / "plan.json").write_text(json.dumps(base))
    (source / "human_review.tsv").write_text("test source")
    (source / "run.json").write_text(
        json.dumps(
            {"plan_hash": digest(base), "human_hash": file_digest(source / "human_review.tsv")}
        )
    )
    nfp = out / "nfp"
    reeval.prepare(pilot, source, nfp)

    def fake(prompt):
        bad = "<ANSWER>\nobjection\n</ANSWER>" in prompt
        return json.dumps(
            {
                "Sharpness": -1 if bad else 1,
                "Reason": "reason",
                "AnswerEvidence": "objection" if bad else "",
                "InventedPremise": "invented" if bad else "",
            }
        ), MODEL

    monkeypatch.setattr(reeval, "make_caller", lambda *a, **k: fake)
    reeval.score(nfp)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("report is offline"))
    mod.report(pilot, out, nfp)
    result = load_json(out / "report.json")
    assert (
        len(result["identical_answer_sharing"]) == 30
    )  # CoT exact same FPQ answer uses Plain score
    assert result["summary"]["premise CoT"]["fpq"]["pass"] == 0
    assert result["summary"]["always FP Identification"]["fpq"]["pass"] == 100
    assert result["summary"]["always FP Identification"]["nfp"]["pass"] == 0
    assert len(result["random_draw_metrics"]) == 500
    # Deterministic report, no further scoring.
    mod.report(pilot, out, nfp)


def test_group_leakage_is_rejected(data):
    _, _, manifest = data
    fit_group = next(q["group_id"] for q in manifest["questions"] if q["partition"] == "fit")
    next(q for q in manifest["questions"] if q["partition"] == "dev")["group_id"] = fit_group
    with pytest.raises(ValueError, match="group-disjoint"):
        mod.split_fit(manifest)


def test_extraction_uses_question_last_token_and_resumes(data, monkeypatch):
    import torch
    from types import SimpleNamespace
    from src import pilot_model, modeling

    pilot, out, _ = data
    fresh = out.parent / "extract-check"
    identity = {
        "source_model": {"model_id": "google/gemma-2-9b-it"},
        "resolved_revision": "r",
        "tokenizer_revision": "t",
        "chat_template_hash": "h",
    }
    (pilot / "dev/plain.jsonl.run.json").write_text(json.dumps({"identity": identity}))
    calls = []

    class Block(torch.nn.Module):
        def forward(self, h):
            return h + 1

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block() for _ in range(42)])

        def forward(self, input_ids, **kwargs):
            calls.append(input_ids.tolist())
            assert kwargs["use_cache"] is False
            h = input_ids.float().unsqueeze(-1).repeat(1, 1, 6)
            for layer in self.layers:
                h = layer(h)
            return SimpleNamespace(last_hidden_state=h)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 1 and messages[0]["role"] == "user"
            assert "correction" not in messages[0]["content"]
            return messages[0]["content"]

        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    model = SimpleNamespace(
        base_model=Base(),
        device=torch.device("cpu"),
        config=SimpleNamespace(hidden_size=6, max_position_embeddings=100),
    )
    monkeypatch.setattr(pilot_model, "load_model", lambda cfg: (model, Tokenizer()))
    monkeypatch.setattr(pilot_model, "model_identity", lambda *args: identity)
    monkeypatch.setattr(modeling, "decoder_layers", lambda m: m.base_model.layers)
    mod.extract(pilot, fresh, "configs/gemma2_9b.yaml")
    assert len(calls) == 180  # fit+dev only; test untouched
    assert len(model.base_model.layers[20]._forward_hooks) == 0
    p = next((fresh / "cache").glob("*.npz"))
    with np.load(p) as z:
        assert (z["x"] == 24).all()  # token 3 + 21 blocks; not final output 45
    mod.extract(pilot, fresh, "configs/gemma2_9b.yaml")
    assert len(calls) == 180
