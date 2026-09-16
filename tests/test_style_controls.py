"""Style-confound controls: content-blind readouts, claude backend, paraphrase
checks, matched operating points, and the three-condition driver."""

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from src import baseline_gates as gates
from src import baseline_suite as suite
from src import llm_backend
from src import style_controls as sc
from src.paraphrase import FIDELITY_PROMPT, PARAPHRASE_PROMPT, PREMISE_KEEP_PROMPT, RETRY_PROMPT, make_paraphrase, word_jaccard
from src.style_features import FUNCTION_WORDS, mask_content, style_feature_names, style_vector


# --- content-blind features -------------------------------------------------

def test_style_vector_is_finite_named_and_blind_to_content_words():
    names = style_feature_names()
    a = style_vector("I heard that turmeric cures colon cancer. Is that true?")
    b = style_vector("I heard that cinnamon stops colon cancer. Is that true?")
    assert a.shape == (len(names),) and np.isfinite(a).all()
    # Same register, different content words: identical style vector.
    assert np.allclose(a, b)
    c = style_vector("What is the recommended screening interval for someone with a family history?")
    assert not np.allclose(a, c)
    assert style_vector("").shape == (len(names),)


def test_mask_content_keeps_function_words_and_masks_absolutes():
    masked = mask_content("Does turmeric always cure my colon cancer? I read 100% of cases.")
    assert masked == "does _ _ _ my _ _ ? i _ 0 % of _ ."
    assert "always" not in FUNCTION_WORDS and "cure" not in FUNCTION_WORDS
    assert mask_content("I can't sleep") == "i _n't _"


# --- claude -p backend ------------------------------------------------------

def _result(**kw):
    base = {"type": "result", "subtype": "success", "is_error": False, "result": "YES",
            "modelUsage": {"claude-x-1": {"inputTokens": 1}}}
    base.update(kw)
    return json.dumps(base)


def test_parse_claude_result_reads_reply_and_served_model():
    answer, model = llm_backend.parse_claude_result("banner line\n" + _result(), "requested")
    assert (answer, model) == ("YES", "claude-x-1")
    _, model = llm_backend.parse_claude_result(_result(modelUsage={}), "requested")
    assert model == "requested"
    with pytest.raises(RuntimeError):
        llm_backend.parse_claude_result(_result(is_error=True, result="Authentication error"), "m")
    with pytest.raises(RuntimeError):
        llm_backend.parse_claude_result(_result(result=""), "m")
    with pytest.raises(RuntimeError):
        llm_backend.parse_claude_result("not json", "m")


def test_parse_claude_result_ignores_background_haiku_usage():
    import json
    payload = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "Rating: 4",
                          "modelUsage": {"claude-haiku-4-5-20251001": {"inputTokens": 3}, "claude-sonnet-5": {"inputTokens": 900}}})
    assert llm_backend.parse_claude_result(payload, "claude-sonnet-5") == ("Rating: 4", "claude-sonnet-5")
    dated = payload.replace('"claude-sonnet-5"', '"claude-sonnet-5-20260601"')
    assert llm_backend.parse_claude_result(dated, "claude-sonnet-5") == ("Rating: 4", "claude-sonnet-5-20260601")
    # Requested model absent from usage: the mix is reported, not guessed.
    assert llm_backend.parse_claude_result(payload, "claude-opus-5")[1] == "claude-haiku-4-5-20251001,claude-sonnet-5"


def test_run_claude_uses_print_mode_without_tools(monkeypatch):
    seen = {}

    def fake_run(cmd, input, capture_output, text, timeout):
        seen.update(cmd=cmd, prompt=input, timeout=timeout)
        return SimpleNamespace(returncode=0, stdout=_result(), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    call = llm_backend.make_caller("claude", "sonnet", timeout=42, claude_cmd="claude")
    assert call("hello") == ("YES", "claude-x-1")
    cmd = seen["cmd"]
    assert cmd[0] == "claude" and "-p" in cmd and "--output-format" in cmd and "json" in cmd
    assert cmd[cmd.index("--tools") + 1] == "" and "--no-session-persistence" in cmd
    assert "--bare" not in cmd and "--system-prompt" in cmd  # --bare never reads the subscription login
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert seen["prompt"] == "hello" and seen["timeout"] == 42

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="boom"))
    with pytest.raises(RuntimeError):
        llm_backend.run_claude("x", "", 5)


def test_run_claude_retries_transient_token_refresh_then_gives_up(monkeypatch):
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)
    transient = '{"is_error":true,"result":"another Claude Code process is refreshing it or exited mid-refresh. This is usually transient; retry in a minute","type":"result"}'
    outcomes = iter([SimpleNamespace(returncode=1, stdout=transient, stderr=""),
                     SimpleNamespace(returncode=0, stdout=_result(), stderr="")])
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (calls.append(1), next(outcomes))[1])
    assert llm_backend.run_claude("x", "sonnet", 5) == ("YES", "claude-x-1")
    assert len(calls) == 2
    # Non-transient failure (usage limit) is not retried.
    calls.clear()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (calls.append(1), SimpleNamespace(returncode=1, stdout="", stderr="usage limit reached"))[1])
    with pytest.raises(RuntimeError, match="usage limit"):
        llm_backend.run_claude("x", "sonnet", 5)
    assert len(calls) == 1
    # Persistent transient failure gives up after the attempt cap.
    calls.clear()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (calls.append(1), SimpleNamespace(returncode=1, stdout=transient, stderr=""))[1])
    with pytest.raises(RuntimeError, match="transient"):
        llm_backend.run_claude("x", "sonnet", 5)
    assert len(calls) == llm_backend.CLAUDE_TRANSIENT_ATTEMPTS


def test_backend_available_checks_claude_binary(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
    assert llm_backend.backend_available("claude")
    assert not llm_backend.backend_available("claude", claude_cmd="claude-missing")
    assert not llm_backend.backend_available("codex")


# --- paraphrases ------------------------------------------------------------

def _scripted(replies):
    log = []

    def llm(prompt):
        log.append(prompt)
        for key, reply in replies:
            if key in prompt:
                return reply
        raise AssertionError("unexpected prompt")

    llm.log = log
    return llm


FPQ = {"id": "fpq_1", "set": "fpq", "group_id": "g1", "partition": "fit",
       "question": "I heard turmeric cures colon cancer, so can I skip chemo?",
       "premise_text": "Turmeric cures colon cancer.", "correction": "It does not."}


def test_paraphrase_accepts_only_rewrites_that_keep_premise_and_meaning():
    llm = _scripted([("ORIGINAL:", "YES"), ("BELIEF:", "YES"),
                     ("MESSAGE:", "Since turmeric cures colon cancer, is it fine to skip chemotherapy?")])
    row, status, audit = make_paraphrase(FPQ, llm, writer="claude:test")
    assert status == "ok" and row["id"] == "fpq_1_para" and row["paraphrase_of"] == "fpq_1"
    assert row["set"] == "fpq" and row["writer"] == "claude:test" and audit["premise_checked"]
    assert 0 < audit["jaccard"] < 1
    assert any(p.startswith(PARAPHRASE_PROMPT[:30]) for p in llm.log)
    assert any(p.startswith(PREMISE_KEEP_PROMPT[:30]) for p in llm.log)
    assert any(p.startswith(FIDELITY_PROMPT[:30]) for p in llm.log)

    lost = _scripted([("BELIEF:", "NO"), ("MESSAGE:", "Is it fine to skip chemotherapy?")])
    row, status, audit = make_paraphrase(FPQ, lost)
    assert status == "lost_premise" and audit["rewrite"] == "Is it fine to skip chemotherapy?"
    assert audit["premise_verdict"] == "NO"
    unfaithful = _scripted([("ORIGINAL:", "NO"), ("BELIEF:", "YES"),
                            ("MESSAGE:", "Because turmeric eliminates colon cancer, I plan to stop chemotherapy, right?")])
    assert make_paraphrase(FPQ, unfaithful)[1] == "not_faithful"
    same = _scripted([("MESSAGE:", FPQ["question"].upper())])
    assert make_paraphrase(FPQ, same)[1] == "unchanged"
    long = _scripted([("MESSAGE:", "x " * 200)])
    assert make_paraphrase(FPQ, long)[1] == "too_long"
    assert make_paraphrase(FPQ, _scripted([("MESSAGE:", "")]))[1] == "empty"

    # Near copy: retried once with the retry prompt; still a copy -> dropped.
    copy = _scripted([("MESSAGE:", "I heard turmeric cures colon cancer, so can I skip the chemo?")])
    row, status, audit = make_paraphrase(FPQ, copy)
    assert status == "near_copy" and audit["retried"] and audit["jaccard"] > 0.6
    assert any(p.startswith(RETRY_PROMPT[:30]) for p in copy.log)
    fixed = _scripted([("ORIGINAL:", "YES"), ("BELIEF:", "YES"),
                       ("NEW REWRITE:", "Since turmeric gets rid of colon cancer, is chemotherapy something I can avoid?"),
                       ("MESSAGE:", "I heard turmeric cures colon cancer, so can I skip the chemo?")])
    row, status, audit = make_paraphrase(FPQ, fixed)
    assert status == "ok" and audit["retried"] and audit["jaccard"] <= 0.6

    nfp = {"id": "nfp_1", "set": "nfp", "group_id": "g2", "partition": "fit",
           "question": "How often should I get a colonoscopy after 50?"}
    llm = _scripted([("ORIGINAL:", "YES"), ("MESSAGE:", "After turning 50, how frequently is a colonoscopy needed?")])
    row, status, audit = make_paraphrase(nfp, llm)
    assert status == "ok" and row["set"] == "nfp" and not audit["premise_checked"]
    assert not any(p.startswith(PREMISE_KEEP_PROMPT[:30]) for p in llm.log)
    assert word_jaccard("a b c", "a b d") == pytest.approx(0.5)


# --- matched operating points ----------------------------------------------

def test_matched_operating_point_flags_same_negative_share_per_fold():
    rng = np.random.default_rng(3)
    preds = []
    for fold in range(2):
        for i in range(40):
            preds.append({"id": f"n{fold}{i}", "group_id": f"n{fold}{i}", "set": "nfp", "fold": fold,
                          "signal": "s", "score": float(rng.normal()), "threshold": 0, "gate_on": None})
        for i in range(60):
            preds.append({"id": f"p{fold}{i}", "group_id": f"p{fold}{i}", "set": "fpq", "fold": fold,
                          "signal": "s", "score": float(rng.normal() + 1), "threshold": 0, "gate_on": None})
    m = gates.matched_operating_point(preds, 0.10)
    assert m["fp"] == 2 * 4 and m["nfp_n"] == 80 and m["fpq_n"] == 120 and 0 < m["tpr"] <= 1
    assert gates.matched_operating_point(preds, 0.0)["fp"] == 0
    p = gates.partial_auroc(preds, 0.1)
    assert 0 <= p <= 1
    with pytest.raises(ValueError):
        gates.matched_operating_point(preds, 1.0)
    with pytest.raises(ValueError):
        gates.partial_auroc(preds, 0)
    text, summaries = gates.matched_report({"predictions": preds}, repeats=20)
    assert "Matched operating points" in text and summaries["s"]["matched"]["0.1"]["ci"] is not None


def test_gate_comparison_runs_content_blind_signals(gate_rows):
    rows, split = gate_rows
    result = gates.run_gate_comparison(rows, split, signals=("text", "style", "masked"), c_grid=(0.1, 1.0))
    assert {p["signal"] for p in result["predictions"]} == {"text", "style", "masked"}
    assert all(s["best"]["layer"] is None for s in result["selections"])
    text, summaries = gates.gate_report(result, repeats=5)
    assert "| style |" in text and "| masked |" in text and "Matched operating points" in text
    assert "matched" in summaries["style"]


@pytest.fixture
def gate_rows():
    rows = []
    for g in range(60):
        for kind in ("fpq", "nfp"):
            rows.append({"id": f"{kind}_{g:03}", "set": kind, "group_id": f"source_{g:03}",
                         "question": (f"I heard that remedy {g} cures cancer, is it true?" if kind == "fpq"
                                      else f"What is the screening interval for condition {g}?"),
                         "partition": "fit" if g < 40 else "dev" if g < 50 else "test",
                         "correction": "correct statement" if kind == "fpq" else ""})
    rows = suite.check_questions(rows)
    return rows, suite.split_plan(rows)


# --- three-condition driver --------------------------------------------------

@pytest.fixture
def controls_data():
    rng = np.random.default_rng(7)
    natural, twins, para, fparas, features = [], [], [], [], {}
    for g in range(60):
        fpq_id, nfp_id = f"fpq_{g:03}", f"nfp_{g:03}"
        natural.append({"id": fpq_id, "set": "fpq", "group_id": f"src_{g:03}", "partition": "fit",
                        "question": f"I heard that herb {g} cures cancer, so is chemo unnecessary?",
                        "correction": "no"})
        natural.append({"id": nfp_id, "set": "nfp", "group_id": f"src_{g + 100:03}", "partition": "fit",
                        "question": f"What screening schedule applies to condition {g}?",
                        "from_model": "gpt-4o" if g % 3 == 0 else "gemini"})
        if g % 2 == 0:
            twins.append({"id": fpq_id + "_true", "set": "tpair", "pair_id": fpq_id,
                          "question": f"I heard that herb {g} does not cure cancer, so is chemo unnecessary?"})
        if g % 4 == 0:
            fparas.append({"id": fpq_id + "_fpara", "set": "fpara", "label_false_premise": 1, "pair_id": fpq_id,
                           "question": f"I heard that herb {g} wipes out cancer, so is chemo unnecessary?"})
        para.append({"id": fpq_id + "_para", "set": "fpq", "paraphrase_of": fpq_id,
                     "question": f"Herb {g} cures cancer; is chemo needed?"})
        para.append({"id": nfp_id + "_para", "set": "nfp", "paraphrase_of": nfp_id,
                     "question": f"Condition {g}; which screening schedule is needed?"})
        for row_id, y in ((fpq_id, 1), (nfp_id, -1), (fpq_id + "_para", 1), (nfp_id + "_para", -1)):
            features[row_id] = {11: np.array([2.0 * y, *rng.normal(size=3)])}
    return natural, twins, para, fparas, features


def test_conditions_cover_every_row_once_without_origin_leakage(controls_data):
    natural, twins, para, fparas, features = controls_data
    tw_rows, fp_rows = sc.split_twin_file(twins + fparas)
    assert len(tw_rows) == 30 and len(fp_rows) == 15
    nat, tw, pa, fp = sc.assemble(natural, tw_rows, para, fp_rows)
    assignment = sc.fold_assignment(natural, folds=3, seed=1)
    result = sc.run_conditions(nat, tw, pa, fp, assignment=assignment, signals=("text", "style", "hidden"),
                               features=features, layers=(11,), c_grid=(1.0,), seed=1)
    preds = result["predictions"]
    conditions = {p["condition"] for p in preds}
    assert conditions == set(sc.CONDITIONS)
    # Every evaluation row is scored exactly once per condition/signal.
    for name in ("natural", "twins", "para", "natural->twins"):
        for sig in ("text", "style"):
            ids = [p["id"] for p in preds if p["condition"] == name and p["signal"] == sig]
            assert len(ids) == len(set(ids))
    assert len([p for p in preds if p["condition"] == "twins" and p["signal"] == "text"]) == 60
    same = [p for p in preds if p["condition"] == "same_writer" and p["signal"] == "text"]
    assert sum(1 - p["label"] for p in same) == 20 and sum(p["label"] for p in same) == 60
    assert len([p for p in preds if p["condition"] == "natural->same_writer" and p["signal"] == "text"]) == 80
    edited = [p for p in preds if p["condition"] == "edited" and p["signal"] == "text"]
    assert len(edited) == 30 and sum(p["label"] for p in edited) == 15
    assert all(p["source"] in ("fparas", "twins") for p in edited)
    assert len([p for p in preds if p["condition"] == "para" and p["signal"] == "text"]) == 120
    # Rewrites inherit the fold: an origin is never in train and evaluation of one fold.
    for p in preds:
        assert p["fold"] == assignment[p["origin"]]
    # Hidden skipped where twins lack features, run where every row has them.
    skipped = {(s["condition"], s.get("signal")) for s in result["skipped"]}
    assert ("twins", "hidden") in skipped and ("natural->twins", "hidden") in skipped and ("edited", "hidden") in skipped
    assert any(p["condition"] == "para" and p["signal"] == "hidden" for p in preds)
    summaries = sc.summarize(result, repeats=10, seed=1)
    assert summaries["natural"]["signals"]["text"]["auroc"] > 0.9
    assert summaries["natural"]["signals"]["hidden"]["auroc"] > 0.9
    assert "hidden" in summaries["natural"]["delta_vs_text"] and summaries["natural"]["delta_vs_text"]["hidden"]["ci"]
    text = sc.report(result, summaries)
    assert "| natural | text |" in text and "| twins | style |" in text and "| edited | text |" in text and "Skipped:" in text


def test_assemble_rejects_mislabeled_rewrites(controls_data):
    natural, twins, para, fparas, _ = controls_data
    with pytest.raises(ValueError):
        sc.assemble(natural, [{**twins[0], "set": "nfp"}], [])
    with pytest.raises(ValueError):
        sc.assemble(natural, [], [{**para[0], "set": "nfp"}])
    with pytest.raises(ValueError):
        sc.assemble(natural, [{**twins[0], "pair_id": "nfp_000"}], [])
    with pytest.raises(ValueError):
        sc.assemble(natural, [{**twins[0], "id": natural[0]["id"]}], [])
    with pytest.raises(ValueError):
        sc.assemble(natural, [], [], [{**fparas[0], "label_false_premise": 0}])
    with pytest.raises(ValueError):
        sc.split_twin_file([{"id": "x", "set": "weird", "pair_id": "fpq_000", "question": "q"}])
    assert sc.condition_rows("twins", *sc.assemble(natural, [], [])) is None
    assert sc.condition_rows("edited", *sc.assemble(natural, twins, [], [])) is None


def test_twin_remap_keeps_row_ids_for_feature_lookup():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("rsc", root / "scripts" / "run_style_controls.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rows = [{"id": "fpq_2_true_ab12", "pair_id": "e1_fpq_2", "question": "x"},
            {"id": "fpq_9_true_cd34", "pair_id": "e1_missing", "question": "y"}]
    out = mod.remap_twins(rows, {"q two": "fpq_2"}, {"e1_fpq_2": "q two"}, "true")
    assert out == [{"id": "fpq_2_true_ab12", "pair_id": "fpq_2", "question": "x"}]
