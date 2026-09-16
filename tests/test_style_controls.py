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
from src.paraphrase import FIDELITY_PROMPT, PARAPHRASE_PROMPT, RETRY_PROMPT, make_paraphrase, word_jaccard
from src.rows import TWIN_CHECK_PROMPT
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
    assert any(p.startswith(TWIN_CHECK_PROMPT[:30]) for p in llm.log)
    assert any(p.startswith(FIDELITY_PROMPT[:30]) for p in llm.log)

    lost = _scripted([("BELIEF:", "NO"), ("MESSAGE:", "Is it fine to skip chemotherapy?")])
    assert make_paraphrase(FPQ, lost)[1] == "lost_premise"
    unfaithful = _scripted([("ORIGINAL:", "NO"), ("BELIEF:", "YES"),
                            ("MESSAGE:", "Turmeric cures colon cancer so I will skip chemo, right?")])
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
    assert not any(p.startswith(TWIN_CHECK_PROMPT[:30]) for p in llm.log)
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
    natural, twins, para, features = [], [], [], {}
    for g in range(60):
        fpq_id, nfp_id = f"fpq_{g:03}", f"nfp_{g:03}"
        natural.append({"id": fpq_id, "set": "fpq", "group_id": f"src_{g:03}", "partition": "fit",
                        "question": f"I heard that herb {g} cures cancer, so is chemo unnecessary?",
                        "correction": "no"})
        natural.append({"id": nfp_id, "set": "nfp", "group_id": f"src_{g + 100:03}", "partition": "fit",
                        "question": f"What screening schedule applies to condition {g}?"})
        if g % 2 == 0:
            twins.append({"id": fpq_id + "_true", "set": "tpair", "pair_id": fpq_id,
                          "question": f"I heard that herb {g} does not cure cancer, so is chemo unnecessary?"})
        para.append({"id": fpq_id + "_para", "set": "fpq", "paraphrase_of": fpq_id,
                     "question": f"Herb {g} cures cancer; is chemo needed?"})
        para.append({"id": nfp_id + "_para", "set": "nfp", "paraphrase_of": nfp_id,
                     "question": f"Condition {g}; which screening schedule is needed?"})
        for row_id, y in ((fpq_id, 1), (nfp_id, -1), (fpq_id + "_para", 1), (nfp_id + "_para", -1)):
            features[row_id] = {11: np.array([2.0 * y, *rng.normal(size=3)])}
    return natural, twins, para, features


def test_conditions_cover_every_row_once_without_origin_leakage(controls_data):
    natural, twins, para, features = controls_data
    nat, tw, pa = sc.assemble(natural, twins, para)
    assignment = sc.fold_assignment(natural, folds=3, seed=1)
    result = sc.run_conditions(nat, tw, pa, assignment=assignment, signals=("text", "style", "hidden"),
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
    assert len([p for p in preds if p["condition"] == "para" and p["signal"] == "text"]) == 120
    # Rewrites inherit the fold: an origin is never in train and evaluation of one fold.
    for p in preds:
        assert p["fold"] == assignment[p["origin"]]
    # Hidden skipped where twins lack features, run where every row has them.
    skipped = {(s["condition"], s.get("signal")) for s in result["skipped"]}
    assert ("twins", "hidden") in skipped and ("natural->twins", "hidden") in skipped
    assert any(p["condition"] == "para" and p["signal"] == "hidden" for p in preds)
    summaries = sc.summarize(result, repeats=10, seed=1)
    assert summaries["natural"]["signals"]["text"]["auroc"] > 0.9
    assert summaries["natural"]["signals"]["hidden"]["auroc"] > 0.9
    assert "hidden" in summaries["natural"]["delta_vs_text"] and summaries["natural"]["delta_vs_text"]["hidden"]["ci"]
    text = sc.report(result, summaries)
    assert "| natural | text |" in text and "| twins | style |" in text and "Skipped:" in text


def test_assemble_rejects_mislabeled_rewrites(controls_data):
    natural, twins, para, _ = controls_data
    with pytest.raises(ValueError):
        sc.assemble(natural, [{**twins[0], "set": "nfp"}], [])
    with pytest.raises(ValueError):
        sc.assemble(natural, [], [{**para[0], "set": "nfp"}])
    with pytest.raises(ValueError):
        sc.assemble(natural, [{**twins[0], "pair_id": "nfp_000"}], [])
    with pytest.raises(ValueError):
        sc.assemble(natural, [{**twins[0], "id": natural[0]["id"]}], [])
    assert sc.condition_rows("twins", *sc.assemble(natural, [], [])) is None
