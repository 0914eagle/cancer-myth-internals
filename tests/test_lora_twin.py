"""Offline tests for the twin LoRA data/eval logic: no torch, peft, GPU or judge."""
import json
from pathlib import Path

import pytest

from src import lora_twin as lt
from src.pilot import digest


SUITE = [
    {"id": "fpq_1", "set": "fpq", "partition": "fit", "group_id": "g1", "question": "Does sugar feed my tumor?", "correction": "c"},
    {"id": "fpq_2", "set": "fpq", "partition": "fit", "group_id": "g2", "question": "Since chemo always fails, what now?", "correction": "c"},
    {"id": "fpq_3", "set": "fpq", "partition": "dev", "group_id": "g3", "question": "Is my deodorant causing cancer?", "correction": "c"},
    {"id": "nfp_1", "set": "nfp", "partition": "fit", "group_id": "g4", "question": "How do I manage nausea?"},
    {"id": "nfp_2", "set": "nfp", "partition": "test", "group_id": "g5", "question": "What does stage II mean?"},
]
E1 = [{"id": "e1_a", "question": "Does sugar feed my tumor?"},
      {"id": "e1_b", "question": "Since chemo always fails, what now?"},
      {"id": "e1_c", "question": "Is my deodorant causing cancer?"}]
TWINS = [
    {"id": "e1_a_true", "set": "tpair", "pair_id": "e1_a", "label_false_premise": 0, "question": "Does sugar not specifically feed my tumor?"},
    {"id": "e1_a_fpara", "pair_id": "e1_a", "label_false_premise": 1, "question": "Given that sugar feeds tumors, what should I eat?"},
    {"id": "e1_b_true", "set": "tpair", "pair_id": "e1_b", "label_false_premise": 0, "question": "Since chemo often works, what now?"},
    {"id": "e1_c_true", "set": "tpair", "pair_id": "e1_c", "label_false_premise": 0, "question": "Is my deodorant safe regarding cancer?"},
    {"id": "e1_zz_true", "set": "tpair", "pair_id": "e1_zz", "label_false_premise": 0, "question": "Orphan twin"},
]


def test_map_twin_rows_uses_text_to_reach_suite_ids_and_drops_orphans():
    twins, fparas = lt.map_twin_rows(TWINS, E1, SUITE)
    assert set(twins) == {"fpq_1", "fpq_2", "fpq_3"}
    assert twins["fpq_1"]["id"] == "e1_a_true" and twins["fpq_1"]["origin"] == "fpq_1"
    assert set(fparas) == {"fpq_1"}


def test_assemble_training_keeps_complete_pairs_only_and_filters_by_score():
    twins, fparas = lt.map_twin_rows(TWINS, E1, SUITE)
    fp_answers = {"fpq_1": "Actually sugar does not feed tumors...", "fpq_2": "Chemo does not always fail...", "fpq_3": "x"}
    fp_scores = {"fpq_1": 5, "fpq_2": 3, "fpq_3": 5}
    twin_answers = {"e1_a_true": "Sugar in moderation is fine...", "e1_b_true": "Chemo often works..."}
    rows, summary = lt.assemble_training(SUITE, twins, fparas, fp_answers, fp_scores, twin_answers)
    kinds = [(r["kind"], r["origin"]) for r in rows]
    assert kinds == [("fpq", "fpq_1"), ("fpara", "fpq_1"), ("twin", "fpq_1")]
    assert rows[1]["target"] == rows[0]["target"]  # false paraphrase shares the accepted correction
    assert summary["dropped"] == {"fp_answer_below_min_score": 1}  # fpq_2 scored 3; fpq_3 is not a training partition
    assert summary["kinds"] == {"fpq": 1, "fpara": 1, "twin": 1}
    assert summary["data_hash"] == digest(rows)
    # a twin without a Plain answer removes the whole origin, so pairs stay complete
    rows2, summary2 = lt.assemble_training(SUITE, twins, fparas, fp_answers, {"fpq_1": 5, "fpq_2": 5}, {"e1_b_true": "ok"})
    assert [r["origin"] for r in rows2] == ["fpq_2", "fpq_2"]
    assert summary2["dropped"] == {"twin_without_plain_answer": 1}


def test_assemble_training_none_negatives_is_fpq_only_control():
    twins, fparas = lt.map_twin_rows(TWINS, E1, SUITE)
    rows, summary = lt.assemble_training(SUITE, twins, fparas, {"fpq_1": "a", "fpq_2": "b"}, {"fpq_1": 5, "fpq_2": 5}, {},
                                         negatives="none", fpara_positives=False)
    assert [r["kind"] for r in rows] == ["fpq", "fpq"]
    assert summary["negatives"] == "none"
    with pytest.raises(ValueError):
        lt.assemble_training(SUITE, twins, fparas, {}, {}, {}, negatives="both")


def test_evaluation_rows_hold_out_partitions_and_add_twins_as_nfp_kind_twin():
    twins, _ = lt.map_twin_rows(TWINS, E1, SUITE)
    rows = lt.evaluation_rows(SUITE, twins, ("fit",))
    ids = [r["id"] for r in rows]
    assert ids == ["fpq_3", "nfp_2", "e1_c_true"]
    twin = rows[-1]
    assert twin["set"] == "nfp" and twin["kind"] == "twin" and twin["partition"] == "dev" and twin["origin"] == "fpq_3"
    with pytest.raises(ValueError):
        lt.split_origins(SUITE, ("fit", "dev", "test"))


class FakeTok:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 50 + 1 for c in text]}


def test_encode_example_masks_prompt_and_appends_eos_and_pads():
    tok = FakeTok()

    def render(t, q):
        return "<u>" + q + "</u>"
    ids, labels, cut = lt.encode_example(tok, render, "abc", "de", 100)
    n_prompt = len("<u>abc</u>")
    assert len(ids) == n_prompt + 3 and ids[-1] == 99 and not cut
    assert labels[:n_prompt] == [-100] * n_prompt and labels[n_prompt:] == ids[n_prompt:]
    ids2, labels2, cut2 = lt.encode_example(tok, render, "abc", "de", n_prompt + 1)
    assert cut2 and len(ids2) == n_prompt + 1
    with pytest.raises(ValueError):
        lt.encode_example(tok, render, "abc", "de", n_prompt)
    batch = lt.pad_batch([(ids, labels), (ids2, labels2)], 0)
    assert len(batch["input_ids"][1]) == len(ids) and batch["attention_mask"][1][-1] == 0 and batch["labels"][1][-1] == -100
    assert lt.batches(5, 2, 17, 0) != lt.batches(5, 2, 17, 1) and sorted(sum(lt.batches(5, 2, 17, 0), [])) == list(range(5))


def test_s5_table_reports_per_kind_and_rescue_harm():
    rows = [{"id": "f1", "set": "fpq", "partition": "dev"}, {"id": "f2", "set": "fpq", "partition": "dev"},
            {"id": "n1", "set": "nfp", "partition": "test"}, {"id": "t1", "set": "nfp", "kind": "twin", "partition": "dev"}]
    scores = {"plain": {"f1": 2, "f2": 5, "n1": 5, "t1": 5},
              "lora_x": {"f1": 5, "f2": 5, "n1": 3, "t1": None}}
    lines, table = lt.s5_table(rows, scores)
    fpq = table["lora_x|fpq|all"]
    assert fpq["s5"] == 2 and fpq["rescue"] == 1 and fpq["harm"] == 0
    assert table["lora_x|nfp|all"]["harm"] == 1 and table["lora_x|twin|all"]["valid"] == 0
    assert any(line.startswith("| lora_x | twin | dev |") for line in lines)


def test_judge_scores_reads_ledger_mapping(tmp_path):
    from src import well_eval as we
    plan = {"version": we.VERSION, "mapping": {"fp_unconditional": {"fpq_1": "job1", "fpq_2": "job2"}},
            "jobs": {"job1": {}, "job2": {}}, "model": "claude-sonnet-5"}
    (tmp_path / "judge_plan.json").write_text(json.dumps({"plan_hash": digest(plan), "plan": plan}))
    h = digest(plan)
    events = [{"id": "job1", "plan_hash": h, "status": "started"},
              {"id": "job1", "plan_hash": h, "status": "finished", "raw": "Rating: 5", "model": "claude-sonnet-5", "valid": True, "score": 5}]
    (tmp_path / "attempts.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    scores = lt.judge_scores(tmp_path, "fp_unconditional")
    assert scores == {"fpq_1": 5, "fpq_2": None}
    with pytest.raises(ValueError):
        lt.judge_scores(tmp_path, "plain")


def test_cli_stages_exist():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cli", Path(__file__).resolve().parents[1] / "scripts" / "lora_twin.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    for name in ("stage_data", "stage_train", "stage_generate", "stage_compare"):
        assert callable(getattr(cli, name))
