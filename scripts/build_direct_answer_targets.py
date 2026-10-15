"""Package directly authored Cancer-Myth fit answers; no generation or training.

Uses the existing grouped split. References and labels are authoring metadata,
never appended to the model's question. Rebuild with python3 from the repo root.
"""
from collections import Counter
import csv
import json
from pathlib import Path

from premise_sft import BUNDLE, ROOT, audit_splits, cancer_rows, select_train, sha, target_text

DEST = BUNDLE / "answer_targets_20260930"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def keyed(rows):
    result = {}
    for row in rows:
        assert row["id"] not in result, f"Duplicate: {row['id']}"
        result[row["id"]] = row
    return result


def main():
    canonical = cancer_rows()
    split_counts = audit_splits(canonical)
    fit = [r for r in canonical if r["partition"] == "fit"]
    preview_path = BUNDLE / "direct_answer_drafts_20260930.json"
    preview = keyed(json.loads(preview_path.read_text())["records"])
    with (DEST / "authored.tsv").open(newline="") as f:
        authored = keyed(list(csv.DictReader(f, delimiter="\t")))
    references = keyed([json.loads(s) for s in (DEST / "annotation_context.jsonl").read_text().splitlines()])
    expected = {r["id"] for r in fit}
    assert not (authored.keys() & preview.keys()), "Authorship sources overlap"
    assert authored.keys() | preview.keys() == expected, "Coverage differs from canonical fit"
    assert references.keys() == expected, "Reference coverage differs from fit"
    output, review, chats = [], [], []
    for original in fit:
        rid = original["id"]
        if rid in authored:
            source = authored[rid]
            assert None not in source, f"Malformed TSV row: {rid}"
            answer = source["answer"]
            flags = [v for v in source["review_flags"].split(";") if v]
            note = None
            origin = "authored.tsv"
        else:
            source = preview[rid]
            answer = source["correction_paragraph" if original["label"] else "ordinary_answer"]
            flags = [] if source["annotation_scope"] == "clear" else [source["annotation_scope"]]
            note = source["review_note_ko"]
            origin = "../direct_answer_drafts_20260930.json"
        assert answer == answer.strip() and len(answer.split()) >= 25, rid
        assert not answer.startswith(("{", "[", "```")), rid
        assert "has_false_premise" not in answer, rid
        provenance = {
            "author": "Codex assistant, directly authored in conversation",
            "source": origin,
            "labels_and_references_visible_to_author": True,
            "well_score": None,
            "expert_validated": False,
            "review_flags": flags,
        }
        row = dict(original, answer=answer, answer_provenance=provenance)
        assert row["target_usable"], rid
        output.append(row)
        review.append({
            "id": rid, "partition": "fit", "label": original["label"],
            "question": original["question"],
            "false_premises": original["false_premises"],
            "reference_correction": original["reference_correction"],
            "annotation_context": references[rid]["reference"],
            "answer": answer, "review_flags": flags, "preview_note_ko": note,
        })
        chats.append({"id": rid, "messages": [
            {"role": "user", "content": original["question"]},
            {"role": "assistant", "content": answer},
        ]})
    selected = select_train(output, "answer", "answer_matched")
    assert len(selected) == 438 and Counter(r["label"] for r in selected) == {True: 349, False: 89}
    assert all(target_text(r, "answer") == r["answer"] for r in selected)
    assert len({r["answer"] for r in output}) == 438, "Exact duplicate answers"
    write_rows(DEST / "fit.jsonl", output)
    write_rows(DEST / "chat_training.jsonl", chats)
    write_rows(DEST / "review_records.jsonl", review)
    flagged = [r for r in review if r["review_flags"]]
    write_rows(DEST / "flagged_records.jsonl", flagged)
    word_counts = {}
    for label, name in [(True, "fpq"), (False, "nfp")]:
        lengths = [len(r["answer"].split()) for r in output if r["label"] == label]
        word_counts[name] = {"min": min(lengths), "max": max(lengths), "mean": round(sum(lengths)/len(lengths), 2)}
    sources = [DEST / "authored.tsv", DEST / "annotation_context.jsonl", preview_path,
               BUNDLE / "bundle.json", BUNDLE / "cancer_questions.jsonl", Path(__file__).resolve()]
    products = [DEST / n for n in ["fit.jsonl", "chat_training.jsonl", "review_records.jsonl", "flagged_records.jsonl"]]
    manifest = {
        "status": "complete_authored_training_draft_unjudged",
        "scope": "Cancer-Myth fit only; CREPE deferred; no training or evaluation launched",
        "counts": {"total": 438, "fpq": 349, "nfp": 89},
        "original_split_counts": split_counts,
        "input": "Original question only; no label, annotation, or diagnostic instruction added",
        "output": "FPQ correction-focused English prose; NFP request-focused English prose",
        "loss": "Existing premise_sft.py answer objective masks user tokens; train assistant answer only",
        "objective": "answer", "cohort": "answer_matched",
        "training_invocation_example_not_executed": [
            "source scripts/premise_sft_env.sh",
            'CUDA_VISIBLE_DEVICES=0 "${PREMISE_ENV}/bin/python" -u scripts/premise_sft.py train --data configs/premise_sft/answer_targets_20260930 --model-lock "${PREMISE_ART}/downloads.json" --objective answer --cohort answer_matched --out "${PREMISE_ART}/runs/cancer_direct_answer_v1" --batch-size 2 --grad-accum 8 --max-length 2048',
        ],
        "initialization": "Fresh pinned Qwen2.5-7B-Instruct base, not the earlier JSON-output adapter",
        "cohort_note": "The reader's answer_matched name selects all 438 here, NOT the old 272-answer cohort.",
        "authoring": "426 directly authored TSV answers plus 12 earlier directly authored previews; no Sonnet generation",
        "reference_origin": "Fit-only snapshot of original annotation context from results/audits/answer_behavior_full_20260929_v2 inputs/manifest; no judge outputs used",
        "review": {
            "flagged_count": len(flagged),
            "flags_are": "Assistant notes about scope, source conflicts, or careful wording; not independent validation or automatic exclusion",
            "flag_counts": dict(sorted(Counter(f for r in flagged for f in r["review_flags"]).items())),
            "missing_nfp_premise_annotations": [r["id"] for r in review if not r["label"] and not r["annotation_context"].get("premises")],
        },
        "answer_word_counts": word_counts,
        "limitations": [
            "AI-authored targets, not physician-written or independently medically validated gold answers; no Well scores assigned.",
            "Labels are retained, but a source annotation may infer a general claim absent from the question. Flagged draft rows remain included for transparent review.",
            "NFP answers may qualify medically incorrect background assertions or unsafe plans while answering the request; NFP is not permission to repeat misinformation.",
            "Natural-language answer SFT changes format, content, author, and sometimes cohort relative to prior JSON/teacher-answer runs; not an isolated causal test of premise localization.",
            "FPQ correction-focused answers need not fully answer the practical request; high Well correction scores would not prove full response helpfulness.",
            "Existing dev/test split was previously inspected; this is not a newly unseen evaluation set. No dev/test targets generated here.",
            "Primary-source checks were selective, not a full fact-check or item-by-item external validation.",
        ],
        "selective_reference_urls": [
            "https://www.cancer.gov/about-cancer/causes-prevention/risk/myths",
            "https://www.cancer.gov/types/lymphoma/patient/primary-cns-lymphoma-treatment-pdq",
            "https://www.cancer.gov/types/childhood-cancers/late-effects-pdq",
            "https://www.cancer.gov/types/extracranial-germ-cell/patient/germ-cell-treatment-pdq",
            "https://www.cancer.gov/about-cancer/causes-prevention/risk/hormones/reproductive-history-fact-sheet",
            "https://www.cancer.gov/types/breast/hp/pregnancy-breast-treatment-pdq",
        ],
        "rebuild_command": "python3 scripts/build_direct_answer_targets.py",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "output_sha256": {p.name: sha(p) for p in products},
    }
    write_json(DEST / "manifest.json", manifest)
    print(json.dumps({"path": str(DEST), "counts": manifest["counts"], "flagged": len(flagged),
                      "answer_words": word_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
