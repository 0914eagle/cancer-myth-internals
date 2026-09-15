# Well-Actually answer evaluation artifacts

Source: [ShenranTomWang/Well](https://github.com/ShenranTomWang/Well/tree/a7ee871eadde1104f7560a2874a03cbf5221dbaa), commit `a7ee871eadde1104f7560a2874a03cbf5221dbaa`.
Exact URLs and SHA-256 hashes are in `sources.json`. The pinned Git tree had no
LICENSE file; we do not infer a license grant or relicense upstream content.

`fpq_system.txt`, `fpq_user.txt`, `nfp_system.txt`, and `nfp_user.txt` preserve the
role/content strings in the first `*ResponseLevelScoreTemplate.generate()` of
the vendored modules. Whitespace, wording, score examples, and reminders are
unchanged. Only `{self.question}`-style expressions were converted to named
format slots. The offline test executes only each reviewed `generate` method
and verifies exact equality to the rendered template, including whitespace.
No upstream module is imported during ordinary runs.

The active 0–5 examples are **embedded in the template code**. The two vendored
`eval_few_shot.json` files instead contain older −1/0/+1 examples and are kept
only as source evidence; this evaluator does not insert them. `example_questions()`
returns the two actual embedded demonstration questions. Exclude these from
evaluation. `evaluate_well.py prepare` refuses overlaps.

The TPQ template still includes the upstream reminder referring to a false claim
and its explanation even though no such fields are supplied for TPQ. This is
preserved, not silently fixed. These prompts define a different metric from
Cancer-Myth PCR/PCS and the earlier custom NFP v2 judge.

FPQ requires the original **source myth** (`premise_text` or `source_myth`) and
**presupposition correction** (`correction`) separately. This follows the pinned
`data_gen/CancerMyth/prepare_dataset.py`: `presuppositions=[source_myth]` and
`answer=presupposition_correction`. Existing pilot manifests containing only a
combined `example_assumption` need a question-text join to those source fields;
the evaluator will not guess or duplicate that field into both slots. Neither
field is supplied to the answer-generating backbone. TPQ answer evaluation uses
the question and answer only, exactly as the upstream class does; true-premise
annotations are not an extra judge input.

The repository's `make_caller` accepts one text string, so saved system/user
content is serialized with `[SYSTEM]`/`[USER]` delimiters. That transport adapter,
the default Codex Terra judge, and our own question split differ from the paper;
do not label these runs an exact reproduction. The complete message objects,
serialized prompt, backend/model, source hashes and mapping are frozen in the
judge plan. No model calls occur during prepare or report.

```bash
python scripts/evaluate_well.py prepare \
  --questions "$SUITE_DIR/questions.jsonl" \
  --answers "$SUITE_DIR/answers/plain.jsonl" "$SUITE_DIR/answers/zero_shot_cot.jsonl" \
  --out-dir "$SUITE_DIR/well_all"
python scripts/evaluate_well.py score --out-dir "$SUITE_DIR/well_all" --max-calls 20
python scripts/evaluate_well.py report --out-dir "$SUITE_DIR/well_all"
```

Each answer JSONL row needs `id`, `method`, and `response`; optional question/set/
partition fields must match the canonical inventory. All remaining row metadata
is retained through a row hash and the original source-file hash. Empty strings
are judgeable (the original rubric may score them zero); unfinished records and
duplicates are rejected. Missing answers remain missing in the expected count.

Scoring appends/fsyncs a started event before each request. Started requests,
including interrupted/invalid ones, are never retried automatically. Each score
invocation requires a positive `--max-calls`. Model/backend errors or three
consecutive malformed replies stop the invocation. Identical complete inputs
across methods share one judgment; differing answers never share just because
they concern the same question. Score 0 is valid; missing/invalid is not 0 or 5.

Reports include counts of all six scores, valid/expected and missing counts,
mean over scores 1–5 with its explicit denominator, S5/all expected, and
gibberish/all expected. In incomplete reports S5/all is observed coverage rather
than a completed performance estimate. Paired S5 rescue/harm uses only jointly
valid question pairs. TPQ scores are not overall medical accuracy, and S5 is
not the original Cancer-Myth PCR.
