# Qwen final-answer budget check — planned, not an experimental result

The user-reported baseline status has 362/732 Plain and 40/58 zero-shot CoT
final answers reaching the 512-token cap without EOS. Four inspected dev tails
end in mid-sentence/list. They do not establish repetition, correction failure,
or that a longer budget improves medical accuracy.

Before large judge runs, `scripts/check_final_budget.py` compares final caps
512 and 1024. This changes the **final answer** budget, not the reasoning cap.

- One fixed panel: 8 FPQ + 4 NFP, fit only, one item per source group; hashed ID
  order with seed 17. No selection on scores, cap hits, or observed answers.
- Plain and zero_shot_cot, using current suite prompts unchanged. The latter is
  the existing two-stage step-by-step reasoning then final-answer implementation.
- Each CoT pair shares the exact cached reasoning trace (1024-token cap).
- Generate both final caps anew in a separate directory. Compare the 512 Plain
  control against its original saved answer. Runtime identity must match.
- At most 48 final generations + 12 reasoning generations. Local GPU 0 **or** 1
  only. No judge/backend calls, residual intervention, or gate tuning.
- Save intermediate cached results atomically and resume completed requests.
  Errors stop execution; no retry loop. Normal interruption can be resumed by
  the same command. A hard kill can leave a lock: inspect its PID before removal.
- Source suite identity/specs and selected Plain files are hashed; the source
  generation code must match. Ongoing baseline generation writes elsewhere.
- Report final/review cap hits, EOS, paired answer changes, decoded-text prefix
  divergence, control replay discrepancies, and complete paired answers.
  Prefix checks are **not token-ID equivalence**, nor correctness metrics.
- `report` is offline and works on incomplete runs. There is no automatic judge
  score or medical verdict. Inspect added text for corrections, extra errors,
  unsupported rebuttals, and repetition. A 12-item panel cannot establish final
  performance or full-suite cap rates. Retain original metrics separately.

## Server commands

Run in another tmux window after baseline extraction has released GPU 1:

```bash
cd /home/eagle0914/cancer-myth-internals
git pull --ff-only origin main
source /data1/heejae/uv/cancer_myth_internals/bin/activate
source scripts/env.sh /data1/heejae
export SUITE_DIR=/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_v1
export BUDGET_DIR="$SUITE_DIR/final_budget_fit_v1"
python -m pytest -q tests/test_final_budget_check.py
CUDA_VISIBLE_DEVICES=1 python -u scripts/check_final_budget.py all \
  --suite-dir "$SUITE_DIR" --out-dir "$BUDGET_DIR"
```

`all` = prepare, generate, report; it never launches the original full suite.
Check GPU 1 is still available before generation. Do not concurrently launch two
copies into this diagnostic directory. Resume via the same command.

```bash
python scripts/check_final_budget.py report \
  --suite-dir "$SUITE_DIR" --out-dir "$BUDGET_DIR"
```

Review files: `$BUDGET_DIR/report.md`, `$BUDGET_DIR/examples.md`.
Tests are provided for the server; no local pytest, torch, or GPU checks were run.

## User-reported outcome and full-suite transition

Source: user-pasted server report in this discussion; original ledger/examples
not yet copied into the repository. All 12 pairs completed for each method:

| Method | 512 length stops | 1024 length stops | Changed answers | Prefix divergence |
|---|---:|---:|---:|---:|
| Plain | 7/12 | 0/12 | 7/12 | 0/12 |
| Zero-shot CoT | 8/12 | 0/12 | 8/12 | 0/12 |

All 15 length-stopped answers ended with EOS at 1024; reviews did not change.
Source Plain replay matched 12/12. This supports a 1024 final budget, not improved
correction accuracy; other methods' 1024 cap rates still need monitoring.

`scripts/prepare_final1024.py` snapshots the source suite into a new, non-nested
directory. It copies immutable shared generation/binary caches, features,
available detection records and completed question-level gate results. It does
not copy final-answer ledgers, generation method specs, judge scores, process
locks or logs. Cache keys include exact prompt, budget and runtime identity.
Hence saved reasoning/extractions with the same budget can be reused; old
512-token finals are not labelled as 1024 outputs, even if they ended at EOS.
Final answers are regenerated with 1024, not continued by appending saved text.

The new `generation_defaults.json` freezes final cap=1024 for the existing CLI
and shell wrapper. Old suites without that file continue to default to 512.
Generation code/prompts and cache signatures are unchanged. Prepared snapshots
are idempotent; rerunning preparation does not import later source work into an
already active destination. Never run preparation/generation concurrently on
the destination. No judge is invoked by migration, generate, detect, gate, or
judge-plan. A separate explicitly capped `judge` command is still required.

Stop the old **512 generation** job in its original tmux window with Ctrl-C
before starting the replacement (do not use a broad process kill). This avoids
paying GPU time for both full suites. Keep the original folder.

```bash
cd /home/eagle0914/cancer-myth-internals
git pull --ff-only origin main
source /data1/heejae/uv/cancer_myth_internals/bin/activate
source scripts/env.sh /data1/heejae
python -m pytest -q tests/test_final1024_migration.py tests/test_baseline_cli.py
export OLD_SUITE_DIR=/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_v1
export SUITE_DIR=/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1
export CUDA_VISIBLE_DEVICES=0
python scripts/prepare_final1024.py --source-dir "$OLD_SUITE_DIR" --out-dir "$SUITE_DIR" &&
bash scripts/run_baselines.sh generate &&
bash scripts/run_baselines.sh detect &&
bash scripts/run_baselines.sh gate &&
bash scripts/run_baselines.sh judge-plan
```

Run inside tmux. `generate` does not repeat feature extraction; `detect` reuses
available premise-review and binary caches. The chain does **not** score answers.
Resume via the same commands. Afterward inspect `status`, cap rates, and judge
plan before approving a bounded scoring batch. No local pytest/model run was
performed for this transition code; only static syntax/diff checks.
