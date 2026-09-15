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
