# Experiments — server setup and run commands

> **Current first experiment:** follow [the single-Gemma pilot](docs/23_gemma_steering_pilot.md)
> and `scripts/run_gemma_pilot.sh`. It provides fit-only C/scale, grouped
> fit/dev/test, prompting baselines, dev selection and a locked test setting.
> Pilot judge: `JUDGE_BACKEND=codex JUDGE_MODEL=gpt-5.6-sol` (GPT-5 access failure fix, 2026-09-10).
> Run `bash scripts/run_gemma_pilot.sh check-judge` to verify server access before generation.
> Uses the server's Codex CLI login; no OpenAI API key is required. The historical
> GPT-4o calibration/protocol discussion below does not override this pilot setting.
> The E1/E2 commands below describe historical diagnostics; the legacy E2
> direction is trained on all questions and is not a held-out result.
> Judge v2 refuses unsigned legacy caches and no longer scores TPQ with NFP's rubric.

Layout follows `medical_nla` exactly, with the `CANCER_MYTH_` prefix:

| | server 125 | server 62 |
|---|---|---|
| code | `/home/eagle0914/cancer-myth-internals` | same |
| data root | `/data1/heejae` | `/data/heejae` |
| artifacts | `${DATA_ROOT}/cancer_myth_internals/{data,external,activations,results,reports,logs}` | same |
| venv | `${DATA_ROOT}/uv/cancer_myth_internals` | same |
| HF cache | `${DATA_ROOT}/hf_cache` (HF_HOME only, never TRANSFORMERS_CACHE) | same |
| GPUs | four RTX 4090, physical 0-3 | physical 2,3 |

Paths are not written into configs. One variable per machine,
`CANCER_MYTH_DATA_ROOT`, is substituted into every `configs/*.yaml`
(`src/config.py`). `scripts/env.sh` finds it from the venv location.

## First time on a machine

```bash
mkdir -p /home/eagle0914 && cd /home/eagle0914
git clone https://github.com/0914eagle/cancer-myth-internals.git
cd cancer-myth-internals
DATA_ROOT=/data1/heejae bash scripts/bootstrap_server.sh
```

The bootstrap creates the uv venv (python 3.11), installs torch 2.5.1 from the
cu121 index first (the same pin as medical_nla's bootstrap), then
`uv pip install -e ".[dev]"`, clones `bill1235813/cancer-myth` and
`ShenranTomWang/Well` under `${DATA_ROOT}/cancer_myth_internals/external`, and
runs the GPU check.

Server 125's driver is CUDA 12.2 (`nvidia-smi` shows 535.x). A torch wheel
built against a newer CUDA loads but reports "The NVIDIA driver on your system
is too old" and `torch.cuda.is_available()` is False, so every worker stops at
the GPU check. If a venv ended up with such a wheel, replace it in place:

```bash
source /data1/heejae/uv/cancer_myth_internals/bin/activate
uv pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# expect: 2.5.1+cu121 True 4
```

Gated checkpoints (Llama-3.1, Gemma-2, Gemma-3) need the account that accepted
the licences:

```bash
export HF_HOME=/data1/heejae/hf_cache
huggingface-cli login      # newer clients: hf auth login
huggingface-cli whoami
```

The judge and the premise-span alignment need an LLM. Default transport is
`codex exec` (medical_nla's `run_judge.py` pattern; uses the codex login, no
API key). The OpenAI API is the alternative:

```bash
codex --version                       # JUDGE_BACKEND=codex (default)
export OPENAI_API_KEY=...             # JUDGE_BACKEND=openai; ~/.bashrc, never the repo
```

**The judge is a measurement instrument and must be calibrated once.** The
paper's judge was GPT-4o; `all_data.json` ships GPT-4o's scores for eight
models' answers on all 585 questions, so any judge can be checked against it
without generating anything:

```bash
bash scripts/run_calibrate_judge.sh                          # codex default model, detaches
BACKEND=openai bash scripts/run_calibrate_judge.sh           # gpt-4o via API (should agree ~100% with itself)
cat $ART/reports/judge_calibration/codex_default/calibration.md
```

codex with a ChatGPT login cannot serve gpt-4o ("not supported when using
Codex with a ChatGPT account", checked 2026-09-07), so the paper's judge is
reachable only through the OpenAI API.

Result on 2026-09-07 (codex served gpt-5.6-sol): 3-way agreement 58%,
kappa 0.28, and only 58% of GPT-4o's +1 labels kept as +1 -- a consistently
*stricter* judge (PCR 22-30% lower, PCS 0.25 lower), not noise. Rankings are
preserved. Decision (docs/experiments/01): iterate with codex (relative
comparisons only), score every number that enters a table with the OpenAI
API and gpt-4o (`JUDGE_BACKEND=openai`), so it matches the paper's judge.

## Long jobs detach themselves

Every wrapper that can take more than half an hour (`run_e0_rows.sh`,
`run_e1_model.sh`, `run_e1_4gpu_125.sh`, `run_e2_steer_125.sh`,
`run_calibrate_judge.sh`) re-launches itself under `nohup` and returns at
once, printing the log path (`scripts/lib/detach.sh`). A dropped SSH session
never kills a run.

```bash
bash scripts/jobs.sh          # every launch: running/done, last log line
bash scripts/jobs.sh gpu      # plus nvidia-smi
tail -f <log path printed at launch>
FOREGROUND=1 bash scripts/...   # run inline instead (debugging)
```

Logs live under `$ART/logs/<tag>_<timestamp>.log`; `jobs.tsv` there is the index.

## Every session

```bash
cd /home/eagle0914/cancer-myth-internals
source /data1/heejae/uv/cancer_myth_internals/bin/activate
source scripts/env.sh /data1/heejae     # prints host, roots, GPUs, whether the key is set
pytest -q                                # 24 tests, no GPU
```

## E0 — rows (once, CPU + API, 약 10 min)

```bash
DATA_ROOT=/data1/heejae bash scripts/run_e0_rows.sh
```

Writes `$DATA/e1_rows_v1/{questions,activation_rows}.jsonl`, an alignment
audit and 40 sample alignments. **Read `sample_alignments.md` before E1**:
the premise span is the position-A readout, and a bad alignment is a bad
readout. The TPQ loader prints the HF columns it found; if
`shenranw/CancerMyth-TPQ` names its presupposition field differently from
`presuppositions`, fix `src/rows.py::_first_presupposition` and rerun.

Re-alignment (after changing the prompt, or to retry rows that got no span):
`run_e0_rows.sh` reuses every verified LLM span and asks the LLM again only
for the rest; A rows carry the span in their id, so `STAGES="2"` then
extracts only the new A rows and `prune_manifests.py` drops the old ones from
the manifests. Rerun stages 4-8 afterwards (CPU except 5 and 8, which find
nothing new to extract).

## E1 — pre-diagnostic on four cards (약 2 days)

```bash
bash scripts/run_e1_4gpu_125.sh      # detaches itself; prints the log path
bash scripts/jobs.sh gpu
```

Phase 1 runs Llama-3.1-8B (GPU 0), Qwen2.5-7B (GPU 1), Gemma-2-9B (GPU 2)
in parallel: Plain responses, then A/B/D activations at every hidden-state
index. Phase 2 runs Gemma-2-27B in bf16 on GPUs 1,2,3 (no quantization: it
changes activation values). Phase 3, per model: judge (`JUDGE_BACKEND`, default codex), position-E
rows, E activations, the A readout sweep, the C direction.

While the 27B is still in phase 2, GPU 0 is idle and the judge needs no
GPU, so the finished small models can start their stages 3-7 early:

```bash
bash scripts/run_e1_stages_125.sh    # llama, qwen, gemma9b in sequence on GPU 0; stages 3-7
```

Every stage resumes, so when the driver later reaches phase 3 for the same
model its judge finds nothing left and the rest recomputes from disk. The
one thing to avoid is two judges on one output file at the same time:
`run_judge.py` holds a lock per file and the second writer exits, which
fails that driver worker. If that happens, rerun `run_e1_stages_125.sh`
with `MODELS=` set to whatever is still missing.

One model, one stage at a time (each stage resumes):

```bash
CONFIG=configs/llama31_8b.yaml GPUS=0 STAGES="1 2" bash scripts/run_e1_model.sh
CONFIG=configs/llama31_8b.yaml GPUS=0 STAGES="3 4 5 6 7" bash scripts/run_e1_model.sh
LIMIT=20 CONFIG=configs/llama31_8b.yaml GPUS=0 bash scripts/run_e1_model.sh   # smoke
```

Outputs per model under `$ART/results/e1/<model>/`:

| file | what |
|---|---|
| `plain_responses.jsonl`, `plain_judge.jsonl`, `plain_summary.json` | Plain PCR / PCS / NFP / TPQ (Table 3 reproduction) |
| `probe_sweep/heatmap.png`, `summary.md` | A readout AUROC, layer x position, logistic and diff-of-means |
| `direction_c/table.md`, `directions.npz` | PCR predictability at D, cos(A, C) per layer, gate probe |
| `direction_c/table_pair.md` | stage 8: paired C (reference +1 minus -1 answers behind the same prompt), its transfer to the model's own PCR, cos with A and c_e |

Activations: `$ART/activations/e1_<model>_ad/layerNN/{last_token,last_subtoken,span_mean}/manifest.jsonl`
and `e1_<model>_e/…` — the medical_nla manifest layout, readable by its
`src.run_nla` for the NLA test on Gemma-3-12B (`configs/gemma3_12b.yaml`, GPUs 2,3).

### Stage 8: the paired C direction

With PCR at 3-6 % under the codex judge, an 8B backbone corrects 20-35 of
585 questions, too few for a diff-of-means C from its own responses. Stage 8
builds C from Cancer-Myth's all_data.json instead: for 232 questions there is
a reference answer GPT-4o scored +1 and one scored -1; both are teacher-forced
behind the same Plain prompt and the response opening (first 5 and first 32
tokens, span mean) is read. corr minus follow per question cancels the
question, and the mean over questions is `c_pair`. Reference authors are
balanced across the two sides so the difference is not "Gemini style minus
MDAgents style". `table_pair.md` reports whether `c_pair` separates the
model's own +1 from its own -1 (leave-question-out), whether it reads the
model's own PCR before generation at D, and its cosine with A and with the
natural `c_e`. `directions.npz` gains `L{layer}_c_pair5` / `_c_pair32`, which
`run_steer.py --direction-key c_pair32` consumes.

```bash
MODELS="llama31_8b qwen25_7b gemma2_9b" STAGES="8" bash scripts/run_e1_stages_125.sh
MODELS="gemma2_27b" GPUS=1,2,3 STAGES="8" bash scripts/run_e1_stages_125.sh
```

### Stage 9: minimal pairs for the A readout

Text alone (TF-IDF on the question) separates fpq from NFP at 0.77, above
the B/D probes and close to the A probe, so the fpq-vs-NFP contrast carries
benchmark style. Stage 9 reads A on minimal pairs instead: every fpq with a
verified span gets a twin in which only that span is replaced by the correct
belief (spliced, so the rest is byte-identical; a second LLM call confirms the
false belief is gone). Build the twins once (codex, CPU), then per model:

```bash
DATA_ROOT=/data1/heejae bash scripts/run_e0_twins.sh                  # after run_e0_rows.sh
MODELS="gemma2_27b" GPUS=1,2,3 STAGES="9" bash scripts/run_e1_stages_125.sh
```

`probe_sweep_twins/summary.md` and `text_baseline.md` there are the numbers to
compare: the probe must clearly exceed the text row on the pairs.

## Reading E1 (the fork)

| step | number | read |
|---|---|---|
| 1 | best A AUROC at `A_premise` or `B_question_end` | ≥ 0.80 strong; ≈ 0.70 = Two Axes' CREPE level; < 0.65 the representation is weak |
| 2 | `pcr_auroc_D_diffmeans` and `cos_A_C_at_D` | PCR predictable (≥ 0.75) and cos < 0.5: the A × C decomposition holds. cos > 0.8: A and C are one direction |
| 3 | E2 below | gated PCR up with NFP ≈ Plain: framework paper |

## E2 — steering (one model, one card, ~half a day)

```bash
DATA_ROOT=/data1/heejae CONFIG=configs/llama31_8b.yaml GPUS=0 \
  LAYER=16 ALPHAS="2 4 8" bash scripts/run_e2_steer_125.sh
```

Pick `LAYER` from `direction_c/table.md`. Runs plain / unconditional /
A-gated at each alpha on fpq 100 + nfp 100 + tpq, judges, and prints one
table: PCR, PCS, NFP, TPQ per condition. The row that matters is
`gated_*` against `uncond_*` at the same alpha.

## What each script does

| script | stage |
|---|---|
| `scripts/make_rows.py` | E0 rows + premise alignment (LLM verbatim substring, heuristic fallback) |
| `scripts/run_generate.py` | Plain responses (HF generate, greedy; `--paper-protocol` for T=0.7) |
| `src/extract_activations.py` | hidden states at A/B/D/E, every layer, medical_nla layout |
| `scripts/run_judge.py` | validate.py / validate_nfp.py prompts through codex exec or OpenAI; resumable; lock; `--dry-run` |
| `scripts/make_paired_rows.py`, `scripts/run_direction_pair.py` | stage 8: paired C rows from all_data.json reference answers; the direction, its checks, npz update |
| `scripts/run_text_baseline.py` | stage 6/9: TF-IDF text-only AUROC, the surface-form ceiling for the A readout |
| `scripts/make_true_twins.py`, `scripts/prune_manifests.py` | stage 9: true-premise twins (minimal pairs); manifest cleanup after re-alignment |
| `scripts/calibrate_judge.py` | agreement of the chosen judge with GPT-4o on the answers shipped in all_data.json |
| `scripts/summarize_judge.py` | PCR / PCS / NFP / TPQ, by category |
| `scripts/make_response_rows.py`, `merge_labels_into_manifests.py` | E rows and labels |
| `scripts/run_probe_sweep.py` | A readout, grouped 5-fold, heatmap |
| `scripts/run_direction_c.py` | C direction, cos(A,C), PCR predictability, gate probe |
| `scripts/run_steer.py` | unconditional / gated steering with the C direction |

## Rules carried over from medical_nla

- No vLLM/SGLang in the activation path; generation and extraction both go
  through `transformers` so the same tokenization serves both.
- No quantization unless approved: it changes activation values.
- Never `rm` an activation run to "start clean"; extraction resumes by id.
- Restricted or personal data does not enter the repo; Cancer-Myth and NFP
  are public and live under `external/`.
