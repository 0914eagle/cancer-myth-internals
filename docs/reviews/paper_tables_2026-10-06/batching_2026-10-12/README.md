# Shared Luna extraction and GPU batching (2026-10-12)

## Experimental condition

At the user's request, both PreWoMe and Extract+Verify use one fixed GPT-6 Luna extractor. Its raw output and parsed premises are shared by Qwen2.5-7B, Gemma3-12B and Luna answer pipelines. Review/fact checking and final answers remain the responsibility of the evaluated model. Report this as **shared Luna extraction**, not an unmodified all-stages-own-model reproduction.

The extractor sees the fixed extraction prompt, fixed dataset examples and question only. It does not receive the question's FPQ/NFP label, correction or judge feedback. Fixed extraction is reusable across folds because no parameters or prompts are selected using held-out answers. The cache is keyed by actual messages, extractor configuration and source hashes, rather than by answer-model identity. Different questions or dataset few-shot messages do not share a cache key.

A per-key lock prevents duplicate extraction; four cross-process CLI slots cap concurrency. This is concurrent independent CLI requests, not GPU inference batch size. `prepare_shared_luna_extractions.py` can populate the FPQA cache independently of GPU generation. The pipeline also uses the same cache on demand. Authentication/provider failures are surfaced; the extractor never silently falls back to the local model.

## Batch transport

- Qwen applied cap: 8, padded input plus maximum output token budget 73,728.
- Gemma applied cap: 4, corresponding token budget 49,152. Validated on its two assigned GPUs.
- Batch requests across independent question pipelines. Within each question, extraction, review/checking and answer dependencies remain sequential.
- Left padding with an attention mask; greedy decoding, checkpoint, prompts and 8,192-token input/output caps remain unchanged.
- Per-row EOS parsing discards padding after that row terminates. Output-limit failures are kept as failures, not silently shortened answers.
- OOM halves the batch cap and retries the same inputs. Single-row OOM remains an infrastructure failure.
- Actual batch size, padded length and timing are recorded in each new call. Existing answers remain immutable; old calls without batch metadata were singleton calls. This is an execution-setting change and must be disclosed, not claimed to be identical generation.
- Single GPU dispatcher per model; parallel question workers do not call `generate` concurrently. Probe forwards and generated batches share a GPU lock.
- Cache locks preserve reuse of identical Plain/correction/extraction calls across methods.

## Small train-only validation

Qwen2.5-7B on a spare RTX 4090; fixed Plain prompt, full normal output cap, no test-score selection:

| Requests | Sequential seconds | Batched seconds | Wall-clock ratio | Identical answer text |
|---|---:|---:|---:|---:|
| 4 | 20.933 | 8.467 | 2.47x | 1/4 |
| 8 | 47.040 | 9.295 | 5.06x | 1/8 |

Gemma 3-12B: four train questions took 150.197s sequentially and 71.378s batched (2.10x); 0/4 texts were identical.

These small short-answer checks establish working transport and potential throughput, not an overall speed guarantee or equal task accuracy. Text differences are not evidence that correction scores changed. Long CoT and premise prompts require separate observation; the token budget and OOM fallback handle memory pressure. No additional full benchmark re-generation is justified solely by wording differences.

The Luna extraction smoke test returned a parsed premise list successfully. Following the user's explicit instruction to run the parallel work, v4 services now use Qwen batch 8, Gemma batch 4 and the shared GPT pool of four. See the [v4 runbook](../baseline_v4_runbook_2026-10-12.md) for the active protocol, Haiku regrading, sampling and GEPA settings. Initial service launches lacked CLI paths; replacement `-cli` services include both CLI directories and are running.

## Execution priority

All Table 1/2 generation and optimization precede additional native QA generation. Shared extraction and final Haiku grading run independently of GPU generation. Existing answers and previous judge results are preserved; only compatible answers are reused in the new ledger.
