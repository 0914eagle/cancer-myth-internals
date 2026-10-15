#!/usr/bin/env bash
set -euo pipefail
cd /home/eagle0914/cancer-myth-internals
QWEN_PYTHON=/data1/heejae/uv/cancer_myth_qwen38_transformers/bin/python
QWEN_RUN=/home/eagle0914/cancer-myth-internals/results/qwen38/full_transformers_20260926_v1
export HF_HOME=/data1/heejae/hf_cache
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
export PATH=/home/eagle0914/.local/bin:/usr/local/bin:/usr/bin:/bin

# The generator itself also uses an exclusive lock. Never launch a duplicate.
if [[ -e "$QWEN_RUN/suite.lock" ]]; then
    echo 'Generation lock still exists; finish/review the previous process first.' >&2
    exit 1
fi
for QWEN_KIND in generation judge; do
    if [[ -f "$QWEN_RUN/$QWEN_KIND.pid" ]] && kill -0 "$(cat "$QWEN_RUN/$QWEN_KIND.pid")" 2>/dev/null; then
        echo "$QWEN_KIND process is already running." >&2
        exit 1
    fi
done

# Verify trust and source integrity once before launch. The frozen, smoke-tested
# kernel then loads locally without an online publisher lookup during generation.
CUDA_VISIBLE_DEVICES='' "$QWEN_PYTHON" scripts/pin_qwen38_kernel.py
export LOCAL_KERNELS=kernels-community/finegrained-fp8=/data1/heejae/hf_cache/hub/kernels--kernels-community--finegrained-fp8/snapshots/8c178950e8710e26a2210e4e909cd02dc16c8715

CUDA_VISIBLE_DEVICES=0,1 nohup setsid "$QWEN_PYTHON" -u scripts/run_qwen38_transformers.py \
    --out-dir "$QWEN_RUN" --batch-size 8 >> "$QWEN_RUN/generation.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$QWEN_RUN/generation.pid"
CUDA_VISIBLE_DEVICES='' nohup setsid "$QWEN_PYTHON" -u scripts/judge_qwen38_answers.py \
    --run-dir "$QWEN_RUN" --paired --workers 5 >> "$QWEN_RUN/judge.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$QWEN_RUN/judge.pid"
sleep 2
kill -0 "$(cat "$QWEN_RUN/generation.pid")"
kill -0 "$(cat "$QWEN_RUN/judge.pid")"
