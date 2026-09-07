# Source this at the start of every session:  source scripts/env.sh [DATA_ROOT]
#
# Same contract as medical_nla/scripts/env.sh, with the CANCER_MYTH_ prefix.
# The data root is found from the uv virtualenv created once per machine
# underneath it; pass it as an argument to say it explicitly:
#
#   source scripts/env.sh                  # find it
#   source scripts/env.sh /data1/heejae    # say it (server 125)
#
# Passed as an argument, not as `VAR=... source scripts/env.sh`: `source` is a
# builtin, so a prefixed assignment is gone by the next command.

_CMI_FOUND=""
case "${1:-}" in
  /*) CANCER_MYTH_DATA_ROOT="$1"; _CMI_FOUND="given as an argument" ;;
esac
if [ -z "${CANCER_MYTH_DATA_ROOT:-}" ]; then
  _CMI_CANDS=$(ls -d /data*/*/uv/cancer_myth_internals /data*/uv/cancer_myth_internals 2>/dev/null \
               | sed 's|/uv/cancer_myth_internals$||')
  _CMI_N=$(printf '%s\n' "$_CMI_CANDS" | grep -c . || true)
  if [ "${_CMI_N:-0}" -ge 1 ]; then
    CANCER_MYTH_DATA_ROOT=$(printf '%s\n' "$_CMI_CANDS" | head -1)
    _CMI_FOUND="found via $CANCER_MYTH_DATA_ROOT/uv/cancer_myth_internals"
    if [ "$_CMI_N" -gt 1 ]; then
      echo "[env] several candidate roots on this host:"
      printf '[env]   %s\n' $_CMI_CANDS
      echo "[env] using the first; set CANCER_MYTH_DATA_ROOT to override"
    fi
  else
    CANCER_MYTH_DATA_ROOT=/data1/heejae
    _CMI_FOUND="default (no uv/cancer_myth_internals found on this host)"
  fi
else
  _CMI_FOUND="from the environment"
fi
export CANCER_MYTH_DATA_ROOT
unset _CMI_CANDS _CMI_N

if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _CMI_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  _CMI_HERE="$PWD"
fi
export CANCER_MYTH_CODE_ROOT="${CANCER_MYTH_CODE_ROOT:-$_CMI_HERE}"
unset _CMI_HERE

export PYTHONPATH="$CANCER_MYTH_CODE_ROOT"

# HF_HOME only; TRANSFORMERS_CACHE means something different and doubles the
# cache (medical_nla found 46GB of duplicates that way).
unset TRANSFORMERS_CACHE
export HF_HOME="$CANCER_MYTH_DATA_ROOT/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# One card by default: every E1 backbone but Gemma-2-27B / Gemma-3-12B fits
# one 24GB 4090. max_memory keys in the configs index the *visible* devices,
# so a job on another card is `export CUDA_VISIBLE_DEVICES=2; source scripts/env.sh`.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export RAW="$CANCER_MYTH_DATA_ROOT"
export ART="$CANCER_MYTH_DATA_ROOT/cancer_myth_internals"
export DATA="$ART/data"

echo "[env] host      $(hostname)"
echo "[env] data root $CANCER_MYTH_DATA_ROOT $([ -d "$CANCER_MYTH_DATA_ROOT" ] || echo '(MISSING -- wrong machine?)')"
echo "[env]           $_CMI_FOUND"
echo "[env] code root $CANCER_MYTH_CODE_ROOT"
echo "[env] hf cache  $HF_HOME"
echo "[env] gpus      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[env] judge     codex=$(command -v codex >/dev/null 2>&1 && echo "$(command -v codex)" || echo 'NOT ON PATH') | OPENAI_API_KEY=$([ -n "${OPENAI_API_KEY:-}" ] && echo set || echo unset)"
unset _CMI_FOUND
