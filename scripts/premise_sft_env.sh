#!/usr/bin/env bash
# Source this from the repository root. Do not source the older scripts/env.sh.
export PREMISE_ROOT="${PREMISE_ROOT:-/data3/heejae}"
export PREMISE_ENV="${PREMISE_ROOT}/uv/cancer-myth"
export PREMISE_ART="${PREMISE_ROOT}/cancer-myth"
export HF_HOME="${PREMISE_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${PREMISE_ART}/datasets_cache"
export UV_CACHE_DIR="${PREMISE_ROOT}/uv_cache"
export UV_PYTHON_INSTALL_DIR="${PREMISE_ROOT}/uv_python"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PATH="${PREMISE_ROOT}/tools:${PATH}"
