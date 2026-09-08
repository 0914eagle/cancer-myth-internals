#!/usr/bin/env bash
set -euo pipefail

# First-time setup on a GPU server, following medical_nla's layout:
#   code   /home/eagle0914/cancer-myth-internals   (this checkout)
#   data   ${DATA_ROOT}/cancer_myth_internals/{data,external,activations,results,reports,logs}
#   venv   ${DATA_ROOT}/uv/cancer_myth_internals
#   HF     ${DATA_ROOT}/hf_cache
#
#   DATA_ROOT=/data1/heejae bash scripts/bootstrap_server.sh      # server 125
#   DATA_ROOT=/data/heejae  bash scripts/bootstrap_server.sh      # server 62

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT (/data1/heejae on server 125; /data/heejae on server 62)}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ART="${DATA_ROOT}/cancer_myth_internals"
VENV="${DATA_ROOT}/uv/cancer_myth_internals"
INSTALL_ENV="${INSTALL_ENV:-1}"
CLONE_EXTERNAL="${CLONE_EXTERNAL:-1}"
# Same torch as medical_nla/scripts/bootstrap_direct_server.sh. Server 125's
# driver is CUDA 12.2 (nvidia-smi: 535.x), so the wheel must be a cu121 build;
# the PyPI default wheel is built against a newer CUDA and torch then reports
# "The NVIDIA driver on your system is too old" and sees no GPU.
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

mkdir -p "${ART}"/{data,external,activations,results,reports,logs} "${DATA_ROOT}/hf_cache" "${DATA_ROOT}/uv"

if [[ "${INSTALL_ENV}" == "1" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env"
  fi
  if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv "${VENV}" --python 3.11
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  cd "${CODE_ROOT}"
  # torch first, pinned to the driver-compatible build; the editable install
  # below then keeps it (torch>=2.3 is satisfied) instead of resolving anew.
  uv pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
  uv pip install -e ".[dev]"
  python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print(f"[torch] {torch.__version__} cuda={ok} devices={torch.cuda.device_count() if ok else 0}")
if not ok:
    raise SystemExit("[error] torch sees no GPU. If the warning says the driver is too old, "
                     "rerun with a matching build, e.g. TORCH_VERSION=2.5.1 "
                     "TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121")
EOF
fi

if [[ "${CLONE_EXTERNAL}" == "1" ]]; then
  cd "${ART}/external"
  [[ -d cancer-myth ]] || git clone --depth 1 https://github.com/bill1235813/cancer-myth
  [[ -d Well ]] || git clone --depth 1 https://github.com/ShenranTomWang/Well
  echo "[external] $(ls)"
fi

cd "${CODE_ROOT}"
# shellcheck disable=SC1091
source scripts/env.sh "${DATA_ROOT}"

# Gated checkpoints (Llama-3.1, Gemma-2) need the account that accepted the
# licences; the token lives under $HF_HOME.
if ! python -c "import huggingface_hub as h; print('[hf] logged in as', h.whoami()['name'])" 2>/dev/null; then
  echo "[hf] not logged in. Run:  huggingface-cli login   (or: hf auth login)"
fi

python scripts/check_gpu_setup.py --config configs/llama31_8b.yaml || true
echo "[bootstrap] done. Next: bash scripts/run_e0_rows.sh"
