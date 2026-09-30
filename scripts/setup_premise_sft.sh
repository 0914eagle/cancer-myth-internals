#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/premise_sft_env.sh
mkdir -p "${PREMISE_ART}"/{runs,logs} "${HF_HOME}" "${PREMISE_ROOT}/tools" "${PREMISE_ROOT}/uv"
# uv bootstrap stays under the requested data root. Already installed uv is reused.
if ! command -v uv >/dev/null 2>&1; then
  curl --fail --location --silent --show-error https://astral.sh/uv/0.6.17/install.sh -o "${PREMISE_ROOT}/tools/install-uv.sh"
  UV_INSTALL_DIR="${PREMISE_ROOT}/tools" UV_NO_MODIFY_PATH=1 sh "${PREMISE_ROOT}/tools/install-uv.sh"
fi
if [[ ! -x "${PREMISE_ENV}/bin/python" ]]; then
  uv venv --python 3.11.11 "${PREMISE_ENV}"
fi
uv pip install --python "${PREMISE_ENV}/bin/python" 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
# Pin torch again during dependency resolution: no latest-CUDA upgrade.
uv pip install --python "${PREMISE_ENV}/bin/python" -r configs/premise_sft/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
uv pip check --python "${PREMISE_ENV}/bin/python"
uv pip freeze --python "${PREMISE_ENV}/bin/python" > "${PREMISE_ART}/environment.freeze.txt"
uv --version > "${PREMISE_ART}/uv-version.txt"
"${PREMISE_ENV}/bin/python" scripts/premise_sft.py check --out "${PREMISE_ART}/environment.json" --gpus 2
echo "Environment ready. Next: source scripts/premise_sft_env.sh"
