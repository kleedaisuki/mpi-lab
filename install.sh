#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EXTRAS="${EXTRAS:-all}"
INSTALL_TARGET="${ROOT_DIR}"

if [[ -n "${EXTRAS}" ]]; then
    INSTALL_TARGET="${ROOT_DIR}[${EXTRAS}]"
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install --editable "${INSTALL_TARGET}"

echo "Installed mpi-svd-lab into ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"
