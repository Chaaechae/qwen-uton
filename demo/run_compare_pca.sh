#!/usr/bin/env bash
# Wrapper for demo/compare_pca.py that activates the same conda env as
# training (so the GPU-enabled spconv-cuXXX is on path) before running.
#
# Usage examples (forwarded to demo/compare_pca.py):
#   bash demo/run_compare_pca.sh \
#       --v1m3a exp/utonia_q35_align_only/model/model_last.pth \
#       --v1m3b exp/utonia_q35_align_ssl/model/model_last.pth \
#       --save compare.html
#
#   bash demo/run_compare_pca.sh --utonia "" --v1m3a ... --save compare.png
#
# Env overrides (same convention as run_train.sh):
#   CONDA_SH      default ~/anaconda3/etc/profile.d/conda.sh
#   CONDA_ENV     default ~/anaconda3/envs/pointcept
#   SKIP_CONDA=1  bypass activation entirely (already inside the env)

set -eo pipefail

UTONIA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
    CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
    CONDA_ENV="${CONDA_ENV:-$HOME/anaconda3/envs/pointcept}"
    if [[ ! -f "${CONDA_SH}" ]]; then
        echo "[error] conda.sh not found at ${CONDA_SH}. " \
             "Set CONDA_SH or SKIP_CONDA=1." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    echo "[setup] Activated conda env: ${CONDA_ENV} ($(python --version 2>&1))"
fi

python "${UTONIA_ROOT}/demo/compare_pca.py" "$@"
