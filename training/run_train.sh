#!/usr/bin/env bash
# Single-shot launcher for Utonia × Qwen3.5 distillation training.
#
# Usage:
#   bash run_train.sh                                # defaults to variant B
#   bash run_train.sh A                              # align-only
#   bash run_train.sh B                              # align + EMA-fixed SSL
#   bash run_train.sh B save_path=exp/foo            # extra train.py --options
#   bash run_train.sh --utonia-root /path/to/repo B  # override repo location
#
# Required env vars (overridable):
#   QWEN3_5_4B_PATH         e.g. /data/hf/Qwen3.5-4B   (or HF repo id)
#   UTONIA_PRETRAINED_CKPT  e.g. /data/ckpt/utonia.pth (strongly recommended)
#
# Optional env vars / overrides:
#   UTONIA_ROOT      path to this qwen-uton checkout. Order of resolution:
#                    (1) --utonia-root flag, (2) UTONIA_ROOT env var,
#                    (3) auto-detect from script location.
#                    Cluster default: /group-volume/chaewon.yun/qwen-uton.
#   POINTCEPT_LOCAL  path to a local Pointcept clone used instead of fetching
#                    from GitHub when initializing the submodule. Order:
#                    (1) --pointcept-local flag, (2) POINTCEPT_LOCAL env,
#                    (3) cluster default /group-volume/chaewon.yun/Pointcept_org.
#                    Falls back to GitHub if no local mirror is found.
#   DATASET_ROOT     (default /group-volume/3Ddataset)
#   DIST_BACKEND     (default gloo; nccl|mpi also valid)
#   NUM_GPUS         (default 1)
#
# What this script does, in order:
#   0. Activate the conda env (skip via SKIP_CONDA=1).
#   1. Initialize the Pointcept submodule if it isn't yet.
#   2. Re-run install_into_pointcept.sh to symlink our model files + configs.
#   3. Symlink ${DATASET_ROOT}/data into Pointcept (Pointcept resolves splits.json
#      paths relative to its own cwd).
#   4. Set PYTHONPATH=./ so `python tools/train.py` can find `pointcept`.
#   5. Launch tools/train.py with the chosen config.

set -eo pipefail

# ---- Flag parsing -----------------------------------------------------------
# Pull out `--utonia-root <path>` (or `--utonia-root=<path>`) and
# `--pointcept-local <path>` before treating the remaining args as
# `[VARIANT] [extra train.py --options ...]`.
UTONIA_ROOT_ARG=""
POINTCEPT_LOCAL_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --utonia-root)
            UTONIA_ROOT_ARG="$2"
            shift 2
            ;;
        --utonia-root=*)
            UTONIA_ROOT_ARG="${1#--utonia-root=}"
            shift
            ;;
        --pointcept-local)
            POINTCEPT_LOCAL_ARG="$2"
            shift 2
            ;;
        --pointcept-local=*)
            POINTCEPT_LOCAL_ARG="${1#--pointcept-local=}"
            shift
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        -*)
            echo "[error] Unknown flag: $1" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

VARIANT="${1:-B}"
shift || true
EXTRA_OPTS="$*"

# ---- Resolve UTONIA_ROOT ----------------------------------------------------
# Priority: --utonia-root flag > UTONIA_ROOT env var > auto-detect from script.
if [[ -n "${UTONIA_ROOT_ARG}" ]]; then
    UTONIA_ROOT="${UTONIA_ROOT_ARG}"
elif [[ -n "${UTONIA_ROOT:-}" ]]; then
    : # use env var as-is
else
    UTONIA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
# Canonicalize and validate.
UTONIA_ROOT="$(cd "${UTONIA_ROOT}" 2>/dev/null && pwd || echo "${UTONIA_ROOT}")"
if [[ ! -f "${UTONIA_ROOT}/training/install_into_pointcept.sh" ]]; then
    echo "[error] UTONIA_ROOT=${UTONIA_ROOT} does not look like a qwen-uton checkout" >&2
    echo "         (missing training/install_into_pointcept.sh)." >&2
    exit 1
fi
PCEPT_ROOT="${UTONIA_ROOT}/third_party/Pointcept"

# ---- 0. Conda env ------------------------------------------------------------
# Activate the conda environment that has torch / spconv / pointops / flash-attn
# / transformers installed for Pointcept. Override via env vars:
#   CONDA_SH      path to conda.sh        (default ~/anaconda3/etc/profile.d/conda.sh)
#   CONDA_ENV     env name or path        (default ~/anaconda3/envs/pointcept)
# Set SKIP_CONDA=1 to skip activation entirely (e.g. when running inside an
# already-activated env or a container).
if [[ "${SKIP_CONDA:-0}" != "1" ]]; then
    CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
    CONDA_ENV="${CONDA_ENV:-$HOME/anaconda3/envs/pointcept}"
    if [[ ! -f "${CONDA_SH}" ]]; then
        echo "[error] conda.sh not found at ${CONDA_SH}. Set CONDA_SH or SKIP_CONDA=1." >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    echo "[setup] Activated conda env: ${CONDA_ENV} ($(python --version 2>&1))"
fi

# ---- 1. Submodule -----------------------------------------------------------
# Prefer a local Pointcept mirror (e.g. /group-volume/chaewon.yun/Pointcept_org
# on this cluster) — clones from there are network-free and faster. The
# submodule URL in .git/config is overridden to point at the local mirror
# before `git submodule update` runs, so the working tree at
# third_party/Pointcept is still a regular checkout (no symlink games),
# meaning install_into_pointcept.sh can safely drop our symlinks into it
# without touching the shared mirror.
#
# Priority:  --pointcept-local flag > POINTCEPT_LOCAL env > cluster default.
if [[ -n "${POINTCEPT_LOCAL_ARG}" ]]; then
    POINTCEPT_LOCAL="${POINTCEPT_LOCAL_ARG}"
else
    POINTCEPT_LOCAL="${POINTCEPT_LOCAL:-/group-volume/chaewon.yun/Pointcept_org}"
fi

if [[ ! -d "${PCEPT_ROOT}/pointcept" ]]; then
    if [[ -d "${POINTCEPT_LOCAL}/pointcept" || -d "${POINTCEPT_LOCAL}/.git" ]]; then
        echo "[setup] Initializing Pointcept submodule from local mirror: ${POINTCEPT_LOCAL}"
        git -C "${UTONIA_ROOT}" submodule init -- third_party/Pointcept
        git -C "${UTONIA_ROOT}" config "submodule.third_party/Pointcept.url" "${POINTCEPT_LOCAL}"
        git -C "${UTONIA_ROOT}" submodule update --recursive third_party/Pointcept
    else
        echo "[warn] Local Pointcept mirror not found at ${POINTCEPT_LOCAL};"
        echo "       falling back to GitHub clone..."
        git -C "${UTONIA_ROOT}" submodule update --init --recursive
    fi
fi

# ---- 2. Install symlinks ----------------------------------------------------
echo "[setup] Installing Utonia model files + configs into Pointcept tree..."
bash "${UTONIA_ROOT}/training/install_into_pointcept.sh" >/dev/null

# ---- 3. Dataset symlink -----------------------------------------------------
# `data_root` in the configs and splits.json reach the data either as an
# absolute path or relative to cwd (third_party/Pointcept). We create both:
# DATASET_ROOT for the configs, and a `./data` symlink for any path inside
# splits.json that starts with literal "data/...".
DATASET_ROOT="${DATASET_ROOT:-/group-volume/3Ddataset}"
if [[ ! -d "${DATASET_ROOT}/data" ]]; then
    echo "[error] DATASET_ROOT=${DATASET_ROOT} has no data/ subdirectory." >&2
    exit 1
fi
if [[ -L "${PCEPT_ROOT}/data" ]]; then
    # Re-point if pointing somewhere else (idempotent for repeated runs).
    ln -sfn "${DATASET_ROOT}/data" "${PCEPT_ROOT}/data"
    echo "[setup] Re-linked ${PCEPT_ROOT}/data -> $(readlink "${PCEPT_ROOT}/data")"
elif [[ -e "${PCEPT_ROOT}/data" ]]; then
    echo "[error] ${PCEPT_ROOT}/data exists and is not a symlink; refusing to overwrite." >&2
    exit 1
else
    ln -sfn "${DATASET_ROOT}/data" "${PCEPT_ROOT}/data"
    echo "[setup] Linked ${PCEPT_ROOT}/data -> ${DATASET_ROOT}/data"
fi

# ---- 4. Env vars ------------------------------------------------------------
# Cluster defaults — override with `export QWEN3_5_4B_PATH=... etc` before
# invoking this script, or pass via the environment.
QWEN3_5_4B_PATH="${QWEN3_5_4B_PATH:-/group-volume/chaewon.yun/QWEN3.5-4B}"
UTONIA_PRETRAINED_CKPT="${UTONIA_PRETRAINED_CKPT:-/group-volume/Utonia/utonia.pth}"
if [[ ! -e "${QWEN3_5_4B_PATH}" ]]; then
    echo "[warn] QWEN3_5_4B_PATH=${QWEN3_5_4B_PATH} does not exist locally;"
    echo "       transformers will try to interpret it as an HF repo id."
fi
if [[ ! -f "${UTONIA_PRETRAINED_CKPT}" ]]; then
    echo "[warn] UTONIA_PRETRAINED_CKPT=${UTONIA_PRETRAINED_CKPT} not found —"
    echo "       student/teacher PTv3 will start from random init."
fi
# Distributed backend. NCCL is broken on this cluster, so default to gloo.
# Override with `DIST_BACKEND=nccl bash run_train.sh ...` if NCCL becomes
# available. Our launch.py (symlinked into Pointcept) reads this env var.
DIST_BACKEND="${DIST_BACKEND:-gloo}"

export QWEN3_5_4B_PATH UTONIA_PRETRAINED_CKPT DATASET_ROOT DIST_BACKEND

# ---- 5. Pick config ---------------------------------------------------------
case "${VARIANT}" in
    A|a)
        CONFIG="configs/utonia/distill-utonia-v1m3-A-scannet-only-qwen3_5-4b.py"
        DEFAULT_SAVE="exp/utonia_q35_align_only"
        ;;
    B|b)
        CONFIG="configs/utonia/distill-utonia-v1m3-B-scannet-only-qwen3_5-4b.py"
        DEFAULT_SAVE="exp/utonia_q35_align_ssl"
        ;;
    *)
        echo "[error] Unknown variant '${VARIANT}'. Expected 'A' or 'B'." >&2
        exit 1
        ;;
esac

# ---- 6. Default save_path if user didn't override ---------------------------
if [[ "${EXTRA_OPTS}" != *save_path* ]]; then
    EXTRA_OPTS="save_path=${DEFAULT_SAVE} ${EXTRA_OPTS}"
fi

NUM_GPUS="${NUM_GPUS:-1}"

# ---- 7. Launch --------------------------------------------------------------
cd "${PCEPT_ROOT}"
export PYTHONPATH=./

echo
echo "=========================================================="
echo "  Utonia × Qwen3.5  —  variant ${VARIANT^^}"
echo "----------------------------------------------------------"
echo "  cwd                  : $(pwd)"
echo "  config               : ${CONFIG}"
echo "  num_gpus             : ${NUM_GPUS}"
echo "  --options            : ${EXTRA_OPTS}"
echo "  QWEN3_5_4B_PATH      : ${QWEN3_5_4B_PATH}"
echo "  UTONIA_PRETRAINED... : ${UTONIA_PRETRAINED_CKPT:-<unset>}"
echo "  POINTCEPT_LOCAL      : ${POINTCEPT_LOCAL}"
echo "  DATASET_ROOT         : ${DATASET_ROOT}"
echo "  DIST_BACKEND         : ${DIST_BACKEND}"
echo "=========================================================="
echo

# shellcheck disable=SC2086
python tools/train.py \
    --config-file "${CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    --options ${EXTRA_OPTS}
