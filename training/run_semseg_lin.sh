#!/usr/bin/env bash
# Linear/decoder/full-finetune probe launcher for a Utonia checkpoint.
#
# Wraps Pointcept's semseg-utonia-v1m1-* configs so you can verify whether
# a distillation backbone (v1m3-H, etc.) actually produces task-useful 3D
# features by running a ScanNet (or ScanNet200) semantic-segmentation probe
# on top of the FROZEN backbone (linear / decoder) or fine-tuning the whole
# stack.
#
# Usage:
#   bash run_semseg_lin.sh -w /path/to/ckpt.pth                  # default: 0a (scannet linear, frozen backbone)
#   bash run_semseg_lin.sh -w ... -v 0b                          # scannet decoder probe
#   bash run_semseg_lin.sh -w ... -v 0c                          # scannet full fine-tune
#   bash run_semseg_lin.sh -w ... -v 1a                          # scannet200 linear probe
#   bash run_semseg_lin.sh -w ... --num-gpus 8                   # multi-GPU
#   bash run_semseg_lin.sh -w ... epoch=100 eval_epoch=10        # override knobs via --options
#   bash run_semseg_lin.sh -w ... --diag                         # apply diagnostic defaults (epoch=100, eval_epoch=10)
#
# Required:
#   -w / --weight <path>   backbone checkpoint to probe.  Two formats handled:
#                          (a) v1m3-H training save:
#                              keys = `module.student.backbone.*`
#                              Pointcept's default CheckpointLoader entry
#                              (keywords="module.student.backbone", replacement=
#                              "module.backbone") maps these correctly.
#                          (b) Published utonia.pth HF release:
#                              keys = `module.*`  (raw PT-v3).  README says to
#                              change keywords to "module".  Pass --hf-ckpt to
#                              auto-rewrite the symlinked config's
#                              CheckpointLoader hook in-place for you, or set
#                              the keyword via --options if your Pointcept
#                              build supports list-index override.
#
# Variants (-v / --variant):
#   0a  scannet  linear  probe   semseg-utonia-v1m1-0a-scannet-lin.py   (default)
#   0b  scannet  decoder probe   semseg-utonia-v1m1-0b-scannet-dec.py
#   0c  scannet  full fine-tune  semseg-utonia-v1m1-0c-scannet-ft.py
#   1a  scannet200 linear        semseg-utonia-v1m1-1a-scannet200-lin.py
#   1b  scannet200 decoder       semseg-utonia-v1m1-1b-scannet200-dec.py
#   1c  scannet200 full FT       semseg-utonia-v1m1-1c-scannet200-ft.py
#   <X> if not one of the above, looked up as
#         configs/utonia/semseg-utonia-v1m1-${X}.py
#
# Flags:
#   -w | --weight PATH      backbone checkpoint (required).
#   -v | --variant TAG      probe variant (default: 0a).
#   --num-gpus N            GPUs per machine (default: NUM_GPUS env or 1).
#   --diag                  apply diagnostic defaults (epoch=100, eval_epoch=10,
#                           batch_size_val=8) — for "does the backbone help?"
#                           runs, NOT for paper numbers.
#   --hf-ckpt               in-place rewrite of the resolved config's
#                           CheckpointLoader hook so its `keywords` matches the
#                           published utonia.pth key layout (`module.*`).
#                           Does NOT touch the upstream submodule file — only
#                           the symlinked target under Pointcept.  Toggle ONLY
#                           when -w points at the HF release.
#   --utonia-root PATH      override qwen-uton checkout location.
#   --pointcept-local PATH  override local Pointcept mirror.
#   -h | --help             this help.
#
# Anything else after positional flags is forwarded to `train.py --options`
# verbatim, e.g. `bash run_semseg_lin.sh -w ckpt.pth save_path=exp/foo`.
#
# Env vars:
#   DATASET_ROOT     defaults to /group-volume/3Ddataset
#   DIST_BACKEND     default gloo (NCCL broken on this cluster)
#   DIST_TIMEOUT_MIN default 120
#   CONDA_SH, CONDA_ENV, SKIP_CONDA — same semantics as run_train.sh

set -eo pipefail

# ---- Flag parsing -----------------------------------------------------------
UTONIA_ROOT_ARG=""
POINTCEPT_LOCAL_ARG=""
NUM_GPUS_ARG=""
WEIGHT_ARG=""
VARIANT_ARG=""
DIAG=0
HF_CKPT=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--weight)            WEIGHT_ARG="$2"; shift 2 ;;
        --weight=*)             WEIGHT_ARG="${1#--weight=}"; shift ;;
        -v|--variant)           VARIANT_ARG="$2"; shift 2 ;;
        --variant=*)            VARIANT_ARG="${1#--variant=}"; shift ;;
        --num-gpus)             NUM_GPUS_ARG="$2"; shift 2 ;;
        --num-gpus=*)           NUM_GPUS_ARG="${1#--num-gpus=}"; shift ;;
        --utonia-root)          UTONIA_ROOT_ARG="$2"; shift 2 ;;
        --utonia-root=*)        UTONIA_ROOT_ARG="${1#--utonia-root=}"; shift ;;
        --pointcept-local)      POINTCEPT_LOCAL_ARG="$2"; shift 2 ;;
        --pointcept-local=*)    POINTCEPT_LOCAL_ARG="${1#--pointcept-local=}"; shift ;;
        --diag)                 DIAG=1; shift ;;
        --hf-ckpt)              HF_CKPT=1; shift ;;
        --)                     shift; POSITIONAL+=("$@"); break ;;
        -h|--help)              sed -n '2,68p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)                     echo "[error] Unknown flag: $1" >&2; exit 1 ;;
        *)                      POSITIONAL+=("$1"); shift ;;
    esac
done

set -- "${POSITIONAL[@]}"
VARIANT="${VARIANT_ARG:-0a}"
EXTRA_OPTS="$*"

if [[ -z "${WEIGHT_ARG}" ]]; then
    echo "[error] -w / --weight <path> is required (backbone checkpoint to probe)." >&2
    echo "        e.g. /group-volume/Utonia/utonia.pth" >&2
    echo "             exp/utonia_q35_h/model/model_last.pth" >&2
    exit 1
fi
if [[ ! -f "${WEIGHT_ARG}" ]]; then
    echo "[error] checkpoint not found: ${WEIGHT_ARG}" >&2
    exit 1
fi

# ---- Resolve UTONIA_ROOT ----------------------------------------------------
if [[ -n "${UTONIA_ROOT_ARG}" ]]; then
    UTONIA_ROOT="${UTONIA_ROOT_ARG}"
elif [[ -n "${UTONIA_ROOT:-}" ]]; then
    :
else
    UTONIA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
UTONIA_ROOT="$(cd "${UTONIA_ROOT}" 2>/dev/null && pwd || echo "${UTONIA_ROOT}")"
if [[ ! -f "${UTONIA_ROOT}/training/install_into_pointcept.sh" ]]; then
    echo "[error] UTONIA_ROOT=${UTONIA_ROOT} does not look like a qwen-uton checkout" >&2
    exit 1
fi
PCEPT_ROOT="${UTONIA_ROOT}/third_party/Pointcept"

# ---- Conda env --------------------------------------------------------------
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

# ---- Pointcept checkout (same logic as run_train.sh) -----------------------
if [[ -n "${POINTCEPT_LOCAL_ARG}" ]]; then
    POINTCEPT_LOCAL="${POINTCEPT_LOCAL_ARG}"
else
    POINTCEPT_LOCAL="${POINTCEPT_LOCAL:-/group-volume/chaewon.yun/Pointcept_org}"
fi
if [[ ! -d "${PCEPT_ROOT}/pointcept" ]]; then
    if [[ -d "${POINTCEPT_LOCAL}/pointcept" ]]; then
        echo "[setup] Cloning Pointcept from local mirror: ${POINTCEPT_LOCAL}"
        if [[ -e "${PCEPT_ROOT}" && ! -d "${PCEPT_ROOT}/.git" ]]; then
            rmdir "${PCEPT_ROOT}" 2>/dev/null || true
        fi
        git -c protocol.file.allow=always clone \
            --shared "${POINTCEPT_LOCAL}" "${PCEPT_ROOT}"
        SUB_COMMIT="$(git -C "${UTONIA_ROOT}" ls-tree HEAD third_party/Pointcept 2>/dev/null | awk '{print $3}' || true)"
        if [[ -n "${SUB_COMMIT}" ]] && \
                git -C "${PCEPT_ROOT}" cat-file -e "${SUB_COMMIT}" 2>/dev/null; then
            git -C "${PCEPT_ROOT}" checkout -q --detach "${SUB_COMMIT}"
            echo "[setup] Checked out submodule-pinned commit: ${SUB_COMMIT:0:8}"
        else
            echo "[setup] Using mirror's current HEAD ($(git -C "${PCEPT_ROOT}" rev-parse --short HEAD))"
        fi
    else
        echo "[warn] Local Pointcept mirror not found at ${POINTCEPT_LOCAL};"
        echo "       falling back to GitHub clone via git submodule..."
        git -C "${UTONIA_ROOT}" submodule update --init --recursive
    fi
fi

# ---- Install Utonia symlinks (model files + distill configs) ---------------
# Semseg configs already live in the submodule's configs/utonia/, but the
# v1m1 model file lives there too (we don't override it from training/).
# Re-running the installer is cheap and idempotent — keeps v1m3b model wired
# in case anyone resolves a custom variant pointing at it.
echo "[setup] Installing Utonia model files + configs into Pointcept tree..."
bash "${UTONIA_ROOT}/training/install_into_pointcept.sh" >/dev/null

# ---- Dataset symlink --------------------------------------------------------
DATASET_ROOT="${DATASET_ROOT:-/group-volume/3Ddataset}"
if [[ ! -d "${DATASET_ROOT}/data" ]]; then
    echo "[error] DATASET_ROOT=${DATASET_ROOT} has no data/ subdirectory." >&2
    exit 1
fi
if [[ -L "${PCEPT_ROOT}/data" ]]; then
    ln -sfn "${DATASET_ROOT}/data" "${PCEPT_ROOT}/data"
    echo "[setup] Re-linked ${PCEPT_ROOT}/data -> $(readlink "${PCEPT_ROOT}/data")"
elif [[ -e "${PCEPT_ROOT}/data" ]]; then
    echo "[error] ${PCEPT_ROOT}/data exists and is not a symlink; refusing to overwrite." >&2
    exit 1
else
    ln -sfn "${DATASET_ROOT}/data" "${PCEPT_ROOT}/data"
    echo "[setup] Linked ${PCEPT_ROOT}/data -> ${DATASET_ROOT}/data"
fi

# ---- Distributed knobs ------------------------------------------------------
DIST_BACKEND="${DIST_BACKEND:-gloo}"
DIST_TIMEOUT_MIN="${DIST_TIMEOUT_MIN:-120}"
export DATASET_ROOT DIST_BACKEND DIST_TIMEOUT_MIN

# ---- Resolve config ---------------------------------------------------------
# The semseg-utonia configs live under the Pointcept submodule (we don't
# override them from training/).  Look them up there.
case "${VARIANT}" in
    0a) CONFIG="configs/utonia/semseg-utonia-v1m1-0a-scannet-lin.py"     ; DEFAULT_SAVE="exp/probe_scannet_lin"  ;;
    0b) CONFIG="configs/utonia/semseg-utonia-v1m1-0b-scannet-dec.py"     ; DEFAULT_SAVE="exp/probe_scannet_dec"  ;;
    0c) CONFIG="configs/utonia/semseg-utonia-v1m1-0c-scannet-ft.py"      ; DEFAULT_SAVE="exp/probe_scannet_ft"   ;;
    1a) CONFIG="configs/utonia/semseg-utonia-v1m1-1a-scannet200-lin.py"  ; DEFAULT_SAVE="exp/probe_scannet200_lin";;
    1b) CONFIG="configs/utonia/semseg-utonia-v1m1-1b-scannet200-dec.py"  ; DEFAULT_SAVE="exp/probe_scannet200_dec";;
    1c) CONFIG="configs/utonia/semseg-utonia-v1m1-1c-scannet200-ft.py"   ; DEFAULT_SAVE="exp/probe_scannet200_ft" ;;
    *)
        # Free-form variant: try the canonical filename.
        _CAND="${PCEPT_ROOT}/configs/utonia/semseg-utonia-v1m1-${VARIANT}.py"
        if [[ -f "${_CAND}" ]]; then
            CONFIG="configs/utonia/$(basename "${_CAND}")"
            DEFAULT_SAVE="exp/probe_${VARIANT}"
            echo "[setup] Resolved variant '${VARIANT}' -> ${CONFIG}"
        else
            echo "[error] Unknown variant '${VARIANT}'." >&2
            echo "        Expected 0a/0b/0c/1a/1b/1c or a config matching:" >&2
            echo "          ${_CAND}" >&2
            exit 1
        fi
        ;;
esac

if [[ ! -f "${PCEPT_ROOT}/${CONFIG}" ]]; then
    echo "[error] Config not found: ${PCEPT_ROOT}/${CONFIG}" >&2
    exit 1
fi

# ---- Optional: rewrite CheckpointLoader for HF-format utonia.pth -----------
# Published utonia.pth state_dict has keys `module.<raw-PTv3>`; the default
# config expects training-time keys `module.student.backbone.<raw-PTv3>`.
# Pointcept README §4 shows the manual edit; --hf-ckpt automates it on the
# symlinked target so the upstream submodule file stays untouched.
if [[ "${HF_CKPT}" == "1" ]]; then
    # Resolve the symlink to its actual file (the installer may or may not
    # have linked this particular config — semseg configs aren't symlinked
    # by install_into_pointcept.sh, so we edit the file in the submodule).
    _CFG_PATH="${PCEPT_ROOT}/${CONFIG}"
    if [[ -L "${_CFG_PATH}" ]]; then
        _CFG_PATH="$(readlink -f "${_CFG_PATH}")"
    fi
    echo "[setup] --hf-ckpt: patching CheckpointLoader in ${_CFG_PATH}"
    # Match the two lines:
    #   keywords="module.student.backbone",
    #   replacement="module.backbone",
    # and rewrite keywords to "module".  Idempotent — second run is a no-op.
    python - "${_CFG_PATH}" <<'PY'
import re, sys, pathlib
p = pathlib.Path(sys.argv[1])
text = p.read_text()
new = re.sub(
    r'keywords\s*=\s*"module\.student\.backbone"',
    'keywords="module"',
    text,
)
if new != text:
    p.write_text(new)
    print(f"  rewrote keywords -> 'module' in {p}")
else:
    print(f"  (already patched or pattern not found in {p})")
PY
fi

# ---- Diagnostic-default knobs ----------------------------------------------
# Pointcept's recipe defaults to epoch=800, eval_epoch=100 — eval runs 100
# times over the full val set, which is the real bottleneck between iters.
# --diag drops eval to 10x and epochs to 100 for "does this checkpoint give
# better features than random?" runs.  For paper-grade numbers, omit --diag.
if [[ "${DIAG}" == "1" ]]; then
    # Only inject defaults the user hasn't already overridden via positionals.
    [[ "${EXTRA_OPTS}" == *"epoch="*       ]] || EXTRA_OPTS="epoch=100 ${EXTRA_OPTS}"
    [[ "${EXTRA_OPTS}" == *"eval_epoch="*  ]] || EXTRA_OPTS="eval_epoch=10 ${EXTRA_OPTS}"
    [[ "${EXTRA_OPTS}" == *"batch_size_val="* ]] || EXTRA_OPTS="batch_size_val=8 ${EXTRA_OPTS}"
fi

# ---- Default save_path + weight ---------------------------------------------
if [[ "${EXTRA_OPTS}" != *save_path* ]]; then
    EXTRA_OPTS="save_path=${DEFAULT_SAVE} ${EXTRA_OPTS}"
fi
# `weight` is read by Pointcept's CheckpointLoader (see _base_/default_runtime.py).
# Forward as --options instead of a separate flag so it stays composable with
# the rest of the override syntax.
if [[ "${EXTRA_OPTS}" != *"weight="* ]]; then
    EXTRA_OPTS="weight=${WEIGHT_ARG} ${EXTRA_OPTS}"
fi

# ---- NUM_GPUS resolution ----------------------------------------------------
if [[ -n "${NUM_GPUS_ARG}" ]]; then
    NUM_GPUS="${NUM_GPUS_ARG}"
else
    NUM_GPUS="${NUM_GPUS:-1}"
fi

# ---- Launch -----------------------------------------------------------------
cd "${PCEPT_ROOT}"
export PYTHONPATH=./

echo
echo "=========================================================="
echo "  Utonia semseg probe — variant ${VARIANT}"
echo "----------------------------------------------------------"
echo "  cwd               : $(pwd)"
echo "  config            : ${CONFIG}"
echo "  weight            : ${WEIGHT_ARG}"
echo "  --hf-ckpt         : ${HF_CKPT}"
echo "  --diag            : ${DIAG}"
echo "  num_gpus          : ${NUM_GPUS}"
echo "  --options         : ${EXTRA_OPTS}"
echo "  DATASET_ROOT      : ${DATASET_ROOT}"
echo "  DIST_BACKEND      : ${DIST_BACKEND}"
echo "  DIST_TIMEOUT_MIN  : ${DIST_TIMEOUT_MIN}"
echo "=========================================================="
echo

# shellcheck disable=SC2086
python tools/train.py \
    --config-file "${CONFIG}" \
    --num-gpus "${NUM_GPUS}" \
    --options ${EXTRA_OPTS}
