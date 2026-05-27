#!/usr/bin/env bash
# Symlink the Utonia-side distill module + recipes into the Pointcept submodule
# tree so `python tools/train.py --config-file configs/utonia/...` works without
# duplicating files. Re-runs are idempotent.
#
# Run from the Utonia repo root:
#     bash training/install_into_pointcept.sh
set -euo pipefail

UTONIA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PCEPT_ROOT="${UTONIA_ROOT}/third_party/Pointcept"

if [[ ! -d "${PCEPT_ROOT}/pointcept" ]]; then
    echo "ERROR: Pointcept submodule not initialized at ${PCEPT_ROOT}." >&2
    echo "Run: git submodule update --init --recursive" >&2
    exit 1
fi

link() {
    local src="$1" dst="$2"
    mkdir -p "$(dirname "$dst")"
    if [[ -L "$dst" || -f "$dst" ]]; then
        rm -f "$dst"
    fi
    ln -s "$src" "$dst"
    echo "  linked: ${dst#${PCEPT_ROOT}/} -> ${src#${UTONIA_ROOT}/}"
}

echo "Installing Utonia distill files into ${PCEPT_ROOT}:"

# New distill modules + extended __init__.py for the utonia model package.
link "${UTONIA_ROOT}/training/pointcept/models/utonia/utonia_v1m2_qwen3_5_distill.py" \
     "${PCEPT_ROOT}/pointcept/models/utonia/utonia_v1m2_qwen3_5_distill.py"
link "${UTONIA_ROOT}/training/pointcept/models/utonia/utonia_v1m3a_qwen3_5_align_only.py" \
     "${PCEPT_ROOT}/pointcept/models/utonia/utonia_v1m3a_qwen3_5_align_only.py"
link "${UTONIA_ROOT}/training/pointcept/models/utonia/utonia_v1m3b_qwen3_5_distill_ema.py" \
     "${PCEPT_ROOT}/pointcept/models/utonia/utonia_v1m3b_qwen3_5_distill_ema.py"
link "${UTONIA_ROOT}/training/pointcept/models/utonia/__init__.py" \
     "${PCEPT_ROOT}/pointcept/models/utonia/__init__.py"

# Replacement launch.py — adds DIST_BACKEND env var support (NCCL | gloo).
link "${UTONIA_ROOT}/training/pointcept/engines/launch.py" \
     "${PCEPT_ROOT}/pointcept/engines/launch.py"

# Skip-on-error dataset wrapper — logs and skips bad samples instead of
# crashing the run on a single bad scene.
link "${UTONIA_ROOT}/training/pointcept/datasets/skip_on_error_dataset.py" \
     "${PCEPT_ROOT}/pointcept/datasets/skip_on_error_dataset.py"

# Alignment evaluation script — runs forward on held-out scenes and dumps
# pos/neg patch-cosine histograms.
link "${UTONIA_ROOT}/training/tools/eval_alignment.py" \
     "${PCEPT_ROOT}/tools/eval_alignment.py"

# Comprehensive alignment eval — adds within-scene retrieval (R@K, MRR)
# and linear CKA on top of the simpler eval_alignment.py.
link "${UTONIA_ROOT}/training/tools/eval_alignment_full.py" \
     "${PCEPT_ROOT}/tools/eval_alignment_full.py"

# 2D → 3D retrieval (the direction eval_alignment_full does NOT measure).
# Phase A mirrors eval_alignment_full's R@K on the K×K matrix; Phase B
# expands the candidate set to the full per-point cloud (deployment-
# realistic — the regime open-vocab text→3D actually runs in).
link "${UTONIA_ROOT}/training/tools/eval_2d_to_3d_retrieval.py" \
     "${PCEPT_ROOT}/tools/eval_2d_to_3d_retrieval.py"

# Effective-rank diagnostic — singular-value entropy of the pooled
# f2/f3_proj features.  Distinguishes "collapse" (need retraining)
# from "smooth manifold" (fixable with post-processing).
link "${UTONIA_ROOT}/training/tools/feature_rank_diagnostic.py" \
     "${PCEPT_ROOT}/tools/feature_rank_diagnostic.py"

# Recipes.
for cfg in "${UTONIA_ROOT}/training/configs/utonia/"distill-utonia-v1m3-*.py; do
    [[ -e "$cfg" ]] || continue
    link "$cfg" "${PCEPT_ROOT}/configs/utonia/$(basename "$cfg")"
done

echo
echo "Done. From the Pointcept submodule you can now run:"
echo "    cd ${PCEPT_ROOT}"
echo "    python tools/train.py \\"
echo "        --config-file configs/utonia/distill-utonia-v1m3-1-scannet-only-qwen3_5-4b.py \\"
echo "        --num-gpus <N>"
