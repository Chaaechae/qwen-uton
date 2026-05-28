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
    echo "ERROR: Pointcept source tree not found at ${PCEPT_ROOT}." >&2
    echo "Expected vendored copy under third_party/Pointcept/." >&2
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

# New distill module + extended __init__.py for the utonia model package.
link "${UTONIA_ROOT}/training/pointcept/models/utonia/utonia_v1m2_qwen3_5_distill.py" \
     "${PCEPT_ROOT}/pointcept/models/utonia/utonia_v1m2_qwen3_5_distill.py"
link "${UTONIA_ROOT}/training/pointcept/models/utonia/__init__.py" \
     "${PCEPT_ROOT}/pointcept/models/utonia/__init__.py"

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
