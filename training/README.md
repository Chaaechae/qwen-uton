# Utonia + Qwen3.5 ViT distillation training recipe

This directory holds the training-side additions used to fine-tune Utonia with
the Qwen3.5 (unified VLM) vision tower as the 2D distillation target. Pointcept
itself is brought in as a git submodule under `third_party/Pointcept/`, and an
install script symlinks our model module + recipes into the Pointcept tree —
so a single `git clone --recursive` of this repo plus `bash
training/install_into_pointcept.sh` gives you a runnable Pointcept checkout
without copying files around.

## Layout

```
training/
├── configs/utonia/
│   ├── distill-utonia-v1m3-0-base-qwen3_5-4b.py          # 16-dataset full base recipe
│   └── distill-utonia-v1m3-1-scannet-only-qwen3_5-4b.py  # ScanNet-only sanity recipe
├── pointcept/models/utonia/
│   ├── __init__.py                                        # exposes Utonia-v1m2_qwen3_5_distill
│   └── utonia_v1m2_qwen3_5_distill.py                     # the model module
├── install_into_pointcept.sh                              # symlink installer
└── README.md                                              # this file

third_party/
└── Pointcept/                                             # git submodule

tools/
└── test_qwen3_5_vit_path.py                               # standalone Qwen3.5 ViT smoke test
```

## One-time setup

```bash
# After cloning Utonia (or if you cloned without --recursive):
git submodule update --init --recursive

# Symlink our model + configs into the Pointcept tree.
bash training/install_into_pointcept.sh

# Install Pointcept's runtime deps separately (torch, spconv, pointops,
# torch_scatter, flash-attn, transformers ≥ 4.57.0.dev). Follow:
#   third_party/Pointcept/README.md  (Installation section)
```

## Design (recap)

- **Student / teacher (PTv3)**: both at Utonia base channels
  `(54, 108, 216, 432, 576)`. SSL losses (mask, roll-mask, unmask) are kept on
  alongside the 2D-3D alignment loss; weights
  `{mask, roll, unmask, enc2d} = {1/8, 1/8, 2/8, 4/8}`.
- **What "teacher" means here**: Concerto-v1m2_distill keeps the standard DINO-
  style teacher = EMA of student, but additionally lets you *initialize* the
  teacher from a pretrained checkpoint for a much faster warm-start. Setting
  `UTONIA_TEACHER_CKPT` is therefore **optional but strongly recommended** —
  with it, the very first batch already gives a meaningful SSL target.
- **2D teacher**: `Qwen/Qwen3.5-4B` vision tower, used pre-merge so patch
  tokens are `1024-d` on a `patch=16` grid (before the `merge_size=2` reshape
  that produces the LM input). Loaded via `AutoModelForImageTextToText`, only
  `model.visual` is kept; the LM is freed.
- **Image preprocessing**: Qwen3.5 preprocessor uses `mean=std=(0.5, 0.5, 0.5)`,
  not ImageNet stats — recipes wire `QWEN_VIT_MEAN/STD` into every
  `ImgAugmentation`'s `Imgnormalize`.
- **Crop**: `512×512`, `patch_size=16` → `32×32 = 1024` patch tokens/view.
- **Upcast level**: `enc2d_upcast_level=3` → 1332-d student feature aligned
  with Qwen3.5 1024-d patch; stage 0 (embedding) excluded.

## External resources (env-var overrides)

Both recipes resolve the Qwen3.5 weight and the optional Utonia warm-start
ckpt from environment variables, so you don't edit the config to change paths:

```bash
export QWEN3_5_4B_PATH=/data/hf/Qwen3.5-4B          # local path or HF repo id
export UTONIA_TEACHER_CKPT=/data/ckpt/utonia.pth    # optional; recommended
```

Or pass them inline via Pointcept's CLI:

```bash
python tools/train.py --config-file ... \
    --options model.image_weight_path=/data/hf/Qwen3.5-4B \
              model.teacher_pretrained_path=/data/ckpt/utonia.pth
```

`UTONIA_TEACHER_CKPT` should point to the Utonia HuggingFace checkpoint
(`Pointcept/Utonia → utonia.pth`); it is loaded *only* into the teacher's
PTv3 backbone — student is trained from scratch. If you leave it unset, the
teacher starts at random init and is pulled toward student via EMA only;
training still works but converges much more slowly.

## ScanNet sanity-check (recommended first run)

Use the ScanNet-only recipe for a 1-GPU end-to-end smoke run before scaling
to the 16-dataset base recipe.

### 1. Preprocess ScanNet

```bash
cd third_party/Pointcept
bash pointcept/datasets/preprocessing/concerto/scannet/preprocess_scannet.sh \
    -d <RAW_SCANNET_DIR> \
    -o data/scannet \
    -n 32 \
    -c
python pointcept/datasets/preprocessing/concerto/scannet/splits.py \
    --dataset_root data/scannet
```

Output layout under `third_party/Pointcept/data/scannet/`:

```
data/scannet/
├── train/<scene_id>.pth            # point cloud
├── val/<scene_id>.pth
├── test/<scene_id>.pth
├── splits/{train,val,test}.json
└── images/{train,val,test}/
    ├── color/<scene_id>/<frame>.jpg
    ├── correspondence/<scene_id>/<frame>.npy   # pixel ↔ point index
    ├── intrinsic/<scene_id>/<frame>.txt
    └── pose/<scene_id>/<frame>.txt
```

### 2. Pre-flight: Qwen3.5 ViT forward path

```bash
cd <utonia-repo>
python tools/test_qwen3_5_vit_path.py --model "$QWEN3_5_4B_PATH"
# Expect [4/4 PASS] with strategy=[manual: position_embeddings=(cos,sin) ...]
```

### 3. (Optional but recommended) Get the Utonia warm-start ckpt

```bash
huggingface-cli download Pointcept/Utonia utonia.pth --local-dir /data/ckpt
export UTONIA_TEACHER_CKPT=/data/ckpt/utonia.pth
```

### 4. Launch ScanNet-only sanity run

```bash
cd third_party/Pointcept
export QWEN3_5_4B_PATH=/data/hf/Qwen3.5-4B
# UTONIA_TEACHER_CKPT optional — leave unset to skip teacher warm-start.

python tools/train.py \
    --config-file configs/utonia/distill-utonia-v1m3-1-scannet-only-qwen3_5-4b.py \
    --num-gpus 1 \
    --options save_path=exp/utonia_q35_scannet_sanity
```

What to verify in the first ~100 steps:
- `loss` finite, decreasing.
- `enc2d_loss` non-zero and decreasing (this is the 2D-3D alignment).
- `mask_loss` / `unmask_loss` finite (no NaN). If teacher warm-start is
  disabled, expect these to be near-uniform initially and only start to
  drop after a few hundred steps.

## Scaling to the full 16-dataset base recipe

After the ScanNet sanity run is healthy, switch to
`configs/utonia/distill-utonia-v1m3-0-base-qwen3_5-4b.py` and bring up the
remaining datasets one at a time. The full recipe assumes `data/{nuscenes,
semantic_kitti, waymo, cap3d, partnet_data_v0, graspnet, scanobjectnn_raw,
arkitscenes, scannet, scannetpp, s3dis, hm3d_fix, structured3d, re10k_align,
hk_3d_maps_N}` are all preprocessed in the Concerto/Utonia layout.

## Downstream consumers (precompute)

The published Utonia inference path concats *all* 5 encoder stages back to
the input grid (1386-d). This recipe trains an `enc2d_upcast_level=3`
(1332-d) representation, so any downstream precompute that consumes the
distilled checkpoint must be modified to perform exactly **3** upcast
iterations — e.g. in `precompute_utonia_features.py`, replace the open-ended
`while "pooling_parent" in point.keys(): ...` with `for _ in range(3): ...`,
which produces a 1332-d per-grid feature that matches what the 2D-3D align
head saw during training.
