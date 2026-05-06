# Utonia + Qwen3.5 ViT distillation training recipe

This directory tracks the training-side additions used to fine-tune Utonia with
the Qwen3.5 (unified VLM) vision tower as the 2D distillation target. The files
are organized to mirror the
[Pointcept](https://github.com/Pointcept/Pointcept) tree so they can be dropped
in on top of a Pointcept checkout. The Utonia inference repo itself does not
ship the Pointcept training engine — apply these onto a separate Pointcept
clone to actually run training.

## What's here

```
training/
├── configs/utonia/
│   └── distill-utonia-v1m3-0-base-qwen3_5-4b.py   # base recipe, target=Qwen3.5-4B ViT
└── pointcept/models/utonia/
    ├── __init__.py                                 # exposes the new module
    └── utonia_v1m2_qwen3_5_distill.py              # Utonia-v1m2_qwen3_5_distill model
```

The standalone Qwen3.5 ViT smoke test lives at the repo root: `tools/test_qwen3_5_vit_path.py`.

## Design

- **Student / teacher (PTv3)**: both at Utonia base channels `(54, 108, 216, 432, 576)`.
  SSL losses (mask, roll-mask, unmask) are kept on alongside the 2D-3D
  alignment loss; weights `{mask, roll, unmask, enc2d} = {1/8, 1/8, 2/8, 4/8}`.
- **2D teacher**: `Qwen/Qwen3.5-4B` vision tower, used pre-merge so that
  patch tokens are `1024-d` on a `patch=16` grid (i.e. before the `merge_size=2`
  reshape that produces the LM input). Loaded once via
  `AutoModelForImageTextToText`, only `model.visual` is kept; the LM is freed.
- **Image preprocessing**: Qwen3.5 preprocessor uses `mean=std=(0.5, 0.5, 0.5)`,
  not ImageNet stats — the recipe wires `QWEN_VIT_MEAN/STD` into every
  `ImgAugmentation`'s `Imgnormalize`.
- **Crop**: `512×512`, `patch_size=16` → `32×32 = 1024` patch tokens per view
  (well within the Qwen3.5 ViT's `num_position_embeddings=2304`).
- **Upcast level**: `enc2d_upcast_level=3` (concat encoder stages 1..4),
  giving a 1332-d student feature that aligns to the Qwen3.5 1024-d patch.
  Stage 0 (embedding output) is intentionally excluded — it is too local to
  match a ViT patch's receptive field.

## Apply onto a Pointcept clone

```bash
git clone https://github.com/Pointcept/Pointcept.git ~/work/Pointcept
cd ~/work/Pointcept

# Drop in the new model module + extended __init__.py
cp <utonia-repo>/training/pointcept/models/utonia/utonia_v1m2_qwen3_5_distill.py \
   pointcept/models/utonia/
cp <utonia-repo>/training/pointcept/models/utonia/__init__.py \
   pointcept/models/utonia/__init__.py

# Drop in the recipe
cp <utonia-repo>/training/configs/utonia/distill-utonia-v1m3-0-base-qwen3_5-4b.py \
   configs/utonia/
```

Then launch via Pointcept's normal trainer entrypoint, e.g.

```bash
python tools/train.py \
    --config-file configs/utonia/distill-utonia-v1m3-0-base-qwen3_5-4b.py \
    --num-gpus 8 \
    --options save_path=exp/utonia_qwen3_5_4b
```

## Smoke-test the Qwen3.5 ViT forward path first

Before running the full distillation, verify our `ENC2D_forward` path matches
the actually-installed transformers version:

```bash
cd <utonia-repo>
python tools/test_qwen3_5_vit_path.py --model Qwen/Qwen3.5-4B --crop 512 --batch 2
# CPU only:  --device cpu --dtype float32   (needs ~16GB RAM)
# Tight VRAM:  --batch 1 --crop 256
```

The script independently checks: full VLM load → `model.visual` lookup →
required sub-modules (`patch_embed`, `blocks`, `rot_pos_emb` /
`rotary_pos_emb`, `config`) → end-to-end forward producing
`(B, h*w, 1024)`. If `rot_pos_emb` is named differently in your transformers
version, the script reports the correct attribute name; update the one
`getattr(self.enc2d_model, rope_attr)` line in
`utonia_v1m2_qwen3_5_distill.py::ENC2D_forward` accordingly.

## Downstream consumers (precompute)

The published Utonia inference path concats *all* 5 encoder stages back to the
input grid (1386-d). This recipe trains an `enc2d_upcast_level=3` (1332-d)
representation, so any downstream precompute that consumes the distilled
checkpoint must be modified to perform exactly **3** upcast iterations — e.g.
in `precompute_utonia_features.py`, replace the open-ended
`while "pooling_parent" in point.keys(): ...` with `for _ in range(3): ...`,
which produces a 1332-d per-grid feature that matches what the 2D-3D align
head saw during training.
