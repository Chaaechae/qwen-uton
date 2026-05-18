"""
DINOv2 reference eval config — for use with eval_alignment_full.py.

Purpose
-------
Provides a "known-good" alignment baseline by running the EXACT same
eval pipeline (eval_alignment_full.py) against the *published* Utonia
checkpoint (`/group-volume/Utonia/utonia.pth`).  The published model
was trained with DINOv2-giant-with-registers as the 2D teacher, and
is the model Pointcept reports downstream wins for — so its
alignment numbers (pos / neg / discrim / CKA / retrieval) serve as
the realistic upper bound we want our Qwen H to approach.

Model + 2D teacher
------------------
- Model class: `Utonia-v1m1` (the original Utonia 1.0 architecture;
  same PT-v3m3 backbone, no merger, no qwen_proj, no use_full_merger
  knobs).  `eval_alignment_full.py` uses getattr-fallbacks for those
  v1m3b-only attributes, so the same script works against v1m1.
- 2D teacher: DINOv2 ViT-giant + 4 registers, loaded via
  transformers AutoModel from
  `/group-volume/chaewon.yun/dinov2-with-registers-giant`.
- patch_h = patch_w = 518 // 14 = 37  (matches the published
  training resolution).
- enc2d_head_in_channels = 1536 (DINOv2-G hidden dim).

Caveats on the published checkpoint
-----------------------------------
The HuggingFace release `utonia.pth` typically ships only the
*backbone* state_dict (Pointcept's README explicitly maps `module.*`
into `module.backbone.*` for downstream tasks).  If patch_proj is
not in the checkpoint, it will be randomly initialized in this eval,
which makes pos / neg / retrieval / cka_aligned_proj uninterpretable.
**The fair comparison metric is `cka_aligned_backbone_vs_qwen` — that
one only depends on backbone features and is directly comparable
across H and this reference.**

To verify what's in the checkpoint, run this snippet first:
    import torch
    ckpt = torch.load("/group-volume/Utonia/utonia.pth",
                      map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    print("# keys", len(sd))
    print("has patch_proj:",
          any("patch_proj" in k for k in sd.keys()))

Dataset
-------
Single ScanNet-only branch (SkipOnErrorImagePointDataset, same one
H uses) so the eval covers the same scene distribution.  IMAGENET
normalization (NOT Qwen 0.5/0.5/0.5 stats) — DINOv2 was trained on
ImageNet-standardized inputs.
"""

_base_ = ["../_base_/default_runtime.py"]

import os

DINOV2_PATH = os.environ.get(
    "DINOV2_PATH", "/group-volume/chaewon.yun/dinov2-with-registers-giant"
)
DATASET_ROOT = os.environ.get("DATASET_ROOT", "/group-volume/3Ddataset")
del os

# 518 = 14 * 37 — same crop/patch geometry the published v1m1 was
# trained at. Don't reuse v1m3-H's 512/16 here — patch_size MUST
# divide crop_h, and DINOv2-G uses 14-pixel patches.
crop_h = 518
crop_w = 518
patch_size = 14
batch_size = 16
num_worker = 16
mix_prob = 0.0
clip_grad = 1.0

empty_cache = True
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = True

train = dict(type="DefaultTrainer")
enable_wandb = False

# Match v1m1's pretrain stagev2 model spec (same dims, same upcast
# levels) so the published utonia.pth state_dict snaps in cleanly.
model = dict(
    type="Utonia-v1m1",
    patch_h=crop_h // patch_size,
    patch_w=crop_w // patch_size,
    image_weight_name="dinov2_vitg14_reg",
    image_weight_path=DINOV2_PATH,
    backbone_out_channels=1332,
    embedding_channels=64,
    student_pretrained=False,
    enc2d_upcast_level=3,
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=True,
        traceable=True,
        mask_token=True,
        rope_base=10,
        shift_coords=None,
        jitter_coords=1.1,
        rescale_coords=1.2,
    ),
    teacher_custom=dict(attn_drop=0.0, proj_drop=0.0, drop_path=0.0),
    head_in_channels=576,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    enc2d_head_in_channels=1536,
    enc2d_head_hidden_channels=4096,
    enc2d_head_embed_channels=256,
    enc2d_head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=10,
    mask_size_base=40,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.5,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=1 / 8,
    roll_mask_loss_weight=1 / 8,
    unmask_loss_weight=2 / 8,
    enc2d_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=0,
    enc2d_cos_shift=True,
    sonata_model_type="online",
)

# Eval has no real training, but Pointcept's config parser expects
# scheduler/optimizer blocks. Stub them out with minimal values.
epoch = 1
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.04)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[1e-4],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = []

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

indoor_transform = [
    dict(
        type="Update",
        keys_dict={
            "index_valid_keys": (
                "coord",
                "origin_coord",
                "color",
                "normal",
                "superpoint",
                "strength",
                "segment",
                "instance",
                "correspondence",
                "global_correspondence",
            )
        },
    ),
    dict(
        type="ImgAugmentation",
        crop_h=crop_h,
        crop_w=crop_w,
        patch_h=crop_h // patch_size,
        patch_w=crop_w // patch_size,
        patch_size=patch_size,
        imgtransforms=[
            dict(type="ImgChromaticJitter", p=0.95, std=0.05),
            dict(type="ImgGaussianBlur", p=0.5),
            dict(
                type="Imgnormalize",
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ],
    ),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(type="RandomScale", scale=[0.45, 0.55]),
    dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train"),
    dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropColor", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(type="RandomDropNormal", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropNormal", drop_ratio=0.1, drop_application_ratio=0.5),
    dict(
        type="MultiViewGenerator",
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            dict(type="ElasticDistortion", distortion_params=[[0.1, 0.2], [0.4, 0.8]]),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            dict(type="ElasticDistortion", distortion_params=[[0.1, 0.2], [0.4, 0.8]]),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        max_size=16384,
        enc2d_max_size=16384,
        enc2d_scale=(0.8, 1),
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.01}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_offset",
            "grid_size",
            "name",
            "images",
            "global_correspondence",
            "img_num",
        ),
        offset_keys_dict=dict(),
        global_feat_full_keys=("global_coord", "global_color", "global_normal"),
        global_feat_keys=("global_coord", "global_color", "global_normal"),
        local_feat_keys=("local_coord", "local_color", "local_normal"),
    ),
]

data_weight = None
data_length = None
data = dict(
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="SkipOnErrorImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["train", "val"],
                data_root=f"{DATASET_ROOT}/data/scannet",
                transform=indoor_transform,
                test_mode=False,
                loop=1,
            ),
        ],
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
]
