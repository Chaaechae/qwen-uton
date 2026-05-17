"""
Version D — Batch-centered cosine pull.

The cosine pull `(1 - cos(f2, f3)).mean() * 10` in v1m3-B collapsed to
a trivial "project everything toward Qwen's mean direction" minimum
on Qwen's highly anisotropic features (mean pairwise cos ~0.9). v1m3-C
switched to InfoNCE to break that, but per-scene InfoNCE turned out
to have very weak gradient signal (within-scene Qwen patches all look
similar — too hard).

v1m3-D keeps cosine pull simplicity but subtracts the BATCH mean
(across all patches in the InfoNCE batch) before computing cosine.
The mean-direction cheat now produces a zero vector after centering,
so cos collapses to 0 (worst-case loss) instead of ~0.97. The
optimizer is forced to learn per-patch alignment.

Architecture, SSL losses, EMA, warm-start, loss weights all identical
to v1m3-B / v1m3-C. Only `enc2d_loss_type` flips to "cosine_bc".
"""

_base_ = ["../_base_/default_runtime.py"]

import os

QWEN3_5_4B_PATH = os.environ.get("QWEN3_5_4B_PATH", "Qwen/Qwen3.5-4B")
UTONIA_PRETRAINED_CKPT = os.environ.get("UTONIA_PRETRAINED_CKPT", None)
UTONIA_STUDENT_CKPT = os.environ.get("UTONIA_STUDENT_CKPT", UTONIA_PRETRAINED_CKPT)
UTONIA_TEACHER_CKPT = os.environ.get("UTONIA_TEACHER_CKPT", UTONIA_PRETRAINED_CKPT)
# Dataset root. Real layout on this cluster is
#   /group-volume/chaewon.yun/dataset/data/scannet/{train,val,images,...}
# so the config-side `data_root` reaches scannet at
#   ${DATASET_ROOT}/data/scannet
DATASET_ROOT = os.environ.get("DATASET_ROOT", "/group-volume/3Ddataset")
del os

crop_h = 512
crop_w = 512
patch_size = 16
batch_size = 8
num_worker = 8
mix_prob = 0.0
clip_grad = 1.0

empty_cache = True
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = True

train = dict(type="DefaultTrainer")
enable_wandb = False

model = dict(
    type="Utonia-v1m3b_qwen3_5_distill_ema",
    patch_h=crop_h // patch_size,
    patch_w=crop_w // patch_size,
    image_weight_name="qwen3_5_4b_vit",
    image_weight_path=QWEN3_5_4B_PATH,
    backbone_out_channels=1332,
    enc2d_upcast_level=3,
    backbone_s=dict(
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
    backbone_t=dict(
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
    head_in_channels_s=576,
    head_in_channels_t=576,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    enc2d_head_in_channels=1024,
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
    # Re-balanced for Qwen-as-2D-teacher: 75% alignment, 25% SSL regularizer.
    mask_loss_weight=1 / 16,
    roll_mask_loss_weight=1 / 16,
    unmask_loss_weight=1 / 8,
    enc2d_loss_weight=3 / 4,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=0,
    enc2d_cos_shift=True,
    # Switch enc2d alignment loss: InfoNCE (CLIP-style symmetric) instead of
    # the cosine-pull. Anti-collapse — penalizes the mean-direction trivial
    # solution that v1m3-B converged to.
    enc2d_loss_type="cosine_bc",
    # Lower than CLIP's default 0.07. Qwen-Qwen pairwise cos is ~0.9 so
    # logits need sharper scaling to expose discriminative differences.
    infonce_temperature=0.03,
    ema_teacher_backbone=True,
    student_pretrained_path=UTONIA_STUDENT_CKPT,
    teacher_pretrained_path=UTONIA_TEACHER_CKPT,
)

epoch = 5
eval_epoch = 5
base_lr = 0.004
backbone_lr_scale = 0.05
lr_decay = 0.9

base_wd = 0.04
final_wd = 0.2

backbone_base_lr = base_lr * backbone_lr_scale
dec_depths = model["backbone_s"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=backbone_base_lr
        * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
param_dicts += [
    dict(keyword="student.backbone.", lr=backbone_base_lr),
    dict(keyword="teacher.backbone.", lr=backbone_base_lr),
    dict(keyword="patch_proj.", lr=base_lr),
    dict(keyword="student.mask_head.", lr=base_lr),
    dict(keyword="student.unmask_head.", lr=base_lr),
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
QWEN_VIT_MEAN = (0.5, 0.5, 0.5)
QWEN_VIT_STD = (0.5, 0.5, 0.5)

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
                mean=QWEN_VIT_MEAN,
                std=QWEN_VIT_STD,
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
        max_size=65536,
        enc2d_max_size=65536,
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
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=10),
]
