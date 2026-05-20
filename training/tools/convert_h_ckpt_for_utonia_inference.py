"""
Convert a v1m3-H Pointcept training save into the clean Utonia-inference
checkpoint format that `utonia.load(path)` expects.

Why this exists
---------------
`utonia/model.py:load(...)` loads a checkpoint shaped like::

    {
        "config":     {<PointTransformerV3 __init__ kwargs>},
        "state_dict": {<raw PT-v3 weights, no namespace prefix>},
    }

Our v1m3-H training save (`exp/.../model/model_last.pth`) is shaped like
the Pointcept training format::

    {
        "epoch": <int>,
        "state_dict": {
            "module.student.backbone.<...>": <weight>,
            "module.teacher.backbone.<...>": <weight>,     # EMA copy, drop
            "module.patch_proj.<...>": <weight>,            # alignment head, drop
            "module.qwen_proj.<...>": <weight>,             # alignment head, drop
            "module.enc2d_model.<...>": <weight>,           # frozen Qwen ViT, drop
            ...
        },
        "optimizer": <...>, "scheduler": <...>, ...
    }

This script:
  1) Extracts only the `module.student.backbone.*` slice (the actual
     PT-v3m3 encoder weights — same family as utonia.pth).
  2) Strips the `module.student.backbone.` prefix so keys land at the
     `PointTransformerV3.<...>` root.
  3) Writes a `config` dict mirroring the H training config's backbone
     block (PT-v3m3 hyperparams) so `utonia.load` can rebuild the
     module without referencing the Pointcept config.
  4) Saves into the {config, state_dict} format.

Usage:
    python convert_h_ckpt_for_utonia_inference.py \
        --in  exp/utonia_q35_h/model/model_last.pth \
        --out exp/utonia_q35_h/inference_ckpt.pth

Then in Video-3D-LLM (or any consumer of utonia.load):
    model = utonia.load("exp/utonia_q35_h/inference_ckpt.pth")
"""

import argparse
import torch


# These mirror the H config's `backbone_s` block (and v1m1 paper config) —
# the inference-time PointTransformerV3 must be built with these or the
# loaded weights won't fit.  Keep aligned with
# training/configs/utonia/distill-utonia-v1m3-H-scannet-only-qwen3_5-4b.py
H_PT_V3_CONFIG = dict(
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
    # drop_path: 0.3 was training-time stochastic depth.  Inference uses 0
    # so the forward graph is deterministic.
    drop_path=0.0,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    enc_mode=True,
    traceable=False,
    mask_token=False,        # mask_token used in pretraining only — drop at inference
    rope_base=10,
    shift_coords=None,
    jitter_coords=None,
    rescale_coords=None,
    # Decoder is unused under enc_mode=True but utonia.load's
    # PointTransformerV3 still validates dec_* lengths against num_stages-1.
    # Fill with defaults so the assert passes; they'll never run.
    dec_depths=(3, 3, 3, 3),
    dec_channels=(96, 96, 192, 384),
    dec_num_head=(6, 6, 12, 32),
    dec_patch_size=(1024, 1024, 1024, 1024),
)

STUDENT_PREFIX = "module.student.backbone."


def extract_backbone(state_dict):
    """Return {<raw_PT_v3_key>: tensor} for student.backbone.* entries.
    Drops teacher.backbone, patch_proj, qwen_proj, enc2d_model, heads, etc.
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith(STUDENT_PREFIX):
            out[k[len(STUDENT_PREFIX):]] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="inp",  required=True,
                   help="Pointcept-format training save (e.g. model_last.pth)")
    p.add_argument("--out", required=True,
                   help="Output path for the utonia.load-compatible ckpt")
    p.add_argument("--mask-token", action="store_true",
                   help="Keep mask_token=True in the config. Default is False; "
                        "the pretraining mask token is unused at inference "
                        "and leaving it on can mismatch state_dict expectations "
                        "depending on how PointTransformerV3 was built.")
    args = p.parse_args()

    print(f"[load] {args.inp}")
    src = torch.load(args.inp, map_location="cpu", weights_only=False)
    if isinstance(src, dict) and "state_dict" in src:
        raw = src["state_dict"]
    else:
        raw = src

    sd = extract_backbone(raw)
    if not sd:
        # Some saves use `student.backbone.` without the `module.` DDP prefix.
        # Try that variant.
        alt = "student.backbone."
        for k, v in raw.items():
            if k.startswith(alt):
                sd[k[len(alt):]] = v
        assert sd, (
            f"no keys starting with {STUDENT_PREFIX!r} or {alt!r} in checkpoint; "
            f"first 3 keys = {list(raw.keys())[:3]}"
        )

    print(f"[extract] {len(sd)} student.backbone.* keys")
    sample = next(iter(sd.keys()))
    print(f"          sample key after strip: {sample!r}")

    cfg = dict(H_PT_V3_CONFIG)
    cfg["mask_token"] = args.mask_token

    out_ckpt = {"config": cfg, "state_dict": sd}
    torch.save(out_ckpt, args.out)
    print(f"[save] {args.out}  ({len(sd)} weights, "
          f"{sum(t.numel() for t in sd.values()) / 1e6:.1f}M params)")
    print("\nUse with:\n"
          f"    import utonia\n"
          f"    model = utonia.load({args.out!r})\n")


if __name__ == "__main__":
    main()
