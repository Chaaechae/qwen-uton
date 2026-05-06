"""
Standalone smoke test for the Qwen3.5 vision tower forward path used in
`utonia_v1m2_qwen3_5_distill.py::ENC2D_forward`.

Checks (in order, each is independent):
  1. The full VLM loads, has `model.visual` attribute.
  2. `visual` exposes the sub-modules our code calls:
       - patch_embed
       - rot_pos_emb (or rotary_pos_emb)
       - blocks
       - config (with patch_size, temporal_patch_size, hidden_size, spatial_merge_size)
  3. End-to-end forward of our patchify + flat-token path on a fake (B=2, 3, 512, 512)
     image batch produces (B, h*w, 1024) features without error.
  4. Output L2 norm and dim are sane (non-zero, finite, last-dim==hidden_size==1024).

Run (HF Hub repo id — uses HF cache, downloads if missing):
    python3 tools/test_qwen3_5_vit_path.py --model Qwen/Qwen3.5-4B

Run (local path — never touches the network):
    python3 tools/test_qwen3_5_vit_path.py --model /path/to/Qwen3.5-4B
    # If --model points to an existing directory or file, --local_files_only
    # is implied automatically. Pass --no-local_files_only to override.

Run (HF cache only, skip network even with a repo id):
    python3 tools/test_qwen3_5_vit_path.py --model Qwen/Qwen3.5-4B --local_files_only
    # Equivalent to setting HF_HUB_OFFLINE=1 in the environment.

Custom HF cache root (alternative to HF_HOME env var):
    python3 tools/test_qwen3_5_vit_path.py --cache_dir /scratch/hf_cache --model Qwen/Qwen3.5-4B

It will print PASS/FAIL for each check and a short diagnostic dump of
`dir(visual)` if any attribute is missing, so you can paste the output back.
"""

import argparse
import os
import sys
import traceback

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-4B",
        help="HF Hub repo id (e.g. Qwen/Qwen3.5-4B) OR an absolute/relative "
             "filesystem path to a directory containing config.json and the "
             "model weights. If a path, --local_files_only is implied.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="HF cache root (forwarded to from_pretrained). Defaults to "
             "$HF_HOME or ~/.cache/huggingface when omitted.",
    )
    parser.add_argument(
        "--local_files_only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force transformers to use only on-disk files (no network). "
             "Auto-enabled when --model is a local path; pass "
             "--no-local_files_only to override.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    torch_dtype = dict(
        float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16
    )[args.dtype]

    # Resolve --model: if it's an existing path, expand it and force offline mode.
    model_arg = os.path.expanduser(args.model)
    is_local_path = os.path.isdir(model_arg) or os.path.isfile(model_arg)
    if is_local_path:
        model_arg = os.path.abspath(model_arg)
        if args.local_files_only is None:
            args.local_files_only = True
    elif args.local_files_only is None:
        args.local_files_only = False

    print(f"[info] model={model_arg} (local_path={is_local_path}) "
          f"device={args.device} dtype={args.dtype} "
          f"local_files_only={args.local_files_only} cache_dir={args.cache_dir}")

    # ---------------- step 1: load full VLM ----------------
    try:
        from transformers import AutoModelForImageTextToText
        from_pretrained_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            local_files_only=args.local_files_only,
        )
        if args.cache_dir is not None:
            from_pretrained_kwargs["cache_dir"] = os.path.expanduser(args.cache_dir)
        full = AutoModelForImageTextToText.from_pretrained(
            model_arg, **from_pretrained_kwargs
        )
        print("[ 1/4 PASS] full VLM loaded")
    except Exception as e:
        print(f"[ 1/4 FAIL] could not load full VLM: {e}")
        if is_local_path:
            print(f"           checked path: {model_arg}")
            print(f"           dir contents: {sorted(os.listdir(model_arg))[:20]}"
                  if os.path.isdir(model_arg) else "           (path is a file, not a directory)")
        traceback.print_exc()
        sys.exit(1)

    # ---------------- step 2: locate visual ----------------
    visual = None
    visual_path = None
    for path in ("model.visual", "visual", "model.vision_tower", "vision_tower"):
        cur = full
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            visual = cur
            visual_path = path
            break
    if visual is None:
        print(f"[ 2/4 FAIL] none of (model.visual, visual, model.vision_tower) found")
        print("           top-level attrs:", [n for n in dir(full) if not n.startswith("_")][:30])
        if hasattr(full, "model"):
            print("           full.model attrs:",
                  [n for n in dir(full.model) if not n.startswith("_")][:30])
        sys.exit(2)
    print(f"[ 2/4 PASS] vision tower found at full.{visual_path}")

    # ---------------- step 3: required sub-modules ----------------
    needed = ["patch_embed", "blocks", "config"]
    rope_candidates = ["rot_pos_emb", "rotary_pos_emb"]
    missing = [n for n in needed if not hasattr(visual, n)]
    rope_attr = next((n for n in rope_candidates if hasattr(visual, n)), None)

    if missing or rope_attr is None:
        print(f"[ 3/4 FAIL] missing attrs={missing} rope_attr_found={rope_attr}")
        print("           visual attrs:", [n for n in dir(visual) if not n.startswith("_")])
        sys.exit(3)
    print(f"[ 3/4 PASS] visual.patch_embed / visual.blocks / visual.config / visual.{rope_attr}")

    cfg = visual.config
    print(f"           cfg: hidden_size={cfg.hidden_size} patch_size={cfg.patch_size} "
          f"temporal_patch_size={cfg.temporal_patch_size} "
          f"spatial_merge_size={cfg.spatial_merge_size} depth={cfg.depth}")
    if cfg.hidden_size != 1024 or cfg.patch_size != 16:
        print(f"[warn] expected hidden_size=1024 patch_size=16; got "
              f"{cfg.hidden_size}/{cfg.patch_size}")

    # ---------------- step 4: forward path ----------------
    visual = visual.eval().to(args.device)
    for p in visual.parameters():
        p.requires_grad_(False)

    H = W = args.crop
    P = cfg.patch_size
    T = cfg.temporal_patch_size
    assert H % P == 0 and W % P == 0
    h, w = H // P, W // P
    B = args.batch

    # Mimic preprocessor: mean=std=0.5, so feed in [-1, 1].
    x = (torch.rand(B, 3, H, W, device=args.device) - 0.5) * 2.0
    x = x.to(torch_dtype)

    # Patchify exactly as in ENC2D_forward.
    try:
        with torch.no_grad():
            x_t = x.unsqueeze(1).repeat(1, T, 1, 1, 1)              # (B, T, 3, H, W)
            patches = x_t.view(B, T, 3, h, P, w, P)
            patches = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
            patches = patches.view(B * h * w, T * 3 * P * P)         # (B*h*w, T*3*P*P)
            grid_thw = torch.tensor([[T, h, w]] * B,
                                    device=args.device, dtype=torch.long)

            hidden = visual.patch_embed(patches)
            rotary_pos_emb = getattr(visual, rope_attr)(grid_thw)
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
            for blk in visual.blocks:
                hidden = blk(hidden, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        print(f"[ 4/4 STEP] post-blocks hidden shape = {tuple(hidden.shape)}")
        # Expect (B*T*h*w, hidden_size) before our reshape.
        expected_n = B * T * h * w
        if hidden.shape[0] != expected_n or hidden.shape[-1] != cfg.hidden_size:
            print(f"[ 4/4 FAIL] expected first-dim={expected_n} last-dim={cfg.hidden_size}, "
                  f"got {tuple(hidden.shape)}")
            sys.exit(4)
        hidden = hidden.view(B, T, h * w, -1)[:, 0]                  # (B, h*w, hidden)
        norm = hidden.float().norm(dim=-1).mean().item()
        finite = torch.isfinite(hidden).all().item()
        print(f"[ 4/4 PASS] final shape={tuple(hidden.shape)} mean L2={norm:.3f} finite={finite}")
        if not finite:
            sys.exit(5)
    except Exception as e:
        print(f"[ 4/4 FAIL] forward failed: {e}")
        print("           visual attrs:", [n for n in dir(visual) if not n.startswith("_")])
        traceback.print_exc()
        sys.exit(6)

    print("\n[summary] OK — distillation forward path is compatible with this model "
          f"(use rope_attr='{rope_attr}' in ENC2D_forward).")


if __name__ == "__main__":
    main()
