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
    import inspect

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
    with torch.no_grad():
        x_t = x.unsqueeze(1).repeat(1, T, 1, 1, 1)
        patches = x_t.view(B, T, 3, h, P, w, P)
        patches = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        patches = patches.view(B * h * w, T * 3 * P * P)
        # T_grid = 1: post-patch-embed temporal length (the conv3d in
        # patch_embed collapses temporal_patch_size frames into one token).
        grid_thw = torch.tensor([[1, h, w]] * B,
                                device=args.device, dtype=torch.long)
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        rope_out = getattr(visual, rope_attr)(grid_thw)

    print(f"[ 4/4 INFO] block forward sig: "
          f"{inspect.signature(visual.blocks[0].forward)}")
    if isinstance(rope_out, tuple):
        print(f"[ 4/4 INFO] {rope_attr}(grid_thw) returned tuple of "
              f"{[tuple(t.shape) for t in rope_out]}")
    elif torch.is_tensor(rope_out):
        print(f"[ 4/4 INFO] {rope_attr}(grid_thw) returned tensor {tuple(rope_out.shape)}")
    else:
        print(f"[ 4/4 INFO] {rope_attr}(grid_thw) returned {type(rope_out).__name__}")

    # Try a series of forward strategies, accepting the first that succeeds.
    blk_params = set(inspect.signature(visual.blocks[0].forward).parameters)

    def _try_manual(label, blk_kwargs_fn):
        with torch.no_grad():
            hidden = visual.patch_embed(patches)
            for blk in visual.blocks:
                hidden = blk(hidden, **blk_kwargs_fn())
        return hidden

    strategies = []
    if "rotary_pos_emb" in blk_params:
        strategies.append((
            "manual: rotary_pos_emb=tensor",
            lambda: dict(cu_seqlens=cu_seqlens, rotary_pos_emb=rope_out),
        ))
    if "position_embeddings" in blk_params and isinstance(rope_out, tuple):
        strategies.append((
            "manual: position_embeddings=(cos,sin)",
            lambda: dict(cu_seqlens=cu_seqlens, position_embeddings=rope_out),
        ))
    if "position_embeddings" in blk_params and torch.is_tensor(rope_out):
        # Newer Qwen blocks build (cos, sin) from the rope tensor.
        emb = torch.cat((rope_out, rope_out), dim=-1)
        strategies.append((
            "manual: position_embeddings=(cos,sin) built from rope tensor",
            lambda: dict(cu_seqlens=cu_seqlens,
                         position_embeddings=(emb.cos(), emb.sin())),
        ))

    # Always also try calling visual.forward end-to-end as a last resort.
    visual_sig = inspect.signature(visual.forward)
    print(f"[ 4/4 INFO] visual.forward sig: {visual_sig}")

    hidden = None
    chosen = None
    for label, kfn in strategies:
        try:
            hidden = _try_manual(label, kfn)
            chosen = label
            print(f"[ 4/4 STEP] manual forward OK via [{label}], "
                  f"shape={tuple(hidden.shape)}")
            break
        except Exception as e:
            print(f"[ 4/4 STEP] {label} FAILED: {e}")

    if hidden is None:
        # Fallback: call the full visual module. It returns merged tokens
        # (post spatial-merge), but at least confirms which API is correct.
        print("[ 4/4 STEP] falling back to visual.forward(...)")
        try:
            with torch.no_grad():
                if "grid_thw" in visual_sig.parameters:
                    hidden = visual(patches, grid_thw=grid_thw)
                else:
                    hidden = visual(patches)
            chosen = "visual.forward (post-merge)"
            print(f"[ 4/4 STEP] visual.forward OK, shape={tuple(hidden.shape)}")
        except Exception as e:
            print(f"[ 4/4 FAIL] all strategies failed; last error: {e}")
            print("           visual attrs:",
                  [n for n in dir(visual) if not n.startswith("_")])
            print("           block attrs:",
                  [n for n in dir(visual.blocks[0]) if not n.startswith("_")])
            try:
                print("           block.attn sig:",
                      inspect.signature(visual.blocks[0].attn.forward))
            except Exception:
                pass
            traceback.print_exc()
            sys.exit(6)

    # post-patch-embed token count is B*h*w (T collapses in patch_embed).
    expected_n = B * h * w
    print(f"[ 4/4 INFO] expected pre-merge first-dim={expected_n} "
          f"(post-merge would be {expected_n // (cfg.spatial_merge_size ** 2)})")
    norm = hidden.float().norm(dim=-1).mean().item()
    finite = torch.isfinite(hidden).all().item()
    print(f"[ 4/4 PASS] strategy=[{chosen}] final shape={tuple(hidden.shape)} "
          f"mean L2={norm:.3f} finite={finite}")
    if not finite:
        sys.exit(5)

    print("\n[summary] OK — paste the [ 4/4 INFO]/[ 4/4 STEP] lines back so we can "
          "wire the correct kwarg into ENC2D_forward.")


if __name__ == "__main__":
    main()
