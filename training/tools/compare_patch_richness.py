"""
Per-patch representation richness comparison across vision models.

Why this exists
---------------
Original Utonia framework aligned 3D point cloud features to DINOv2
patches with a simple cosine pull and it worked.  We tried the same
recipe with Qwen3.5 ViT and 5 variants all hit the same chance-level
ceiling.  This script *measures* per-patch information richness so
we can see *why*: DINOv2 was trained with patch-level objectives
(iBOT + multi-crop DINO) so each patch is a self-contained semantic
unit; Qwen-style VL ViTs are trained for LLM consumption where the
patch features are intermediate inputs to a 2×2 merger and the
"real" output lives at the merged 16×16 scale.

Metrics
-------
For each model, runs forward on the same set of test images,
collects per-patch hidden states (post final norm if any), and
reports:

  - eff_rank  : participation ratio (Σσ²)² / Σσ⁴.  Smooth measure
                of effective dimensionality.
  - d99       : # principal components needed to capture 99% of
                variance.  Counts meaningful signal directions.
  - top_ratio : top σ / mean σ.  How dominant the strongest
                direction is.  Big = anisotropic / DC-dominated.
  - pairwise_cos_mean : mean of pairwise cosines between patches.
                Low (~0) = isotropic/discriminative.  High (~0.9)
                = patches all roughly parallel = info compressed
                into a narrow cone.

Usage
-----
  python tools/compare_patch_richness.py \
      --dinov2-hub  dinov2_vitb14 \
      --qwen        /group-volume/chaewon.yun/Qwen3.5-4B \
      --clip        openai/clip-vit-large-patch14 \
      --image-dir   /group-volume/3Ddataset/data/scannet/images \
      --num-images  50 \
      --out-dir     exp/patch_richness

Any of --dinov2-hub / --dinov2-path / --qwen / --clip can be omitted
to skip that model.

For Qwen we additionally report 3 variants:
  (1) raw last-block output (the rank-1 collapse we found)
  (2) + merger.norm           (what v1m3-G uses, 32x32)
  (3) full merger output      (what v1m3-H uses, 16x16, LLM dim)
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dinov2-hub", default=None,
                   help="torch.hub model name, e.g. dinov2_vitb14.")
    p.add_argument("--dinov2-hf", default=None,
                   help="HuggingFace DINOv2 model id, e.g. "
                        "facebook/dinov2-with-registers-giant. This is "
                        "what the ORIGINAL Utonia training used; "
                        "loaded via transformers.AutoModel.")
    p.add_argument("--dinov2-path", default=None,
                   help="Local DINOv2 checkpoint (.pth). Loaded into "
                        "a hub-built model with strict=False.")
    p.add_argument("--qwen", default=None,
                   help="Path to Qwen3.5-style HF checkpoint folder.")
    p.add_argument("--clip", default=None,
                   help="HF CLIP vision model id or path, e.g. "
                        "openai/clip-vit-large-patch14.")
    p.add_argument("--image-dir", required=True,
                   help="Directory containing PNG/JPG images.")
    p.add_argument("--num-images", type=int, default=50)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def collect_images(image_dir, num_images, seed=0):
    """Find PNG/JPG files under image_dir and pick num_images of them."""
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"):
        paths.extend(glob.glob(os.path.join(image_dir, "**", ext),
                               recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"no images under {image_dir}")
    rng = np.random.RandomState(seed)
    if len(paths) > num_images:
        idx = rng.choice(len(paths), num_images, replace=False)
        paths = [paths[i] for i in idx]
    return paths


def build_transform(size, mean, std):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rank_metrics(X, name=None, max_patches=20000):
    """
    X : (N, D) feature matrix.  Centered, then SVD; report PCA-based
    richness metrics.  Subsamples N down to max_patches for tractable
    SVD on large batches.
    """
    X = X.float().detach().cpu()
    if X.shape[0] > max_patches:
        idx = torch.randperm(X.shape[0])[:max_patches]
        X = X[idx]
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    var = s ** 2
    cum = torch.cumsum(var, dim=0) / var.sum().clamp_min(1e-30)

    def dim_for(t):
        return int((cum < t).sum().item()) + 1

    eff = float((var.sum() ** 2) / (var * var).sum().clamp_min(1e-30))
    top_ratio = float(s[0] / s.mean().clamp_min(1e-30))

    # Pairwise cosine — sample 256 patches to avoid O(N²) blow-up.
    K = min(256, X.shape[0])
    sub = X[torch.randperm(X.shape[0])[:K]]
    sn = F.normalize(sub, dim=-1)
    cos = sn @ sn.T
    mask = ~torch.eye(K, dtype=torch.bool)
    pair_cos = cos[mask].mean().item()

    return dict(
        name=name,
        N=X.shape[0],
        D=X.shape[1],
        eff_rank=eff,
        d50=dim_for(0.50),
        d90=dim_for(0.90),
        d99=dim_for(0.99),
        top_ratio=top_ratio,
        pairwise_cos_mean=pair_cos,
        cum=cum.numpy(),
    )


# ---------------------------------------------------------------------------
# Model adapters — each returns a list of (variant_name, patch_feature_extractor)
# Each extractor: PIL.Image -> (n_patches, D) torch tensor
# ---------------------------------------------------------------------------
def load_dinov2(args, device):
    """
    DINOv2 loaders. Three flavors:

      --dinov2-hf  : transformers.AutoModel(...).  This is what the
                     original Utonia configs used (e.g.
                     `facebook/dinov2-with-registers-giant`).
                     Forward returns last_hidden_state with layout
                     [CLS, (4 registers), patch1...patchN]; we slice
                     the trailing patches.
      --dinov2-hub : torch.hub('facebookresearch/dinov2', NAME).
                     For ablation against the HF version.
      --dinov2-path: local checkpoint loaded into a hub-built arch
                     (override arch via env DINOV2_ARCH).

    Multiple flags can be passed simultaneously; each is reported
    as a separate row.
    """
    variants = []

    if args.dinov2_hf is not None:
        print(f"[load] DINOv2 (HF AutoModel)  id={args.dinov2_hf}")
        from transformers import AutoModel, AutoImageProcessor
        model = AutoModel.from_pretrained(
            args.dinov2_hf, trust_remote_code=True
        ).to(device).eval()
        try:
            proc = AutoImageProcessor.from_pretrained(args.dinov2_hf)
        except Exception:
            proc = None
        # ImageNet stats (DINOv2 default). DINOv2-giant typically
        # at 224x224, with 14-pixel patches → 16x16 patch grid.
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        fallback_tf = build_transform(224, IMAGENET_MEAN, IMAGENET_STD)

        @torch.inference_mode()
        def extract_hf(pil_img):
            pil_img = pil_img.convert("RGB")
            if proc is not None:
                x = proc(images=pil_img, return_tensors="pt")["pixel_values"]
                x = x.to(device)
            else:
                x = fallback_tf(pil_img).unsqueeze(0).to(device)
            out = model(pixel_values=x)
            # Patch tokens are the trailing N entries (after CLS +
            # optional registers).  Same indexing the original Utonia
            # code used: last_hidden_state[:, -H*W:, :].
            h_w_total = (x.shape[-2] // 14) * (x.shape[-1] // 14)
            patches = out.last_hidden_state[0, -h_w_total:, :]
            return patches.float().cpu()

        variants.append(
            (f"dinov2-HF {os.path.basename(args.dinov2_hf)}", extract_hf)
        )

    if args.dinov2_hub is not None or args.dinov2_path is not None:
        print(f"[load] DINOv2 (hub)  name={args.dinov2_hub}  "
              f"path={args.dinov2_path}")
        if args.dinov2_hub:
            model = torch.hub.load(
                "facebookresearch/dinov2", args.dinov2_hub, pretrained=True
            )
            tag = args.dinov2_hub
        else:
            arch = os.environ.get("DINOV2_ARCH", "dinov2_vitb14")
            model = torch.hub.load(
                "facebookresearch/dinov2", arch, pretrained=False
            )
            sd = torch.load(args.dinov2_path, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            model.load_state_dict(sd, strict=False)
            tag = arch + "(local)"
        model = model.to(device).eval()

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        tf = build_transform(518, IMAGENET_MEAN, IMAGENET_STD)

        @torch.inference_mode()
        def extract_hub(pil_img):
            x = tf(pil_img.convert("RGB")).unsqueeze(0).to(device)
            feat = model.forward_features(x)
            if isinstance(feat, dict) and "x_norm_patchtokens" in feat:
                patches = feat["x_norm_patchtokens"][0]
            else:
                patches = feat[0, 1:, :]
            return patches.float().cpu()

        variants.append((f"dinov2-hub {tag}", extract_hub))

    return variants


def load_qwen(args, device):
    """
    Qwen3.5 visual: report 3 variants.
      (1) raw last-block hidden     — what was effectively rank ~1
      (2) post merger.norm          — what v1m3-G uses
      (3) full merger output 16x16  — what v1m3-H uses
    """
    if args.qwen is None:
        return []
    print(f"[load] Qwen3.5 visual  path={args.qwen}")
    from transformers import AutoModelForImageTextToText
    full = AutoModelForImageTextToText.from_pretrained(
        args.qwen, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    visual = full.model.visual.eval().to(device)
    del full

    QWEN_MEAN = (0.5, 0.5, 0.5)
    QWEN_STD = (0.5, 0.5, 0.5)
    tf = build_transform(512, QWEN_MEAN, QWEN_STD)
    P = visual.config.patch_size
    T = visual.config.temporal_patch_size

    @torch.inference_mode()
    def forward_blocks(pil_img):
        """Returns (post-blocks hidden, h, w) for one image."""
        x = tf(pil_img.convert("RGB")).unsqueeze(0).to(device, dtype=torch.bfloat16)
        B, C, H, W = x.shape
        h, w = H // P, W // P
        x_t = x.unsqueeze(1).repeat(1, T, 1, 1, 1)
        patches = x_t.view(B, T, C, h, P, w, P)
        patches = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        patches = patches.view(B * h * w, T * C * P * P)
        grid_thw = torch.tensor([[1, h, w]] * B, device=device, dtype=torch.long)
        hidden = visual.patch_embed(patches)
        rotary_pos_emb = visual.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for blk in visual.blocks:
            hidden = blk(
                hidden,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
        return hidden, h, w

    def variant_raw(pil_img):
        hidden, _, _ = forward_blocks(pil_img)
        return hidden.float().cpu()

    def variant_norm(pil_img):
        hidden, _, _ = forward_blocks(pil_img)
        normed = visual.merger.norm(hidden)
        return normed.float().cpu()

    def variant_full_merger(pil_img):
        hidden, h, w = forward_blocks(pil_img)
        D = hidden.shape[-1]
        B = hidden.shape[0] // (h * w)
        h_blk = (
            hidden.view(B, h, w, D)
                  .view(B, h // 2, 2, w // 2, 2, D)
                  .permute(0, 1, 3, 2, 4, 5).contiguous()
                  .view(B * (h // 2) * (w // 2) * 4, D)
        )
        merged = visual.merger(h_blk)  # (B*16*16, LLM_dim)
        return merged.float().cpu()

    return [
        ("qwen raw (post-blocks, no norm)", variant_raw),
        ("qwen + merger.norm  (32x32, v1m3-G)", variant_norm),
        ("qwen full merger    (16x16, v1m3-H)", variant_full_merger),
    ]


def load_clip(args, device):
    """CLIP vision model patch tokens (post-final-layernorm)."""
    if args.clip is None:
        return []
    print(f"[load] CLIP  path={args.clip}")
    from transformers import CLIPVisionModel, CLIPImageProcessor
    model = CLIPVisionModel.from_pretrained(args.clip).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(args.clip)

    @torch.inference_mode()
    def extract(pil_img):
        inputs = proc(images=pil_img.convert("RGB"), return_tensors="pt")
        x = inputs["pixel_values"].to(device)
        out = model(pixel_values=x)
        # last_hidden_state is post the final layernorm.
        # First token is CLS, strip it.
        patches = out.last_hidden_state[0, 1:, :]
        return patches.float().cpu()

    return [("clip (post-norm patches)", extract)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "compare.log")
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"[setup] image-dir={args.image_dir}  num-images={args.num_images}")
    paths = collect_images(args.image_dir, args.num_images, seed=args.seed)
    log(f"[setup] sampled {len(paths)} images")

    variants = []
    variants.extend(load_dinov2(args, args.device))
    variants.extend(load_qwen(args, args.device))
    variants.extend(load_clip(args, args.device))
    if not variants:
        raise SystemExit(
            "no model selected; pass --dinov2-hub / --dinov2-path / "
            "--qwen / --clip"
        )

    # For each variant, run all images and accumulate patches.
    all_results = []
    for name, extractor in variants:
        log(f"\n[{name}] extracting patches ...")
        patches_all = []
        for i, p in enumerate(paths):
            try:
                img = Image.open(p)
                f = extractor(img)
                patches_all.append(f)
            except Exception as e:
                log(f"  ! image {os.path.basename(p)} failed: "
                    f"{type(e).__name__}: {str(e)[:80]}")
                continue
            if (i + 1) % 10 == 0:
                log(f"  {i + 1}/{len(paths)} done")
        if not patches_all:
            log(f"  no successful images; skipping {name}")
            continue
        X = torch.cat(patches_all, 0)
        m = rank_metrics(X, name=name)
        all_results.append(m)
        log(f"  N={m['N']}  D={m['D']}  "
            f"eff_rank={m['eff_rank']:7.2f}  "
            f"d50/d90/d99={m['d50']:3d}/{m['d90']:4d}/{m['d99']:4d}  "
            f"top_ratio={m['top_ratio']:7.2f}  "
            f"pairwise_cos={m['pairwise_cos_mean']:+.3f}")

    # ----- Summary table -----
    log(f"\n[summary] per-patch richness")
    log(f"{'model variant':<42s} {'D':>5s} {'eff_rank':>10s} "
        f"{'d50':>5s} {'d90':>5s} {'d99':>5s} "
        f"{'top_ratio':>10s} {'pair_cos':>10s}")
    log("-" * 105)
    for m in all_results:
        log(f"{m['name']:<42s} {m['D']:>5d} {m['eff_rank']:>10.2f} "
            f"{m['d50']:>5d} {m['d90']:>5d} {m['d99']:>5d} "
            f"{m['top_ratio']:>10.2f} {m['pairwise_cos_mean']:>+10.3f}")

    # ----- Cumulative variance plot -----
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in all_results:
        ax.plot(np.arange(1, len(m["cum"]) + 1), m["cum"],
                label=m["name"], linewidth=1.5)
    ax.set_xscale("log")
    ax.axhline(0.5, ls="--", color="gray", alpha=0.3)
    ax.axhline(0.9, ls="--", color="gray", alpha=0.3)
    ax.axhline(0.99, ls=":", color="gray", alpha=0.3)
    ax.set_xlabel("# principal components (log)")
    ax.set_ylabel("cumulative variance ratio")
    ax.set_title("Per-patch representation richness")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_plot = os.path.join(args.out_dir, "cumulative_variance.png")
    plt.savefig(out_plot, dpi=120)
    plt.close()
    log(f"\n[plot] -> {out_plot}")

    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"[log]  -> {log_path}")


if __name__ == "__main__":
    main()
