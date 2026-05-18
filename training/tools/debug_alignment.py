"""
Single-batch alignment debugger.

5 variants (v1m3-B...F) all hit the same ceiling
(pos_bc ~ 0.01, CKA_proj ~ 0.005). Before designing v1m3-G, this
script answers the prerequisite question: is the pipeline doing
what we think it's doing?

Reports for a single batch loaded from the train split:

  (1) K survival — total patches in batch vs valid pairs vs unique
      patches after scatter_mean. Tells us whether correspondence
      mapping is even reaching most patches.

  (2) Feature stats — norm distribution + pairwise cosine spread
      for feature2d (Qwen ViT), feature3d_raw (PTv3 output), and
      feature3d_proj (patch_proj output).  Tells us if either side
      is degenerate / over-anisotropic.

  (3) Gradient flow — for one forward+backward on this batch with
      the chosen enc2d loss, print  ||patch_proj.grad|| /
      ||patch_proj.weight||  ratio for every Linear inside
      patch_proj.  Vanishing → no learning is mechanically possible.

  (4) Single-batch overfit — freeze the world (backbone + Qwen),
      cache (f3_raw, f2) ONCE, then run N steps of an Adam optimizer
      on patch_proj alone using the configured enc2d loss.  This is
      the strictest possible test of "can patch_proj learn this
      mapping at all on a fixed batch?".  If even this can't drop
      the loss, the loss formulation or the architecture is the
      problem.  If it can, the failure lives somewhere else in
      training (data variance, SSL interference, EMA, etc.).

  (5) Correspondence visualization — pick the first image in the
      batch, render it side-by-side with a patch-grid overlay
      colored by whether each patch has a 3D point mapping
      (green = mapped, gray = no mapping).  Sanity check that the
      mapping isn't pathologically sparse or concentrated.

Usage:

    cd third_party/Pointcept
    export PYTHONPATH=./
    python tools/debug_alignment.py \\
        --config-file configs/utonia/distill-utonia-v1m3-F-scannet-only-qwen3_5-4b.py \\
        --weight     exp/utonia_q35_f/model/model_last.pth \\
        --out-dir    exp/utonia_q35_f/debug \\
        --overfit-steps 100
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pointcept.engines.defaults import default_config_parser
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.models.utils import offset2batch, bincount2offset
from pointcept.models.utils.structure import Point

# Re-use helpers from the eval script.
from tools.eval_alignment_full import (
    _coerce_sample_for_model,
    _load_into_model,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weight", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--scene-idx", type=int, default=0,
                   help="Which scene from the dataset to debug.")
    p.add_argument("--overfit-steps", type=int, default=100)
    p.add_argument("--overfit-lr", type=float, default=4e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--probe-all-layers", action="store_true",
        help="Run a single forward through Qwen ViT and report PCA "
             "effective rank at every block. Use this to pick "
             "`enc2d_layer_idx` when the last block has collapsed.",
    )
    return p.parse_args()


@torch.inference_mode()
def _probe_all_layers(model, batch, device, log):
    """
    Run Qwen ViT once over the batch's images, capturing the hidden
    state after every block. For each block, compute PCA effective
    rank + dims-for-99% on the (B*h*w, D) features.

    Deep ViTs often show "token uniformity collapse" — features pile
    up onto a single direction in the last few layers. This probe
    pinpoints how shallow we need to go to recover usable patch
    discriminability.

    Also tries to apply any final norm / merger that the visual model
    exposes; in standard pre-norm ViT the residual stream is highly
    DC-dominated and the final layernorm is what produces usable
    per-patch features.  If the user reports "all blocks rank < 5",
    that's almost always because this post-norm step is missing.
    """
    imgs = batch["images"]
    if imgs.shape[0] == 0:
        log("[probe] no images in batch — skipping")
        return

    enc2d = model.enc2d_model

    # Quick architectural sanity check: list the top-level submodules
    # so we can see what post-block norm / merger is available.
    log(f"\n[probe] enc2d_model top-level modules:")
    for n, m in enc2d.named_children():
        sub = list(m.named_children())
        if sub:
            log(f"  {n}: {type(m).__name__}  children={[c[0] for c in sub]}")
        else:
            log(f"  {n}: {type(m).__name__}")

    B, C, H_pix, W_pix = imgs.shape
    T = enc2d.config.temporal_patch_size
    P = enc2d.config.patch_size
    h, w = H_pix // P, W_pix // P

    x_t = imgs.unsqueeze(1).repeat(1, T, 1, 1, 1)
    patches = x_t.view(B, T, C, h, P, w, P)
    patches = patches.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
    patches = patches.view(B * h * w, T * C * P * P)

    grid_thw = torch.tensor(
        [[1, h, w]] * B, device=device, dtype=torch.long
    )
    hidden = enc2d.patch_embed(patches)
    rotary_pos_emb = enc2d.rot_pos_emb(grid_thw)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    def _rank_of(name, X):
        Xf = X.view(-1, X.shape[-1]).float().detach().cpu()
        Xc = Xf - Xf.mean(0, keepdim=True)
        s = torch.linalg.svdvals(Xc)
        var = s ** 2
        cum = torch.cumsum(var, dim=0) / var.sum().clamp_min(1e-30)
        d99 = int((cum < 0.99).sum().item()) + 1
        eff = float((var.sum() ** 2) / (var * var).sum().clamp_min(1e-30))
        norm_mean = Xf.norm(dim=-1).mean().item()
        log(f"  {name:>32s}  D={X.shape[-1]:4d}  eff_rank={eff:7.2f}  "
            f"d99={d99:4d}  ||x||_mean={norm_mean:8.3f}")
        return eff, d99

    log(f"\n[probe] {len(enc2d.blocks)} ViT blocks — measuring PCA rank "
        f"per block over (N={B*h*w}, D)")
    rows = []
    eff, d99 = _rank_of("patch_embed", hidden)
    rows.append(("patch_embed", eff, d99))

    for i, blk in enumerate(enc2d.blocks):
        hidden = blk(
            hidden,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        if i in {0, 1, 2, len(enc2d.blocks) // 4,
                 len(enc2d.blocks) // 2,
                 3 * len(enc2d.blocks) // 4,
                 len(enc2d.blocks) - 3,
                 len(enc2d.blocks) - 2,
                 len(enc2d.blocks) - 1}:
            eff, d99 = _rank_of(f"block[{i}]", hidden)
            rows.append((f"block[{i}]", eff, d99))

    # Try common post-block norm / merger paths to see if the missing
    # piece restores rank. We expand into the merger's children too:
    # for Qwen3.5 the merger has `norm` + `mlp` (and possibly more),
    # so trying merger.norm AND each sub-module individually shows
    # which step recovers usable rank.
    base_paths = [
        "merger.ln_q",
        "merger.norm",
        "merger.norm1",
        "merger.norm2",
        "norm",
        "final_layernorm",
        "post_layernorm",
        "ln_post",
    ]
    # Auto-discover every direct child of `merger` if present.
    if hasattr(enc2d, "merger"):
        for child_name, _ in enc2d.merger.named_children():
            p = f"merger.{child_name}"
            if p not in base_paths:
                base_paths.append(p)
    log(f"\n[probe] trying candidate post-block normalizations / mergers:")
    found_any = False
    for path in base_paths:
        obj = enc2d
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False; break
            obj = getattr(obj, part)
        if not ok:
            log(f"  - {path:>32s}  (not present)")
            continue
        try:
            normed = obj(hidden)
        except Exception as e:
            log(f"  ! {path:>32s}  call failed: "
                f"{type(e).__name__}: {str(e)[:80]}")
            continue
        # Only meaningful if the output is still per-patch (same N).
        if normed.shape[0] != hidden.shape[0]:
            log(f"  ! {path:>32s}  output N changed "
                f"({hidden.shape[0]} -> {normed.shape[0]}) — skipping rank "
                f"(spatial merger reduces resolution)")
            continue
        eff, d99 = _rank_of(f"after {path}", normed)
        rows.append((f"after {path}", eff, d99))
        found_any = True
    # Also try a 2-step pipeline: norm then any non-pooling Linear.
    if hasattr(enc2d, "merger") and hasattr(enc2d.merger, "norm"):
        try:
            h_normed = enc2d.merger.norm(hidden)
        except Exception:
            h_normed = None
        if h_normed is not None and hasattr(enc2d.merger, "mlp"):
            mlp = enc2d.merger.mlp
            # Try mlp[0] only (first Linear, no spatial collapse).
            inner = None
            try:
                if isinstance(mlp, nn.Sequential) and len(mlp) > 0:
                    inner = mlp[0](h_normed)
                else:
                    inner = mlp(h_normed)
            except Exception as e:
                log(f"  ! merger.norm -> mlp[0] failed: "
                    f"{type(e).__name__}: {str(e)[:80]}")
            if inner is not None and inner.shape[0] == hidden.shape[0]:
                eff, d99 = _rank_of("merger.norm -> mlp[0]", inner)
                rows.append(("merger.norm->mlp[0]", eff, d99))

    # Full merger with proper 2×2 block-major reordering.
    # Our ENC2D_forward flattens patches in row-major order
    # (row 0 cols 0..w, row 1 cols 0..w, ...), but Qwen merger's
    # internal `view(-1, 4*D)` expects 2×2 spatial blocks to be
    # contiguous in the flat dim. So we permute first, then call
    # merger to get the LLM-aligned 16×16 tokens (rank should be
    # much higher than the per-patch merger.norm output if Qwen's
    # discriminative information lives at the merged scale).
    if hasattr(enc2d, "merger"):
        try:
            B_eff = hidden.shape[0] // (h * w)
            D = hidden.shape[-1]
            assert h % 2 == 0 and w % 2 == 0, \
                f"merger expects even h,w, got {h}x{w}"
            # (N=B*h*w, D) -> (B, h/2, 2, w/2, 2, D) -> block-major
            h_blk = (
                hidden.view(B_eff, h, w, D)
                      .view(B_eff, h // 2, 2, w // 2, 2, D)
                      .permute(0, 1, 3, 2, 4, 5)  # (B, h/2, w/2, 2, 2, D)
                      .contiguous()
                      .view(B_eff * (h // 2) * (w // 2) * 4, D)
            )
            merged = enc2d.merger(h_blk)
            log(f"\n[probe] full merger (with 2x2 block-major reorder):")
            log(f"  input  : (N={hidden.shape[0]}, D={D})  row-major 32×32")
            log(f"  reorder: (N={h_blk.shape[0]}, D={D})  block-major")
            log(f"  output : (N={merged.shape[0]}, D={merged.shape[-1]}) "
                f"= 16×16 tokens in LLM dim")
            eff, d99 = _rank_of("full merger (16×16)", merged)
            rows.append(("full_merger_16x16", eff, d99))
        except Exception as e:
            log(f"\n[probe] full merger attempt failed: "
                f"{type(e).__name__}: {str(e)[:120]}")

    # Last-ditch: apply a freshly-init LayerNorm to see if even a
    # generic post-norm would have helped.
    fresh_ln = nn.LayerNorm(hidden.shape[-1], elementwise_affine=False).to(
        device, dtype=hidden.dtype
    )
    eff, d99 = _rank_of("fresh LayerNorm (no affine)", fresh_ln(hidden))
    rows.append(("fresh_ln", eff, d99))

    # Plot effective rank curve.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    xs = list(range(len(rows)))
    labels = [r[0] for r in rows]
    ax[0].plot(xs, [r[1] for r in rows], marker="o")
    ax[0].set_xticks(xs); ax[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax[0].set_ylabel("effective rank (participation ratio)")
    ax[0].set_title("Where does Qwen ViT collapse — and what un-collapses it?")
    ax[0].grid(alpha=0.3)
    ax[1].plot(xs, [r[2] for r in rows], marker="o", color="tab:orange")
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax[1].set_ylabel("dims for 99% variance")
    ax[1].set_title("Cumulative-variance threshold")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(model._probe_out_dir, "qwen_layer_rank.png")
    plt.savefig(out, dpi=120)
    plt.close()
    log(f"  plot -> {out}")
    if not found_any:
        log("  [warn] no known post-block norm / merger module found; "
            "Qwen3.5 visual may expose it under a different path.")


def _build_indices(model, batch, device):
    """
    Run student.backbone + correspondence pooling and return all the
    indexing tensors so we can both extract pairs and visualize the
    same data the loss sees.

    Returns dict with:
        feature3d_raw         (K_unique, 1332) — PTv3 output, scatter-meaned
                              to per-patch, indexed at all flat patch ids.
        feature2d             (total_patches, 1024) — Qwen ViT patch feats.
        feature_index_unique  (K_unique,) flat patch ids that received a pair.
        feature_index_all     (N_pairs,) raw flat patch id per (point, image)
                              pair (before scatter_mean unique).
        valid_index           per the model's notation.
        offset_img_num        cumulative image count per batch element.
        imgs                  (total_images, 3, H, W) the raw input images.
        patch_h, patch_w
    """
    global_point = Point(
        feat=batch.get("global_feat_full", batch["global_feat"]),
        coord=batch["global_coord"],
        origin_coord=batch["global_origin_coord"],
        offset=batch["global_offset"],
        grid_size=batch["grid_size"][0],
    )
    point_ = model.student.backbone(global_point)
    point_ = model.up_cast(point_)
    point_enc2d = model.up_cast(
        point_, upcast_level=model.enc2d_upcast_level - model.up_cast_level
    )
    to_feature = model.pool_corr(point_enc2d, batch["global_correspondence"])

    offset0 = torch.cat(
        [torch.tensor([0], device=device), to_feature["offset"]], dim=0
    )
    enc2d_count = (
        offset0[1::model.num_global_view]
        - offset0[0:-1:model.num_global_view]
    )
    enc2d_offset = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(enc2d_count, dim=0)]
    )
    enc2d_mask = torch.cat([
        torch.arange(0, c, device=device) + offset0[i * model.num_global_view]
        for i, c in enumerate(enc2d_count)
    ], dim=0)
    offset_points_3d = enc2d_offset[1:]
    batch_points_3d = offset2batch(offset_points_3d)

    bincount_img_num = batch["img_num"]
    offset_img_num_full = bincount2offset(bincount_img_num)
    offset_img_num = torch.cat(
        [torch.tensor([0], device=device), offset_img_num_full]
    )[:-1]

    imgs = batch["images"]
    feature2d = model.ENC2D_forward(imgs)
    feature2d = feature2d.contiguous().view(-1, feature2d.shape[-1])

    feature3d = to_feature["feat"][enc2d_mask]
    correspondence = to_feature["correspondence"][enc2d_mask]
    valid_mask = torch.any(
        correspondence != torch.tensor([-1, -1], device=device), dim=2
    )
    valid_index = torch.where(valid_mask)

    batch_index = batch_points_3d[valid_index[0]]
    batch_img_num = offset_img_num[batch_index]
    feature3d_pixel = feature3d[valid_index[0]]
    feature_index_all = torch.cat([
        batch_img_num.unsqueeze(-1),
        valid_index[1].unsqueeze(-1),
        correspondence[valid_index],
    ], dim=-1).long()
    feature_index = (
        feature_index_all[:, 0] * model.patch_h * model.patch_w
        + feature_index_all[:, 1] * model.patch_h * model.patch_w
        + feature_index_all[:, 2] * model.patch_w
        + feature_index_all[:, 3]
    )
    feature3d_raw_full = torch_scatter.scatter_mean(
        feature3d_pixel, feature_index, dim=0, dim_size=feature2d.shape[0]
    )
    feature_index_unique = torch.unique(feature_index)

    return dict(
        feature3d_raw_full=feature3d_raw_full,
        feature2d=feature2d,
        feature_index_unique=feature_index_unique,
        feature_index_all=feature_index,
        feature_index_components=feature_index_all,
        valid_index=valid_index,
        offset_img_num=offset_img_num,
        bincount_img_num=bincount_img_num,
        imgs=imgs,
        patch_h=model.patch_h,
        patch_w=model.patch_w,
        total_patches=feature2d.shape[0],
    )


def _stats(name, x):
    n = x.norm(dim=-1)
    # pairwise cosine on a random sample (up to 512)
    K = x.shape[0]
    if K > 512:
        idx = torch.randperm(K, device=x.device)[:512]
        xs = x[idx]
    else:
        xs = x
    xn = F.normalize(xs.float(), dim=-1)
    cos = xn @ xn.T
    mask = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
    off = cos[mask]
    return (
        f"{name}: K={K}  norm[min/med/max]="
        f"{n.min():.3g}/{n.median():.3g}/{n.max():.3g}  "
        f"pairwise_cos[mean/std/p05/p95]="
        f"{off.mean():.3f}/{off.std():.3f}/"
        f"{off.quantile(0.05):.3f}/{off.quantile(0.95):.3f}"
    )


def _pca_summary(name, X):
    """
    PCA of the centered feature matrix. Returns (text summary,
    cumulative variance ratio tensor, effective rank).

    Effective rank (participation ratio) = (Σ σ²)² / Σ σ⁴ — a
    smooth scalar that says how many singular directions
    meaningfully contribute. Much more informative than just
    counting non-zero singulars.
    """
    X = X.float().detach().cpu()
    X = X - X.mean(0, keepdim=True)
    # SVD on the (K, D) centered matrix — singular values give us
    # the principal-component scales directly.
    s = torch.linalg.svdvals(X)
    var = s ** 2
    cum = torch.cumsum(var, dim=0) / var.sum()

    def _dim_for(threshold):
        idx = int((cum < threshold).sum().item()) + 1
        return min(idx, cum.numel())

    d50 = _dim_for(0.50)
    d90 = _dim_for(0.90)
    d95 = _dim_for(0.95)
    d99 = _dim_for(0.99)
    eff_rank = float((var.sum() ** 2) / (var * var).sum().clamp_min(1e-30))
    top_sigma_ratio = float(s[0] / s.mean().clamp_min(1e-30))
    summary = (
        f"{name}: D={X.shape[1]}, K={X.shape[0]}\n"
        f"  dims for [50/90/95/99]% variance: "
        f"{d50}/{d90}/{d95}/{d99}\n"
        f"  effective rank (participation ratio): {eff_rank:.1f}\n"
        f"  top σ / mean σ: {top_sigma_ratio:.1f}  (1.0 = uniform)"
    )
    return summary, cum, eff_rank


def _compute_enc2d_loss(model, f3_proj_sel, f2_sel):
    """
    Re-implement the enc2d loss inline so we can drive the overfit
    optimizer without running the full model.forward each step.
    Mirrors the four loss branches in utonia_v1m3b_qwen3_5_distill_ema.
    """
    if getattr(model, "enc2d_cos_shift", False) and model.enc2d_loss_type == "cosine":
        f2_sel = f2_sel - f2_sel.mean(dim=-1, keepdim=True)
        f3_proj_sel = f3_proj_sel - f3_proj_sel.mean(dim=-1, keepdim=True)

    if model.enc2d_loss_type == "cosine":
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return (1 - cos(f2_sel, f3_proj_sel)).mean() * 10
    if model.enc2d_loss_type == "cosine_bc":
        f2c = f2_sel.float() - f2_sel.float().mean(dim=0, keepdim=True)
        f3c = f3_proj_sel.float() - f3_proj_sel.float().mean(dim=0, keepdim=True)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return (1 - cos(f2c, f3c)).mean() * 10
    if model.enc2d_loss_type == "infonce_batch":
        f2c = f2_sel.float() - f2_sel.float().mean(dim=0, keepdim=True)
        f3c = f3_proj_sel.float() - f3_proj_sel.float().mean(dim=0, keepdim=True)
        f3n = F.normalize(f3c, dim=-1)
        f2n = F.normalize(f2c, dim=-1)
        K_full = f3n.shape[0]
        sub = getattr(model, "infonce_batch_subsample", None)
        if sub is not None and sub < K_full:
            idx = torch.randperm(K_full, device=f3n.device)[:sub]
            f3n = f3n[idx]; f2n = f2n[idx]
        sim = (f3n @ f2n.T) / model.infonce_temperature
        labels = torch.arange(sim.shape[0], device=sim.device)
        return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))
    # Fallback: per-scene infonce — collapse to batch-wide for the
    # debug overfit test (we don't have scene_id easily here).
    f2c = f2_sel.float() - f2_sel.float().mean(dim=0, keepdim=True)
    f3c = f3_proj_sel.float() - f3_proj_sel.float().mean(dim=0, keepdim=True)
    f3n = F.normalize(f3c, dim=-1)
    f2n = F.normalize(f2c, dim=-1)
    sim = (f3n @ f2n.T) / model.infonce_temperature
    labels = torch.arange(sim.shape[0], device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


def _visualize_correspondence(out_path, imgs, feature_index_components,
                              valid_index, patch_h, patch_w):
    """
    Render the first image with patch-grid overlay. Patches that
    received at least one (point, image) pair are highlighted green;
    unmapped ones gray.
    """
    img0 = imgs[0].detach().float().cpu()
    img0 = img0 * 0.5 + 0.5  # Qwen normalize: mean=0.5 std=0.5
    img0 = img0.clamp(0, 1).permute(1, 2, 0).numpy()

    # Find which patches in image 0 received a mapping.
    # `feature_index_components` columns are (batch_img_offset,
    # local_img_idx_within_batch_elem, row, col). image 0 globally is
    # batch_img_offset=0 and local_img_idx=0.
    mask = (
        (feature_index_components[:, 0] == 0)
        & (feature_index_components[:, 1] == 0)
    )
    rows = feature_index_components[mask, 2].cpu().numpy()
    cols = feature_index_components[mask, 3].cpu().numpy()
    mapped_grid = np.zeros((patch_h, patch_w), dtype=int)
    for r, c in zip(rows, cols):
        if 0 <= r < patch_h and 0 <= c < patch_w:
            mapped_grid[r, c] += 1

    H, W = img0.shape[:2]
    psize_h = H / patch_h
    psize_w = W / patch_w

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img0)
    ax[0].set_title("Input image (Qwen un-normalized)")
    ax[0].axis("off")

    ax[1].imshow(img0)
    for r in range(patch_h):
        for c in range(patch_w):
            if mapped_grid[r, c] > 0:
                color = (0, 1, 0, 0.35)
            else:
                color = (0.5, 0.5, 0.5, 0.15)
            rect = Rectangle(
                (c * psize_w, r * psize_h), psize_w, psize_h,
                facecolor=color, edgecolor="white", linewidth=0.3,
            )
            ax[1].add_patch(rect)
    n_mapped = int((mapped_grid > 0).sum())
    n_total = patch_h * patch_w
    ax[1].set_title(
        f"Patch coverage: {n_mapped}/{n_total} ({100*n_mapped/n_total:.1f}%) mapped\n"
        f"(green = at least one 3D point projects here)"
    )
    ax[1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return n_mapped, n_total


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "debug.log")
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"[setup] parsing config: {args.config_file}")
    cfg = default_config_parser(args.config_file, None)

    if hasattr(cfg.data.train, "datasets"):
        ds_cfg = cfg.data.train.datasets[0]
    else:
        ds_cfg = cfg.data.train
    ds_cfg.test_mode = False
    log(f"[setup] building dataset: {ds_cfg.type}")
    ds = build_dataset(ds_cfg)
    log(f"[setup] dataset size: {len(ds)}")

    log(f"[setup] building model: {cfg.model.type}")
    model = build_model(cfg.model)
    _load_into_model(model, args.weight, "aligned")
    model = model.to(args.device).eval()
    log(f"[setup] enc2d_loss_type={model.enc2d_loss_type!r}  "
        f"infonce_temperature={getattr(model, 'infonce_temperature', None)}  "
        f"subsample={getattr(model, 'infonce_batch_subsample', None)}  "
        f"cos_shift={getattr(model, 'enc2d_cos_shift', None)}")

    # Print patch_proj / qwen_proj structure to confirm Linear vs MLP
    # and detect two-tower (v1m3-G+) mode.
    log(f"[setup] patch_proj = {model.patch_proj}")
    has_qwen_proj = getattr(model, "common_dim", None) is not None
    if has_qwen_proj:
        log(f"[setup] qwen_proj  = {model.qwen_proj}")
        log(f"[setup] common_dim = {model.common_dim} (two-tower mode)")

    # ---- Load single sample ----
    log(f"\n[batch] loading scene idx={args.scene_idx}")
    raw = ds[args.scene_idx]
    sample = _coerce_sample_for_model(raw)
    batch = {
        k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
        for k, v in sample.items()
    }

    # ---- Optional: layer-by-layer Qwen ViT probe ----
    if args.probe_all_layers:
        model._probe_out_dir = args.out_dir
        _probe_all_layers(model, batch, args.device, log)

    # ---- (1) + (2): K stats and feature stats ----
    with torch.no_grad():
        idx_state = _build_indices(model, batch, args.device)

    K_unique = idx_state["feature_index_unique"].numel()
    N_pairs = idx_state["feature_index_all"].numel()
    total_patches = idx_state["total_patches"]
    log(f"\n[K stats]")
    log(f"  total patches in batch     = {total_patches} "
        f"(num_images * H*W = {batch['img_num'].sum().item()} * "
        f"{idx_state['patch_h']*idx_state['patch_w']})")
    log(f"  raw (point,image) pairs    = {N_pairs}")
    log(f"  unique patches with map    = {K_unique}  "
        f"(coverage {100.0*K_unique/total_patches:.2f}%)")
    log(f"  avg pairs per unique patch = {N_pairs/max(1,K_unique):.2f}")

    f2_sel = idx_state["feature2d"][idx_state["feature_index_unique"]].float()
    f3_raw_sel = idx_state["feature3d_raw_full"][idx_state["feature_index_unique"]].float()
    with torch.no_grad():
        f3_proj_sel = model.patch_proj(f3_raw_sel).float()
        f2_proj_sel = (
            model.qwen_proj(f2_sel).float() if has_qwen_proj else None
        )

    log(f"\n[feature stats]")
    log(f"  {_stats(f'f2_qwen   ({f2_sel.shape[1]}d)', f2_sel)}")
    log(f"  {_stats(f'f3_raw    ({f3_raw_sel.shape[1]}d)', f3_raw_sel)}")
    log(f"  {_stats(f'f3_proj   ({f3_proj_sel.shape[1]}d)', f3_proj_sel)}")
    if f2_proj_sel is not None:
        log(f"  {_stats(f'f2_proj   ({f2_proj_sel.shape[1]}d)', f2_proj_sel)}")

    # ---- (2b): PCA — how concentrated is the variance? ----
    # If most variance lives in a tiny subspace (small "effective rank")
    # then patch_proj must hit a narrow target — exactly the H3 failure
    # mode v1m3-G's two-tower projection is designed to dodge.
    log(f"\n[PCA] effective dimensionality of each feature space")
    pca_targets = [
        ("f2_qwen (raw Qwen patches)", f2_sel),
        ("f3_raw  (PTv3 output)     ", f3_raw_sel),
        ("f3_proj (patch_proj out)  ", f3_proj_sel),
    ]
    if f2_proj_sel is not None:
        pca_targets.append(("f2_proj (qwen_proj out)   ", f2_proj_sel))
    cum_curves = []
    for name, X in pca_targets:
        summary, cum, _ = _pca_summary(name, X)
        log(f"  {summary}")
        cum_curves.append((name, cum))

    # Cumulative variance plot.
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, cum in cum_curves:
        ax.plot(np.arange(1, cum.numel() + 1), cum.numpy(), label=name.strip())
    ax.set_xscale("log")
    ax.set_xlabel("# principal components (log)")
    ax.set_ylabel("cumulative variance ratio")
    ax.axhline(0.9, ls="--", color="gray", alpha=0.4)
    ax.axhline(0.99, ls=":", color="gray", alpha=0.4)
    ax.set_title("Cumulative variance — narrow curves = narrow signal subspace")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pca_path = os.path.join(args.out_dir, "pca_cumulative_variance.png")
    plt.savefig(pca_path, dpi=120)
    plt.close()
    log(f"  curve -> {pca_path}")

    # ---- (3): gradient flow check ----
    log(f"\n[gradient flow] one-step backward, enc2d_loss only")
    model.patch_proj.requires_grad_(True)
    if has_qwen_proj:
        model.qwen_proj.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    # Recompute f3_proj WITH grad. For two-tower also drive grad
    # through qwen_proj.
    f3_proj_grad = model.patch_proj(f3_raw_sel.detach())
    f2_for_loss = (
        model.qwen_proj(f2_sel.detach()) if has_qwen_proj else f2_sel.detach()
    )
    loss = _compute_enc2d_loss(model, f3_proj_grad, f2_for_loss)
    loss.backward()
    log(f"  loss(0) = {loss.item():.4f}")
    for name, p in model.patch_proj.named_parameters():
        wn = p.detach().norm().item()
        gn = p.grad.norm().item() if p.grad is not None else 0.0
        ratio = gn / (wn + 1e-12)
        log(f"  patch_proj.{name:15s}  |w|={wn:.3e}  |g|={gn:.3e}  "
            f"|g|/|w|={ratio:.3e}")
    if has_qwen_proj:
        for name, p in model.qwen_proj.named_parameters():
            wn = p.detach().norm().item()
            gn = p.grad.norm().item() if p.grad is not None else 0.0
            ratio = gn / (wn + 1e-12)
            log(f"  qwen_proj.{name:15s}  |w|={wn:.3e}  |g|={gn:.3e}  "
                f"|g|/|w|={ratio:.3e}")

    # ---- (4): single-batch overfit ----
    head_label = "patch_proj + qwen_proj" if has_qwen_proj else "patch_proj"
    log(f"\n[overfit] cached forward — running {args.overfit_steps} steps "
        f"on {head_label} alone (Adam lr={args.overfit_lr})")
    # Keep the trained head weights so we measure the *current* state's
    # ability to keep descending on a fixed batch (i.e. is there still
    # learnable signal left given the trained init?).
    trainable_params = list(model.patch_proj.parameters())
    if has_qwen_proj:
        trainable_params += list(model.qwen_proj.parameters())
    opt = torch.optim.Adam(trainable_params, lr=args.overfit_lr)
    f3_raw_cached = f3_raw_sel.detach()
    f2_cached = f2_sel.detach()
    history = []
    for step in range(args.overfit_steps + 1):
        opt.zero_grad(set_to_none=True)
        f3_proj_step = model.patch_proj(f3_raw_cached)
        f2_step = (
            model.qwen_proj(f2_cached) if has_qwen_proj else f2_cached
        )
        l = _compute_enc2d_loss(model, f3_proj_step, f2_step)
        l.backward()
        opt.step()
        if step % max(1, args.overfit_steps // 10) == 0 or step == args.overfit_steps:
            with torch.no_grad():
                f3_proj_eval = model.patch_proj(f3_raw_cached).float()
                f2_eval = (
                    model.qwen_proj(f2_cached).float() if has_qwen_proj
                    else f2_cached.float()
                )
                f3c = f3_proj_eval - f3_proj_eval.mean(dim=0, keepdim=True)
                f2c = f2_eval - f2_eval.mean(dim=0, keepdim=True)
                f3n = F.normalize(f3c, dim=-1)
                f2n = F.normalize(f2c, dim=-1)
                pos = (f3n * f2n).sum(dim=-1).mean().item()
                perm = torch.randperm(f2n.shape[0], device=f2n.device)
                neg = (f3n * f2n[perm]).sum(dim=-1).mean().item()
            history.append((step, l.item(), pos, neg))
            log(f"  step {step:4d}: loss={l.item():.4f}  pos_bc={pos:+.4f}  "
                f"neg_bc={neg:+.4f}  gap={pos-neg:+.4f}")

    # Plot the curve.
    if history:
        xs = [h[0] for h in history]
        losses = [h[1] for h in history]
        gaps = [h[2] - h[3] for h in history]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(xs, losses, marker="o")
        ax[0].set_xlabel("step"); ax[0].set_ylabel("enc2d_loss")
        ax[0].set_title(f"Single-batch overfit ({model.enc2d_loss_type})")
        ax[0].grid(alpha=0.3)
        ax[1].plot(xs, gaps, marker="o", color="tab:green")
        ax[1].set_xlabel("step"); ax[1].set_ylabel("pos_bc - neg_bc")
        ax[1].set_title("Discriminative gap")
        ax[1].grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(args.out_dir, "overfit_curve.png")
        plt.savefig(out, dpi=120)
        plt.close()
        log(f"  curve -> {out}")

    # ---- (5): correspondence visualization ----
    vis_path = os.path.join(args.out_dir, "correspondence_grid.png")
    n_mapped, n_total = _visualize_correspondence(
        vis_path,
        idx_state["imgs"],
        idx_state["feature_index_components"],
        idx_state["valid_index"],
        idx_state["patch_h"],
        idx_state["patch_w"],
    )
    log(f"\n[viz] {vis_path}  ({n_mapped}/{n_total} patches mapped in image 0)")

    # ---- Save full log ----
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"\n[done] full log -> {log_path}")


if __name__ == "__main__":
    main()
