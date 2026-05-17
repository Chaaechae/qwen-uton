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
    return p.parse_args()


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

    # Print patch_proj structure to confirm Linear vs MLP.
    log(f"[setup] patch_proj = {model.patch_proj}")

    # ---- Load single sample ----
    log(f"\n[batch] loading scene idx={args.scene_idx}")
    raw = ds[args.scene_idx]
    sample = _coerce_sample_for_model(raw)
    batch = {
        k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
        for k, v in sample.items()
    }

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

    log(f"\n[feature stats]")
    log(f"  {_stats('f2_qwen   (1024d)', f2_sel)}")
    log(f"  {_stats('f3_raw    (1332d)', f3_raw_sel)}")
    log(f"  {_stats('f3_proj   (1024d)', f3_proj_sel)}")

    # ---- (3): gradient flow check ----
    log(f"\n[gradient flow] one-step backward, enc2d_loss only")
    model.patch_proj.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    # Recompute f3_proj WITH grad on patch_proj.
    f3_proj_grad = model.patch_proj(f3_raw_sel.detach())
    loss = _compute_enc2d_loss(model, f3_proj_grad, f2_sel.detach())
    loss.backward()
    log(f"  loss(0) = {loss.item():.4f}")
    for name, p in model.patch_proj.named_parameters():
        wn = p.detach().norm().item()
        gn = p.grad.norm().item() if p.grad is not None else 0.0
        ratio = gn / (wn + 1e-12)
        log(f"  patch_proj.{name:15s}  |w|={wn:.3e}  |g|={gn:.3e}  "
            f"|g|/|w|={ratio:.3e}")

    # ---- (4): single-batch overfit ----
    log(f"\n[overfit] cached forward — running {args.overfit_steps} steps "
        f"on patch_proj alone (Adam lr={args.overfit_lr})")
    # Fresh-init patch_proj for an honest overfit test? Optional. Keeping
    # the trained weights to see whether the *current* state can still
    # descend on a single batch.
    opt = torch.optim.Adam(model.patch_proj.parameters(), lr=args.overfit_lr)
    f3_raw_cached = f3_raw_sel.detach()
    f2_cached = f2_sel.detach()
    history = []
    for step in range(args.overfit_steps + 1):
        opt.zero_grad(set_to_none=True)
        f3_proj_step = model.patch_proj(f3_raw_cached)
        l = _compute_enc2d_loss(model, f3_proj_step, f2_cached)
        l.backward()
        opt.step()
        if step % max(1, args.overfit_steps // 10) == 0 or step == args.overfit_steps:
            # Also compute discriminative gap (pos vs random neg) for ground truth.
            with torch.no_grad():
                f3_proj_eval = model.patch_proj(f3_raw_cached).float()
                f2 = f2_cached.float()
                f3c = f3_proj_eval - f3_proj_eval.mean(dim=0, keepdim=True)
                f2c = f2 - f2.mean(dim=0, keepdim=True)
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
