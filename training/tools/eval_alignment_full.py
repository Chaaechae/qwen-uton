"""
Comprehensive Qwen↔Utonia alignment evaluation.

Extends the simpler tools/eval_alignment.py (pos/neg cosine histogram) with
two further quantitative signals that together give a much fuller picture
of how well the trained patch_proj + backbone actually align with the
frozen Qwen ViT feature manifold.

Metrics computed
================
For ~50 held-out val scenes, runs the alignment branch of the v1m3 model
and aggregates:

  (1) Cosine distribution
        pos_cos : cosine sim between projected 3D point feature and the
                  Qwen patch feature it correctly maps to.
        neg_cos : same, but against a within-scene shuffled patch.
        Reports mean/std/median/p05/p95 + density histogram PNG.

  (2) Within-scene patch retrieval
        For each scene, build the K×K cosine sim matrix between the K
        projected 3D features and the K Qwen patch features (K = number
        of valid patches in that scene). The "true match" for row i is
        column i. Compute the rank of the true match within each row.
            R@1 / R@5 / R@10 : fraction of rows whose true match is in
                               the top K candidates.
            MRR              : mean of 1 / rank of the true match.
            mean_rank        : average rank (1 = perfect).
        Discriminative alignment → high R@K, MRR near 1, mean_rank low.

  (3) Linear CKA
        Centered Kernel Alignment between the matrix of all 3D features
        (across scenes) and the matrix of all Qwen patch features.
        Single scalar in [0, 1] — dimension-agnostic, so it can compare
        a 1332-d backbone output to 1024-d Qwen patch features without
        a learned projector. With --baseline-weight, also computes the
        same CKA for an unaligned baseline checkpoint (typically
        utonia.pth) for a clean before/after comparison.

Usage (cluster, B variant)
==========================

    cd third_party/Pointcept
    export PYTHONPATH=./
    python tools/eval_alignment_full.py \\
        --config-file configs/utonia/distill-utonia-v1m3-B-scannet-only-qwen3_5-4b.py \\
        --weight     exp/utonia_q35_align_ssl/model/model_last.pth \\
        --baseline-weight /group-volume/Utonia/utonia.pth \\
        --num-scenes 50 \\
        --out-dir    exp/utonia_q35_align_ssl/alignment_eval_full

`--baseline-weight` is optional — omit to skip the baseline CKA column.

Outputs
=======
    <out-dir>/cosine_histogram.png
    <out-dir>/per_scene.csv             scene-level pos/neg/MRR/Mean rank
    <out-dir>/summary.txt               headline numbers, all metrics
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pointcept.engines.defaults import default_config_parser
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.models.utils import offset2batch, bincount2offset
from pointcept.models.utils.structure import Point


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weight", required=True,
                   help="Aligned (trained) v1m3-A or v1m3-B checkpoint.")
    p.add_argument("--baseline-weight", default=None,
                   help="Optional: baseline ckpt for CKA comparison "
                        "(e.g. /group-volume/Utonia/utonia.pth). Loaded into "
                        "a fresh PTv3 with the same architecture; patch_proj "
                        "is left random.")
    p.add_argument("--num-scenes", type=int, default=50)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-scene feature extraction (returns paired (f3d_proj, f2d, f3d_raw))
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _extract_pairs(model, batch, device):
    """
    Run the alignment branch on one sample and return:
        f3d_proj : (K, 1024)  patch_proj(3D feat)        — aligned to Qwen
        f2d      : (K, 1024)  Qwen ViT patch feature      — frozen
        f3d_raw  : (K, 1332)  backbone feat (pre-proj)   — used for baseline CKA
    K = # valid patches in this scene.
    Returns (None, None, None) if no valid pair.

    Indexing logic mirrors `Utonia-v1m3{a,b}.forward` enc2d branch but
    extracts the per-pair feature pairs instead of the cosine loss.
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

    imgs = batch["images"]
    if imgs.shape[0] == 0:
        return None, None, None

    feature3d = to_feature["feat"][enc2d_mask]
    correspondence = to_feature["correspondence"][enc2d_mask]
    valid_mask = torch.any(
        correspondence != torch.tensor([-1, -1], device=device), dim=2
    )
    valid_index = torch.where(valid_mask)
    if valid_index[0].numel() == 0:
        return None, None, None

    bincount_img_num = batch["img_num"]
    offset_img_num = bincount2offset(bincount_img_num)

    feature2d = model.ENC2D_forward(imgs)
    feature2d = feature2d.contiguous().view(-1, feature2d.shape[-1])

    offset_img_num = torch.cat(
        [torch.tensor([0], device=device), offset_img_num]
    )[:-1]
    batch_index = batch_points_3d[valid_index[0]]
    batch_img_num = offset_img_num[batch_index]

    feature3d_pixel = feature3d[valid_index[0]]
    feature_index = torch.cat([
        batch_img_num.unsqueeze(-1),
        valid_index[1].unsqueeze(-1),
        correspondence[valid_index],
    ], dim=-1).long()
    feature_index = (
        feature_index[:, 0] * model.patch_h * model.patch_w
        + feature_index[:, 1] * model.patch_h * model.patch_w
        + feature_index[:, 2] * model.patch_w
        + feature_index[:, 3]
    )
    # Per-patch averaged 3D feature (raw, pre-projection).
    feature3d_pixel_raw = torch_scatter.scatter_mean(
        feature3d_pixel, feature_index, dim=0, dim_size=feature2d.shape[0]
    )
    # Projected to Qwen dim.
    feature3d_pixel_proj = model.patch_proj(feature3d_pixel_raw)

    feature_index_unique = torch.unique(feature_index)
    f2 = feature2d[feature_index_unique]
    f3_proj = feature3d_pixel_proj[feature_index_unique]
    f3_raw = feature3d_pixel_raw[feature_index_unique]

    # Two-tower mode (v1m3-G+): the model has a learnable qwen_proj
    # that lifts Qwen patches into the same common space as patch_proj
    # output. Apply it so that f2 and f3_proj live in the same dim
    # (matmul / cosine / CKA all expect matching last-dim).
    if getattr(model, "common_dim", None) is not None and hasattr(
        model, "qwen_proj"
    ):
        f2 = model.qwen_proj(f2)

    if getattr(model, "enc2d_cos_shift", False):
        f2 = f2 - f2.mean(dim=-1, keepdim=True)
        f3_proj = f3_proj - f3_proj.mean(dim=-1, keepdim=True)

    # Cast to float32 for downstream matmul / cosine / CKA. Under AMP the
    # Qwen ViT and patch_proj outputs are bfloat16 / float32 mixed, which
    # makes matmul (`f3n @ f2n.T` in _retrieval_metrics, `X.T @ Y` in
    # _linear_cka) raise "expected scalar type Float but found BFloat16".
    return f3_proj.float(), f2.float(), f3_raw.float()


# ---------------------------------------------------------------------------
# Within-scene patch retrieval (Recall@K, MRR, mean rank)
# ---------------------------------------------------------------------------
def _retrieval_metrics(f3, f2):
    """
    f3, f2 : (K, D) — paired features. True match for row i is column i.
    Returns dict with R@1, R@5, R@10, MRR, mean_rank.
    """
    K = f3.shape[0]
    if K < 2:
        return None  # no negatives possible
    f3n = F.normalize(f3, dim=-1)
    f2n = F.normalize(f2, dim=-1)
    sim = f3n @ f2n.T  # (K, K)
    # rank of true match in each row
    diag = sim.diag().unsqueeze(1)  # (K, 1)
    rank = (sim >= diag).sum(dim=1)  # how many entries ≥ diagonal (tie-permissive)
    rank = rank.clamp(min=1)
    rank_f = rank.float()
    return {
        "K": K,
        "R@1": float((rank == 1).float().mean()),
        "R@5": float((rank <= 5).float().mean()),
        "R@10": float((rank <= 10).float().mean()),
        "MRR": float((1.0 / rank_f).mean()),
        "mean_rank": float(rank_f.mean()),
    }


# ---------------------------------------------------------------------------
# Linear CKA  (dimension-agnostic between X (N, dx) and Y (N, dy))
# ---------------------------------------------------------------------------
def _linear_cka(X, Y):
    """
    X : (N, dx) on CPU/GPU torch tensor, finite, can be different dim from Y.
    Y : (N, dy)
    Returns float in [0, 1]. Mean-centers along N first.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XtY = X.T @ Y                  # (dx, dy)
    XtX = X.T @ X                  # (dx, dx)
    YtY = Y.T @ Y                  # (dy, dy)
    num = (XtY ** 2).sum()
    den = torch.sqrt((XtX ** 2).sum() * (YtY ** 2).sum())
    return float(num / (den + 1e-12))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[setup] parsing config: {args.config_file}")
    cfg = default_config_parser(args.config_file, None)

    if hasattr(cfg.data.train, "datasets"):
        ds_cfg = cfg.data.train.datasets[0]
    else:
        ds_cfg = cfg.data.train
    ds_cfg.test_mode = False
    print(f"[setup] building dataset: {ds_cfg.type}")
    ds = build_dataset(ds_cfg)
    print(f"[setup] dataset size: {len(ds)}")

    print(f"[setup] building aligned model: {cfg.model.type}")
    model = build_model(cfg.model)
    _load_into_model(model, args.weight, "aligned")
    model = model.to(args.device).eval()

    baseline_model = None
    if args.baseline_weight:
        print(f"[setup] building baseline model (same arch, fresh patch_proj)")
        baseline_model = build_model(cfg.model)
        _load_into_model(baseline_model, args.baseline_weight, "baseline")
        baseline_model = baseline_model.to(args.device).eval()

    # Sample held-out scene indices.
    rng = np.random.RandomState(args.seed)
    n = min(args.num_scenes, len(ds))
    indices = rng.choice(len(ds), n, replace=False)

    cuda_gen = torch.Generator(device=args.device)
    cuda_gen.manual_seed(args.seed)

    pos_all, neg_all = [], []
    f3_proj_acc, f2_acc, f3_raw_acc = [], [], []
    base_f3_raw_acc = []  # baseline backbone feats (collected on same scenes)
    per_scene_rows = []
    skipped = 0

    for i, idx in enumerate(indices):
        try:
            raw = ds[int(idx)]
            sample = _coerce_sample_for_model(raw)
        except Exception as e:
            print(f"[{i+1}/{n}] idx={idx} DATA ERROR: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            skipped += 1
            continue

        batch = {
            k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
            for k, v in sample.items()
        }

        try:
            f3_proj, f2, f3_raw = _extract_pairs(model, batch, args.device)
        except Exception as e:
            print(f"[{i+1}/{n}] idx={idx} ALIGNED FORWARD ERROR: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            skipped += 1
            continue
        if f3_proj is None:
            print(f"[{i+1}/{n}] idx={idx} (no valid correspondences)")
            skipped += 1
            continue

        # cosine distribution
        pos = F.cosine_similarity(f3_proj, f2, dim=-1)
        perm = torch.randperm(f2.shape[0], generator=cuda_gen, device=args.device)
        neg = F.cosine_similarity(f3_proj, f2[perm], dim=-1)

        # within-scene retrieval
        retrieval = _retrieval_metrics(f3_proj, f2)

        pos_np = pos.cpu().numpy()
        neg_np = neg.cpu().numpy()
        pos_all.append(pos_np)
        neg_all.append(neg_np)

        # Accumulate for CKA (move to CPU to save GPU mem).
        f3_proj_acc.append(f3_proj.cpu())
        f2_acc.append(f2.cpu())
        f3_raw_acc.append(f3_raw.cpu())

        # Baseline forward on same scene (if requested).
        if baseline_model is not None:
            try:
                base_f3_proj, _, base_f3_raw = _extract_pairs(
                    baseline_model, batch, args.device
                )
                if base_f3_raw is not None:
                    base_f3_raw_acc.append(base_f3_raw.cpu())
            except Exception as e:
                print(f"[{i+1}/{n}] idx={idx} BASELINE FORWARD ERROR: "
                      f"{type(e).__name__}: {str(e)[:120]}")

        row = dict(
            idx=int(idx),
            name=sample.get("name", "?") if isinstance(sample, dict) else "?",
            n_pairs=int(pos_np.size),
            pos_mean=float(pos_np.mean()),
            pos_std=float(pos_np.std()),
            neg_mean=float(neg_np.mean()),
            neg_std=float(neg_np.std()),
        )
        if retrieval is not None:
            row.update(retrieval)
        per_scene_rows.append(row)

        msg = (f"[{i+1}/{n}] idx={idx} K={pos_np.size} "
               f"pos={pos_np.mean():.3f} neg={neg_np.mean():.3f}")
        if retrieval is not None:
            msg += (f" R@1={retrieval['R@1']:.3f} "
                    f"R@5={retrieval['R@5']:.3f} "
                    f"MRR={retrieval['MRR']:.3f}")
        print(msg)

    if not pos_all:
        raise SystemExit("No usable scenes — check data path / correspondences.")

    pos = np.concatenate(pos_all)
    neg = np.concatenate(neg_all)

    # Aggregate retrieval (weighted by # pairs is more honest, but mean per
    # scene is what people usually report; we report both).
    R1  = np.array([r.get("R@1",  np.nan) for r in per_scene_rows])
    R5  = np.array([r.get("R@5",  np.nan) for r in per_scene_rows])
    R10 = np.array([r.get("R@10", np.nan) for r in per_scene_rows])
    MRR = np.array([r.get("MRR",  np.nan) for r in per_scene_rows])
    MR  = np.array([r.get("mean_rank", np.nan) for r in per_scene_rows])

    # CKA (dimension-agnostic; runs on CPU to keep memory simple).
    print("[cka] computing linear CKA on accumulated features ...")
    f3_proj_cat = torch.cat(f3_proj_acc, dim=0)
    f2_cat      = torch.cat(f2_acc,      dim=0)
    f3_raw_cat  = torch.cat(f3_raw_acc,  dim=0)
    cka_aligned_proj = _linear_cka(f3_proj_cat, f2_cat)
    cka_aligned_raw  = _linear_cka(f3_raw_cat,  f2_cat)
    cka_baseline_raw = None
    if base_f3_raw_acc:
        base_f3_raw_cat = torch.cat(base_f3_raw_acc, dim=0)
        cka_baseline_raw = _linear_cka(base_f3_raw_cat, f2_cat)

    # Additional measurement: BATCH-centered cosine (subtract the mean
    # feature across all collected samples before computing cosine). This
    # strips the anisotropic DC component that makes raw cosines uniformly
    # ~0.95+ on ViT-flavoured features. The gap (pos - neg) under batch-
    # centered cosine is a more honest discrimination metric.
    f3_bc = f3_proj_cat - f3_proj_cat.mean(dim=0, keepdim=True)
    f2_bc = f2_cat - f2_cat.mean(dim=0, keepdim=True)
    # Report per-row magnitudes so degenerate / near-zero features (which
    # cause F.cosine_similarity to return values outside [-1,1] due to
    # divide-by-eps) are visible up front.
    f3_bc_norm = f3_bc.float().norm(dim=-1)
    f2_bc_norm = f2_bc.float().norm(dim=-1)
    n_tiny_f3 = int((f3_bc_norm < 1e-6).sum().item())
    n_tiny_f2 = int((f2_bc_norm < 1e-6).sum().item())
    print(f"[bc] f3_bc norm: median={f3_bc_norm.median().item():.4f}, "
          f"min={f3_bc_norm.min().item():.4e}, tiny(<1e-6)={n_tiny_f3}/{f3_bc.shape[0]}")
    print(f"[bc] f2_bc norm: median={f2_bc_norm.median().item():.4f}, "
          f"min={f2_bc_norm.min().item():.4e}, tiny(<1e-6)={n_tiny_f2}/{f2_bc.shape[0]}")

    # Robust cosine: explicit normalize with safer eps, then dot. Clip to
    # [-1,1] so any residual numerical drift can't push the mean outside
    # the legal range (F.cosine_similarity's default eps=1e-8 lets very
    # small denominators amplify numerator noise — we've seen mean=-4.97
    # in practice when patch_proj output shrank to near-zero magnitude).
    f3n_bc = F.normalize(f3_bc.float(), dim=-1, eps=1e-4)
    f2n_bc = F.normalize(f2_bc.float(), dim=-1, eps=1e-4)
    pos_bc = (f3n_bc * f2n_bc).sum(dim=-1).clamp(-1.0, 1.0).numpy()
    perm_bc = torch.randperm(f2_bc.shape[0])
    neg_bc = (f3n_bc * f2n_bc[perm_bc]).sum(dim=-1).clamp(-1.0, 1.0).numpy()

    summary = dict(
        config=args.config_file,
        weight=args.weight,
        baseline_weight=args.baseline_weight or "(none)",
        scenes_sampled=int(n),
        scenes_used=int(len(per_scene_rows)),
        scenes_skipped=int(skipped),
        total_pairs=int(pos.size),

        # Cosine distribution
        pos_mean=float(pos.mean()), pos_std=float(pos.std()),
        pos_median=float(np.median(pos)),
        pos_p05=float(np.percentile(pos, 5)),
        pos_p95=float(np.percentile(pos, 95)),
        neg_mean=float(neg.mean()), neg_std=float(neg.std()),
        neg_median=float(np.median(neg)),
        neg_p05=float(np.percentile(neg, 5)),
        neg_p95=float(np.percentile(neg, 95)),
        discrim_gap=float(pos.mean() - neg.mean()),

        # Within-scene retrieval (per-scene mean)
        retrieval_R1=float(np.nanmean(R1)),
        retrieval_R5=float(np.nanmean(R5)),
        retrieval_R10=float(np.nanmean(R10)),
        retrieval_MRR=float(np.nanmean(MRR)),
        retrieval_mean_rank=float(np.nanmean(MR)),

        # Linear CKA
        cka_aligned_proj_vs_qwen=cka_aligned_proj,
        cka_aligned_backbone_vs_qwen=cka_aligned_raw,
        cka_baseline_backbone_vs_qwen=cka_baseline_raw,

        # Batch-centered cosine (more honest under anisotropic features)
        pos_bc_mean=float(pos_bc.mean()),
        neg_bc_mean=float(neg_bc.mean()),
        discrim_gap_bc=float(pos_bc.mean() - neg_bc.mean()),
    )

    # ---- write outputs --------------------------------------------------
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    csv_path = os.path.join(args.out_dir, "per_scene.csv")
    if per_scene_rows:
        all_keys = list({k for r in per_scene_rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(per_scene_rows)

    fig_path = os.path.join(args.out_dir, "cosine_histogram.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-1.0, 1.0, 100)
    ax.hist(neg, bins=bins, alpha=0.55,
            label=f"neg (shuffle)  μ={summary['neg_mean']:.3f}",
            color="#c0392b", density=True)
    ax.hist(pos, bins=bins, alpha=0.55,
            label=f"pos (matched)  μ={summary['pos_mean']:.3f}",
            color="#2c5fa3", density=True)
    ax.axvline(summary["pos_mean"], color="#2c5fa3", ls="--", lw=1)
    ax.axvline(summary["neg_mean"], color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("cosine similarity (3D point feat ↔ 2D patch feat)")
    ax.set_ylabel("density")
    ax.set_title(
        f"{summary['scenes_used']} scenes, {summary['total_pairs']} pairs | "
        f"gap={summary['discrim_gap']:.3f} | "
        f"R@1={summary['retrieval_R1']:.3f} | "
        f"CKA={summary['cka_aligned_proj_vs_qwen']:.3f}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote:\n  {summary_path}\n  {csv_path}\n  {fig_path}")


def _coerce_sample_for_model(raw):
    """
    Mimic what DataLoader's collate would do for a single sample so model.forward
    can index things like `batch["grid_size"][0]`. The transform pipeline's
    `Update(keys_dict={"grid_size": 0.01})` runs AFTER ToTensor, so grid_size
    arrives as a plain Python float — promote it (and any other 0-D scalars)
    to a 1-element tensor. Everything else is passed through unchanged.
    """
    sample = dict(raw)
    for k in list(sample.keys()):
        v = sample[k]
        if isinstance(v, bool):
            sample[k] = torch.tensor([v], dtype=torch.bool)
        elif isinstance(v, int):
            sample[k] = torch.tensor([v], dtype=torch.long)
        elif isinstance(v, float):
            sample[k] = torch.tensor([v], dtype=torch.float32)
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            sample[k] = torch.tensor([v.item()])
    return sample


def _load_into_model(model, weight_path, label):
    """Best-effort load of a checkpoint (Pointcept training fmt or HF fmt)."""
    print(f"[setup/{label}] loading: {weight_path}")
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    # ORDER MATTERS: published utonia.pth has BOTH "config" and "state_dict"
    # at the top level (the HF format), so we must check for "config" FIRST.
    # Otherwise the raw PTv3 keys go through unprefixed and miss the model's
    # `student.backbone.` namespace entirely.
    if isinstance(ckpt, dict) and "config" in ckpt and "state_dict" in ckpt:
        # Published utonia HF format. Keys are raw PTv3 — wrap with the
        # student.backbone. prefix so they land on student.backbone.*.
        sd = {f"student.backbone.{k}": v for k, v in ckpt["state_dict"].items()}
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Pointcept training format (state_dict keys already have
        # `module.student.backbone.*` etc.).
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    info = model.load_state_dict(sd, strict=False)
    print(f"[setup/{label}] missing={len(info.missing_keys)}, "
          f"unexpected={len(info.unexpected_keys)}")


if __name__ == "__main__":
    main()
