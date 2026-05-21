"""
Comprehensive Qwen↔Utonia alignment evaluation.

Extends the simpler tools/eval_alignment.py (pos/neg cosine histogram) with
two further quantitative signals that together give a much fuller picture
of how well the trained patch_proj + backbone actually align with the
frozen Qwen ViT feature manifold.

Metrics computed
================
Three complementary signals, each answering a different question
about how well Utonia's 3D backbone has been pulled toward Qwen's
visual feature space.  All three use the SAME pre-computed
correspondence (offline ray-cast: "3D point P_i projects to 2D patch
patch_i") — correspondence is not measured; it's the *ground truth
label* against which the LEARNED feature alignment is evaluated.

  (1) Cosine distribution — PAIR DISCRIMINABILITY
        Question: "do learned features make correctly-paired
        (3D point, 2D patch) look more alike than random pairs?"

        pos_cos : cosine sim between projected 3D point feature and
                  the Qwen patch feature it correspondence-pairs with.
        neg_cos : same, but against a within-scene shuffled patch
                  (negative sample).

        Also reports BATCH-CENTERED variants (pos_bc, neg_bc) — the
        more honest version that strips Qwen's anisotropic DC offset.
        The gap (pos_bc - neg_bc) is the actual discrimination signal.

  (2) Within-scene patch retrieval — FINE-GRAINED LOCALIZATION
        Question: "can the 3D feature exactly identify WHICH of the K
        patches in this scene it should match?"

        For each scene, build the K×K cosine sim matrix between the
        K projected 3D features and the K Qwen patch features (K =
        valid patches in the scene). True match for row i is column
        i (the correspondence-paired patch). Rank of the true match:
            R@1 / R@5 / R@10 : fraction of points whose correct patch
                               is ranked top-1 / top-5 / top-10.
            MRR              : mean reciprocal rank.
            mean_rank        : average rank (1 = perfect localization).

        High R@K = the model picks the correct patch among many
        candidates of the same scene.  R@10 >> R@1 means "correct
        region but adjacent patches confound" (spatial smoothness).

  (3) Linear CKA — DISTRIBUTIONAL / STRUCTURAL ALIGNMENT
        Question: "does the OVERALL geometry of the 3D feature cloud
        match the overall geometry of Qwen's patch feature cloud?"

        Centered Kernel Alignment between all 3D features (across
        scenes) and all Qwen patch features.  Scalar in [0, 1] —
        DIMENSION-AGNOSTIC, so 1332-d backbone vs 2560-d Qwen patches
        can be compared directly with no learned projector.

        Three CKA values reported:
          cka_aligned_proj      : patch_proj(3D)  vs Qwen 2D
          cka_aligned_backbone  : raw  backbone(3D) vs Qwen 2D
                                  (← the "fair" comparison; doesn't
                                   depend on whether patch_proj is
                                   loaded or randomly initialized)
          cka_baseline_backbone : same but for --baseline-weight,
                                  e.g. utonia.pth (pre-Qwen-alignment
                                  reference) — usually ~0.30 floor.

How the three signals interact
==============================
        | pos_bc | retrieval R@1 | CKA aligned |
--------+--------+---------------+-------------+
Healthy |  high  |     high      |    high     |
Trained |   ↑    |      ↑↑       |     ↑       |  ← what we want
Smooth  |  high  |    low R@1    |    high     |  ← Qwen merger
        |        |    high R@10  |             |    smoothness
Trivial |   ↓    |    chance     |     ↓       |  ← representation
collapse|        |               |             |    collapse
--------+--------+---------------+-------------+

Calibration on this codebase (Qwen3.5-4B teacher, ScanNet):
  cka_baseline (random PT-v3 vs Qwen)          ≈ 0.30
  cka_aligned_backbone (H, 5 epoch, scannet)   ≈ 0.42
  cka_aligned_proj    (H)                       ≈ 0.48
  pos_bc (H)                                    ≈ 0.53
  neg_bc (H)                                    ≈ 0.001 (no DC bias)
  R@1 / R@5 / R@10 (H)                          ≈ 0.06 / 0.20 / 0.31
  R@K chance (K≈200-256)                        ≈ 0.004 / 0.02 / 0.04

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
    # If use_full_merger is on, ENC2D_forward returns 16×16 tokens, so
    # correspondence values (still in 32×32 units) must be downsampled
    # to the effective grid. Falls back to identity for legacy configs.
    eph = getattr(model, "effective_patch_h", model.patch_h)
    epw = getattr(model, "effective_patch_w", model.patch_w)
    stride = getattr(model, "correspondence_stride", 1)
    # Clamp the spatial axes to the effective grid before composing the
    # flat index — same guard as the training loss path.  In G this is
    # a no-op (stride=1, eph=patch_h); in H the half-resolution grid is
    # tight enough that any pool_corr float that lands at exactly patch_h
    # would escape and trigger the CUDA OOB assert at scatter time.
    batch_img_idx = feature_index[:, 0]
    view_img_idx = feature_index[:, 1]
    row_eff = (feature_index[:, 2] // stride).clamp_(0, eph - 1)
    col_eff = (feature_index[:, 3] // stride).clamp_(0, epw - 1)
    feature_index = (
        batch_img_idx * eph * epw
        + view_img_idx * eph * epw
        + row_eff * epw
        + col_eff
    )
    # Bound-check.  Eval runs single-sample (no batching), so we'd rather
    # drop the offending rows and finish the scene than let the silent
    # CUDA assert kill the whole eval loop.  Print diagnostic stats so
    # the offending tensor shape mismatch is visible.
    _N = feature2d.shape[0]
    _bad = (feature_index < 0) | (feature_index >= _N)
    if _bad.any():
        n_bad = int(_bad.sum().item())
        max_b = int(batch_img_idx.max().item())
        max_v = int(view_img_idx.max().item())
        max_ix = int(feature_index[_bad].max().item())
        print(
            f"  [eval_alignment_full] feature_index OOB in this scene: "
            f"{n_bad}/{feature_index.numel()} rows outside [0, {_N}); "
            f"eph/epw={eph}/{epw} stride={stride} "
            f"feature2d.shape={tuple(feature2d.shape)} "
            f"imgs.shape={tuple(imgs.shape)} "
            f"img_num={int(bincount_img_num.sum())} "
            f"max(batch_img_num)={max_b} max(view_img_idx)={max_v} "
            f"max(bad_idx)={max_ix} — dropping offending rows."
        )
        _keep = ~_bad
        feature_index = feature_index[_keep]
        feature3d_pixel = feature3d_pixel[_keep]
        if feature_index.numel() == 0:
            return None, None, None
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
    #
    # Qwen ViT is loaded in bf16 but qwen_proj is float32 (built by
    # the model constructor with default dtype). Training has AMP
    # autocast to bridge those, eval doesn't — cast f2 to float32
    # before the matmul or this errors out.
    if getattr(model, "common_dim", None) is not None and hasattr(
        model, "qwen_proj"
    ):
        f2 = model.qwen_proj(f2.float())

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

    What this measures — INTRA-SCENE FINE-GRAINED ALIGNMENT
    -------------------------------------------------------
    For each 3D point in a scene, look at *all K Qwen patches in the
    same scene* and rank them by cosine similarity to the 3D feature.
    The "correct" patch is the one the 3D point correspondence-pairs
    with (via the offline ray-cast projection saved in
    correspondence/<id>.npy).

      Query     : f3[i]  (3D point feature, post patch_proj)
      Candidates: f2[0], ..., f2[K-1]  (all 2D patch features in scene)
      Correct   : f2[i]  (the patch this 3D point projects to)
      rank_i    : how many candidates have similarity ≥ similarity to
                  the correct patch (1 = perfect match)

    Interpretation:
      R@1  high  → model picks the EXACT correct patch top-1
      R@5  high  → correct patch is in top-5 nearest
      R@10 high  → correct patch in top-10
      MRR  → averaged 1/rank: smooth fine-grained discriminability

    Chance baseline (K candidates, random ordering):
      R@1  ≈ 1/K        R@5 ≈ 5/K        R@10 ≈ 10/K
      MRR  ≈ ln(K)/K    mean_rank ≈ K/2

    Failure modes:
      All R@K near chance  → no within-scene alignment learned
      R@10 >> R@1          → correct REGION found but spatially-smooth
                             features can't disambiguate adjacent
                             patches (Qwen merger smoothness)
      All R@K = 0          → ALL features ≈ identical (collapse)
    """
    K = f3.shape[0]
    if K < 2:
        return None  # no negatives possible
    f3n = F.normalize(f3, dim=-1)
    f2n = F.normalize(f2, dim=-1)
    sim = f3n @ f2n.T  # (K, K) — cosine sim of every 3D-2D feature pair
    # rank of true match in each row (diagonal entry)
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

    What this measures — DISTRIBUTIONAL / STRUCTURAL ALIGNMENT
    ----------------------------------------------------------
    CKA = Centered Kernel Alignment. Compares the *inter-point
    relational structure* of X-space and Y-space:

        CKA(X, Y) = ‖X̃ᵀ Ỹ‖_F² / (‖X̃ᵀ X̃‖_F · ‖Ỹᵀ Ỹ‖_F)
                  ∈ [0, 1]

    where X̃ = X - mean(X), Ỹ = Y - mean(Y).

    Intuition: X̃ᵀ X̃ is the N×N inter-point similarity pattern in
    X-space, Ỹᵀ Ỹ is the same in Y-space.  CKA asks "do these two
    similarity patterns match?" — i.e., "if point i is close to point
    j in X-space, is it also close in Y-space?"

    Key properties:
      - DIMENSION-AGNOSTIC: works between 1332-d and 2560-d
      - INVARIANT to orthogonal rotation and global scaling
      - 1.0  = perfect representational alignment
      - 0.0  = no shared structure (independent / random)

    Calibration in this codebase:
      cka_baseline (untrained PT-v3 vs Qwen)       ≈ 0.30  (natural floor)
      cka_aligned_backbone (H training, ~5 epoch)  ≈ 0.42
      cka_aligned_proj (post patch_proj)           ≈ 0.48
      Anything < 0.30 → features below natural baseline, likely collapsed

    Note: CKA does NOT measure individual pair correctness — it
    measures whether the OVERALL distribution shapes match.  A high
    CKA + low R@1 means "scene-level structure matches but specific
    pairs swap"; a low CKA + high R@1 is unusual but possible.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    XtY = X.T @ Y                  # (dx, dy)  cross-modal "kernel"
    XtX = X.T @ X                  # (dx, dx)  within-X relational kernel
    YtY = Y.T @ Y                  # (dy, dy)  within-Y relational kernel
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

    # ----------------------------------------------------------------
    # BATCH-CENTERED COSINE (pos_bc / neg_bc / discrim_gap_bc)
    # ----------------------------------------------------------------
    # What this measures — CORRESPONDENCE-LEVEL PAIR DISCRIMINATION
    #
    # For each correctly-paired (3D point, 2D patch) — paired meaning
    # the correspondence file says "3D point P_i projects to 2D patch
    # patch_i" — we ask: is f3(P_i) cosine-similar to f2(patch_i)?
    # And is it dissimilar to f2(random other patch)?
    #
    #     pos_bc[i] = cos( f3̃(P_i), f2̃(patch_i) )      ← paired
    #     neg_bc[i] = cos( f3̃(P_i), f2̃(patch_σ(i)) )   ← shuffled
    #
    # where f̃ = f - mean(f) ("batch-centered") strips the anisotropic
    # DC component.  Raw ViT cosines sit at ~0.95+ for ALL pairs
    # (Qwen-style anisotropy: every feature points roughly the same
    # direction) — batch-centering reveals the *informative* residual.
    #
    # Discriminative alignment expectation:
    #   pos_bc > 0  : paired features look alike (signal)
    #   neg_bc ≈ 0  : random pairs look unrelated (no anisotropic bias)
    #   discrim_gap = pos_bc - neg_bc  → how separated paired vs random
    #
    # Calibration in this codebase:
    #   H (5 epoch, scannet only):  pos_bc=0.53, neg_bc=0.001 (gap 0.5)
    #   Pre-alignment baseline:     pos_bc≈0,    neg_bc≈0  (gap 0)
    #   Collapsed features:         pos_bc≈0,    neg_bc≈0  (gap 0,
    #                                                       but ALL norms tiny)
    #
    # Distinction from CKA:
    #   pos_bc/neg_bc test INDIVIDUAL pair correctness (each correspondence
    #     is its own observation)
    #   CKA tests OVERALL distribution structure (all pairs collectively)
    #   Both can disagree: high CKA + low pos_bc means "structure matches
    #     but pairs are swapped"; high pos_bc + low CKA means "pairs OK
    #     but each scene's feature distribution is too unique" (rare).
    # ----------------------------------------------------------------
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
    """
    Best-effort load of a checkpoint into a Utonia model regardless of
    which format it was saved in.

    Three formats we've seen in the wild:

      A. Pointcept training-time `model_last.pth`:
            ckpt = {"state_dict": {"module.student.backbone.X": ...,
                                   "module.patch_proj.Y": ..., ...}}
         Strip `module.` → already matches v1m1/v1m3b's namespace.

      B. Published utonia HF release (`utonia.pth`):
            ckpt = {"config": "...", "state_dict": {"module.X": ...}}
         where X is the raw PT-v3 weight name (no `student.backbone.`
         prefix).  Must strip `module.` AND prepend `student.backbone.`.

      C. Same as B but without the `module.` prefix:
            ckpt = {"config": "...", "state_dict": {"X": ...}}
         Just prepend `student.backbone.`.

    Earlier we only handled A and C; B mapped to nonsense keys
    (`student.backbone.module.X`) so the published HF checkpoint
    silently loaded *zero* weights → eval ran against a randomly
    initialised backbone (CKA stuck at baseline ~0.31, retrieval at
    chance, cosine outputs in the NaN territory).

    Strategy: try each candidate key transform, keep the one that
    matches the most parameters of the live model.  Print the
    matched-key count and a handful of sample mapped/unmapped names
    so format mismatches surface immediately instead of producing
    silently meaningless metrics.
    """
    print(f"[setup/{label}] loading: {weight_path}")
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)

    # Resolve the raw state_dict regardless of outer wrapping.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        raw = ckpt
    else:
        raise TypeError(f"unexpected checkpoint type {type(ckpt)}")

    model_keys = set(model.state_dict().keys())

    def _strip_module(d):
        return {(k[len("module."):] if k.startswith("module.") else k): v
                for k, v in d.items()}

    candidates = []

    # 1. Identity (covers format A directly).
    candidates.append(("identity", dict(raw)))
    # 2. Strip module. only.
    candidates.append(("strip_module", _strip_module(raw)))
    # 3. Strip module. + prepend student.backbone. (format B).
    stripped = _strip_module(raw)
    candidates.append((
        "strip_module+student.backbone.",
        {f"student.backbone.{k}": v for k, v in stripped.items()},
    ))
    # 4. Prepend student.backbone. without stripping (format C with no
    #    module. prefix to start with).
    candidates.append((
        "student.backbone.",
        {f"student.backbone.{k}": v for k, v in raw.items()},
    ))

    # Pick the transform that hits the most live-model keys.
    best = None
    for name, sd in candidates:
        hits = sum(1 for k in sd.keys() if k in model_keys)
        if best is None or hits > best[0]:
            best = (hits, name, sd)
    hits, chosen, sd = best

    info = model.load_state_dict(sd, strict=False)
    print(f"[setup/{label}] transform={chosen!r}  "
          f"matched={hits}/{len(model_keys)} live-model keys  "
          f"missing={len(info.missing_keys)} "
          f"unexpected={len(info.unexpected_keys)}")
    # Surface a few sample names so format mismatches are obvious.
    sample_unexp = list(info.unexpected_keys)[:3]
    sample_miss = [k for k in info.missing_keys if "backbone" in k][:3]
    if sample_unexp:
        print(f"[setup/{label}]   sample unexpected: {sample_unexp}")
    if sample_miss:
        print(f"[setup/{label}]   sample missing (backbone): {sample_miss}")
    if hits == 0:
        raise RuntimeError(
            f"[setup/{label}] no checkpoint keys matched the model — "
            f"all four prefix strategies missed.  First 3 ckpt keys: "
            f"{list(raw.keys())[:3]}; first 3 model keys: "
            f"{list(model_keys)[:3]}"
        )


if __name__ == "__main__":
    main()
