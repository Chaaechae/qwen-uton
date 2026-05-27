"""
2D → 3D retrieval evaluation — the diagnostic that eval_alignment_full
does NOT do, and that turns out to be the one that actually predicts
how well open-vocab text→3D localization works at deployment time.

Why a separate tool
===================
eval_alignment_full computes R@K in the 3D → 2D direction:

    "For each 3D point's projected feature, rank the 2D patches in
     this scene by cosine; how often is the correspondence-paired
     patch top-K?"

That's a meaningful sanity check but does NOT predict deployment
performance.  At deployment, the question is the REVERSE:

    "I have a 2D patch (or a few patches averaged inside a text-query
     bbox).  I want to find the 3D points it corresponds to.  Cosine
     against EVERY 3D point in the scene — does the right region
     surface?"

Cosine retrieval is asymmetric.  R@1 in one direction does NOT imply
R@1 in the other, especially when the candidate-set sizes differ.
At K×K (the eval_alignment_full setup) both directions see the same
K candidates, so they're often close.  But deployment expands the 3D
candidate set from K to N_s1 — often 1000x more — and that scale
change dominates the noise floor.

This tool measures both regimes so you can see exactly where the
deployment gap appears.

Metrics
=======
Phase A — K × K  (mirrors eval_alignment_full, just flipped)
    For each scene's K paired (point-cluster, patch) features, compute
    the K×K cosine sim matrix and rank the diagonal (correct match)
    in EACH ROW = each 2D patch.

      Query     : f2[i]               (2D patch i, post qwen_proj)
      Candidates: f3_avg[0..K-1]      (per-patch averaged 3D feat, post patch_proj)
      Correct   : f3_avg[i]
      rank_i    : how many candidates have sim ≥ sim(f2[i], f3_avg[i])

    Reports R@1 / R@5 / R@10, MRR.  Direct comparison to
    eval_alignment_full's 3D→2D R@K tells you whether the cosine
    matrix is approximately symmetric (it usually is at K×K scale).

Phase B — K × M  DEPLOYMENT-REALISTIC
    Same query side, but candidates = ALL per-point features in the
    scene (BEFORE the scatter_mean-per-patch averaging that
    eval_alignment_full applies).  This is the regime open-vocab
    text→3D localization runs in: a single query feature pitted
    against the full point cloud's worth of 3D features.

      Query     : f2[i]               (2D patch i)
      Candidates: f3_pt[0..M-1]       (per-point 3D feat, M points)
      Correct   : ANY point whose correspondence maps to patch i
                  (M is much larger than K; multiple points map to
                   the same patch)
      rank      : rank of the FIRST correct point in cosine order

    Reports R@1 / R@10 / R@100, MRR.

    Chance baseline (M candidates, random ordering, ~M_i correct):
      R@1 ≈ M_i/M    R@10 ≈ 10·M_i/M   R@100 ≈ 100·M_i/M

    If Phase A is healthy but Phase B is at chance, the bottleneck
    is the candidate-scale: the alignment "knows" the patch's
    average target but can't pick its specific points out of the
    crowd.  Post-processing (frustum, clustering, depth-slab) is
    the right fix; retraining is overkill.

    If both phases are at chance, the alignment itself is too smooth.
    Retraining with hard negatives / instance-level supervision is
    the only path.

Diagnostic ladder summary
=========================
    eval_alignment_full R@K (3D→2D, K×K)  high → semantic OK
    Phase A R@K (2D→3D, K×K)              high → symmetric retrieval
    Phase B R@K (2D→3D, K×M)              high → deployment-ready
                                          low  → need post-processing
                                                 (NOT retraining)
    Phase A high + Phase B low            → fix at inference time
    Phase A low                           → retraining likely needed

Usage
=====
    cd third_party/Pointcept
    export PYTHONPATH=./
    python tools/eval_2d_to_3d_retrieval.py \\
        --config-file configs/utonia/distill-utonia-v1m3-I-scannet-only-qwen3_5-4b.py \\
        --weight     exp/utonia_q35_i/model/model_last.pth \\
        --num-scenes 30 \\
        --out-dir    exp/utonia_q35_i/eval_2d_to_3d

Outputs
=======
    <out-dir>/per_scene.csv
    <out-dir>/summary.txt
    <out-dir>/rank_distribution.png   (Phase B rank distribution)
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the parser + data + model setup machinery from the sibling tool.
# We need access to a few internal helpers; they were written to be
# stateless so importing them directly is safe.
from eval_alignment_full import (  # type: ignore
    _coerce_sample_for_model,
    _load_into_model,
)

from pointcept.engines.defaults import default_config_parser
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.models.utils import offset2batch, bincount2offset
from pointcept.models.utils.structure import Point
import torch_scatter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weight", required=True,
                   help="Trained checkpoint (v1m3-H/I or similar).")
    p.add_argument("--num-scenes", type=int, default=30)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-phase-b", dest="phase_b",
                   action="store_false", default=True,
                   help="Skip the expensive K×M full-cloud retrieval. "
                        "Phase A only (K×K transposed retrieval).")
    return p.parse_args()


@torch.inference_mode()
def _extract_pairs_with_per_point(model, batch, device):
    """Same setup as eval_alignment_full._extract_pairs, but ALSO
    returns the per-3D-point projected features (no scatter_mean)
    and the patch index each point maps to.

    Returns dict with:
        f2             : (K, D)   Qwen patches, post qwen_proj
        f3_avg_proj    : (K, D)   per-patch averaged 3D, post patch_proj
        f3_pt_proj     : (M, D)   per-3D-point projected (M = # valid points)
        point_to_patch : (M,)     each point's K-index (∈ [0, K))
        K, M           : ints
    Returns None on any failure.
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
        return None

    feature3d = to_feature["feat"][enc2d_mask]
    correspondence = to_feature["correspondence"][enc2d_mask]
    valid_mask = torch.any(
        correspondence != torch.tensor([-1, -1], device=device), dim=2
    )
    valid_index = torch.where(valid_mask)
    if valid_index[0].numel() == 0:
        return None

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

    eph = getattr(model, "effective_patch_h", model.patch_h)
    epw = getattr(model, "effective_patch_w", model.patch_w)
    stride = getattr(model, "correspondence_stride", 1)
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

    _N = feature2d.shape[0]
    _bad = (feature_index < 0) | (feature_index >= _N)
    if _bad.any():
        _keep = ~_bad
        feature_index = feature_index[_keep]
        feature3d_pixel = feature3d_pixel[_keep]
        if feature_index.numel() == 0:
            return None

    # Per-patch averaged 3D feature (same as eval_alignment_full).
    feature3d_pixel_raw = torch_scatter.scatter_mean(
        feature3d_pixel, feature_index, dim=0, dim_size=feature2d.shape[0]
    )
    feature3d_pixel_proj = model.patch_proj(feature3d_pixel_raw)

    # The unique flat indices that actually appeared.
    feature_index_unique = torch.unique(feature_index)
    f2_patch = feature2d[feature_index_unique]
    f3_avg_proj = feature3d_pixel_proj[feature_index_unique]

    # Map each point's feature_index to its position in feature_index_unique
    # (K-index in [0, K)).  torch.unique returns sorted unique values, so
    # searchsorted gives the canonical mapping.
    point_to_patch = torch.searchsorted(feature_index_unique, feature_index)

    # Project per-point features without averaging — these are the
    # deployment-realistic candidates for Phase B.
    f3_pt_proj = model.patch_proj(feature3d_pixel)

    # Two-tower mode: qwen_proj lifts Qwen patches into the common space.
    if getattr(model, "common_dim", None) is not None and hasattr(
        model, "qwen_proj"
    ):
        f2_patch = model.qwen_proj(f2_patch.float())

    if getattr(model, "enc2d_cos_shift", False):
        f2_patch = f2_patch - f2_patch.mean(dim=-1, keepdim=True)
        f3_avg_proj = f3_avg_proj - f3_avg_proj.mean(dim=-1, keepdim=True)
        f3_pt_proj = f3_pt_proj - f3_pt_proj.mean(dim=-1, keepdim=True)

    return dict(
        f2=f2_patch.float(),
        f3_avg_proj=f3_avg_proj.float(),
        f3_pt_proj=f3_pt_proj.float(),
        point_to_patch=point_to_patch,
        K=int(f2_patch.shape[0]),
        M=int(f3_pt_proj.shape[0]),
    )


def _phase_a_2d_to_3d(f2, f3_avg):
    """K×K 2D→3D retrieval.  Same as the 3D→2D retrieval in
    eval_alignment_full but with the cosine matrix transposed —
    interpretation flips from `(query=3D, candidates=patches)` to
    `(query=patch, candidates=patch-averaged 3D)`.
    """
    K = f2.shape[0]
    if K < 2:
        return None
    f2n = F.normalize(f2, dim=-1, eps=1e-4)
    f3n = F.normalize(f3_avg, dim=-1, eps=1e-4)
    sim = f2n @ f3n.T  # (K, K)
    diag = sim.diag().unsqueeze(1)
    # tie-permissive rank: count candidates with sim >= correct sim
    rank = (sim >= diag).sum(dim=1).clamp(min=1)
    r = rank.float()
    return {
        "phaseA_K": K,
        "phaseA_R@1": float((rank == 1).float().mean()),
        "phaseA_R@5": float((rank <= 5).float().mean()),
        "phaseA_R@10": float((rank <= 10).float().mean()),
        "phaseA_MRR": float((1.0 / r).mean()),
        "phaseA_mean_rank": float(r.mean()),
    }


def _phase_b_full_cloud(f2, f3_pt, point_to_patch, max_query=None):
    """K × M 2D→3D retrieval against the full per-point candidate set.

    For each 2D patch i, the "correct" candidates are all points j with
    point_to_patch[j] == i.  Rank is the position of the FIRST such
    correct point in cosine-sorted order.

    Returns dict with R@1, R@10, R@100, MRR, mean_rank, plus the raw
    rank array for downstream histogram.

    max_query : optional cap on # of query patches to score (for very
                large K — uniformly sample if K > max_query).
    """
    K = f2.shape[0]
    M = f3_pt.shape[0]
    if K < 1 or M < 2:
        return None, None

    f2n = F.normalize(f2, dim=-1, eps=1e-4)
    f3n = F.normalize(f3_pt, dim=-1, eps=1e-4)

    # Cap query set if huge.
    if max_query is not None and K > max_query:
        sel = torch.randperm(K, device=f2.device)[:max_query]
        f2n = f2n[sel]
        K_eval = max_query
        sel_set = set(sel.tolist())
    else:
        sel = torch.arange(K, device=f2.device)
        K_eval = K
        sel_set = None

    sim = f2n @ f3n.T  # (K_eval, M)

    ranks = []
    n_correct_per_patch = []
    for row, query_patch_idx in enumerate(sel.tolist()):
        correct_mask = (point_to_patch == query_patch_idx)
        n_correct = int(correct_mask.sum().item())
        if n_correct == 0:
            continue
        n_correct_per_patch.append(n_correct)
        correct_sims = sim[row][correct_mask]
        best_correct = correct_sims.max()
        # tie-permissive rank: count candidates ≥ best correct cosine
        rank = int((sim[row] >= best_correct).sum().item())
        rank = max(1, rank)
        ranks.append(rank)

    if not ranks:
        return None, None

    ranks_np = np.asarray(ranks, dtype=np.float32)
    return {
        "phaseB_K_queried": int(K_eval),
        "phaseB_M_candidates": int(M),
        "phaseB_avg_correct_per_patch": float(np.mean(n_correct_per_patch)),
        "phaseB_R@1": float((ranks_np == 1).mean()),
        "phaseB_R@10": float((ranks_np <= 10).mean()),
        "phaseB_R@100": float((ranks_np <= 100).mean()),
        "phaseB_R@1000": float((ranks_np <= 1000).mean()),
        "phaseB_MRR": float((1.0 / ranks_np).mean()),
        "phaseB_mean_rank": float(ranks_np.mean()),
        "phaseB_median_rank": float(np.median(ranks_np)),
    }, ranks_np


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

    print(f"[setup] building model: {cfg.model.type}")
    model = build_model(cfg.model)
    _load_into_model(model, args.weight, "aligned")
    model = model.to(args.device).eval()

    rng = np.random.RandomState(args.seed)
    n = min(args.num_scenes, len(ds))
    indices = rng.choice(len(ds), n, replace=False)

    per_scene_rows = []
    all_ranks_phase_b = []
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
            ex = _extract_pairs_with_per_point(model, batch, args.device)
        except Exception as e:
            print(f"[{i+1}/{n}] idx={idx} FORWARD ERROR: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            skipped += 1
            continue
        if ex is None:
            print(f"[{i+1}/{n}] idx={idx} (no valid correspondences)")
            skipped += 1
            continue

        f2 = ex["f2"]
        f3_avg = ex["f3_avg_proj"]
        f3_pt = ex["f3_pt_proj"]
        p2p = ex["point_to_patch"]

        row = dict(
            idx=int(idx),
            name=sample.get("name", "?") if isinstance(sample, dict) else "?",
            K=ex["K"], M=ex["M"],
        )

        a = _phase_a_2d_to_3d(f2, f3_avg)
        if a is not None:
            row.update(a)

        if args.phase_b:
            b, ranks_b = _phase_b_full_cloud(f2, f3_pt, p2p)
            if b is not None:
                row.update(b)
                all_ranks_phase_b.append(ranks_b)

        per_scene_rows.append(row)

        msg = f"[{i+1}/{n}] idx={idx} K={ex['K']} M={ex['M']}"
        if a is not None:
            msg += (f"  A.R@1={a['phaseA_R@1']:.3f} "
                    f"A.R@10={a['phaseA_R@10']:.3f}")
        if args.phase_b and "phaseB_R@1" in row:
            msg += (f"  B.R@1={row['phaseB_R@1']:.3f} "
                    f"B.R@100={row['phaseB_R@100']:.3f}")
        print(msg)

    if not per_scene_rows:
        raise SystemExit("No usable scenes — check data path / correspondences.")

    # Aggregate.
    def col(k):
        return np.array([r.get(k, np.nan) for r in per_scene_rows])

    summary = dict(
        config=args.config_file,
        weight=args.weight,
        scenes_sampled=int(n),
        scenes_used=int(len(per_scene_rows)),
        scenes_skipped=int(skipped),

        # Phase A — K×K 2D→3D
        phaseA_R1_mean=float(np.nanmean(col("phaseA_R@1"))),
        phaseA_R5_mean=float(np.nanmean(col("phaseA_R@5"))),
        phaseA_R10_mean=float(np.nanmean(col("phaseA_R@10"))),
        phaseA_MRR_mean=float(np.nanmean(col("phaseA_MRR"))),
        phaseA_mean_rank=float(np.nanmean(col("phaseA_mean_rank"))),
        phaseA_K_avg=float(np.nanmean(col("phaseA_K"))),
    )
    if args.phase_b:
        summary.update(dict(
            phaseB_R1_mean=float(np.nanmean(col("phaseB_R@1"))),
            phaseB_R10_mean=float(np.nanmean(col("phaseB_R@10"))),
            phaseB_R100_mean=float(np.nanmean(col("phaseB_R@100"))),
            phaseB_R1000_mean=float(np.nanmean(col("phaseB_R@1000"))),
            phaseB_MRR_mean=float(np.nanmean(col("phaseB_MRR"))),
            phaseB_mean_rank=float(np.nanmean(col("phaseB_mean_rank"))),
            phaseB_median_rank=float(np.nanmean(col("phaseB_median_rank"))),
            phaseB_M_avg=float(np.nanmean(col("phaseB_M_candidates"))),
        ))
        # Chance baseline: roughly avg_correct_per_patch / M_avg per
        # candidate slot.  Phase B R@K chance ≈ K · avg_correct / M.
        avg_correct = float(np.nanmean(col("phaseB_avg_correct_per_patch")))
        M_avg = summary["phaseB_M_avg"]
        summary["phaseB_chance_R@1"] = avg_correct / max(M_avg, 1)
        summary["phaseB_chance_R@10"] = (10 * avg_correct) / max(M_avg, 1)
        summary["phaseB_chance_R@100"] = (100 * avg_correct) / max(M_avg, 1)

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

    fig_path = None
    if args.phase_b and all_ranks_phase_b:
        ranks_all = np.concatenate(all_ranks_phase_b)
        fig_path = os.path.join(args.out_dir, "rank_distribution.png")
        fig, ax = plt.subplots(figsize=(9, 5))
        # Log-x histogram so the tail is legible.
        bins = np.logspace(0, np.log10(max(ranks_all.max(), 10)), 60)
        ax.hist(ranks_all, bins=bins, color="#2c5fa3", alpha=0.7,
                edgecolor="black", linewidth=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("rank of first correct 3D point (lower is better)")
        ax.set_ylabel("# of 2D patch queries")
        ax.axvline(1, color="green", ls="--", lw=1, label="R@1 boundary")
        ax.axvline(10, color="orange", ls="--", lw=1, label="R@10")
        ax.axvline(100, color="red", ls="--", lw=1, label="R@100")
        ax.set_title(
            f"Phase B (2D→3D vs full cloud) — "
            f"R@1={summary['phaseB_R1_mean']:.3f}  "
            f"R@100={summary['phaseB_R100_mean']:.3f}  "
            f"M≈{summary['phaseB_M_avg']:.0f}"
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=120)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote:\n  {summary_path}\n  {csv_path}")
    if fig_path:
        print(f"  {fig_path}")


if __name__ == "__main__":
    main()
