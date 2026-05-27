"""
Effective-rank diagnostic for the H/I alignment features.

Why this exists
===============
Two pathologies that produce "scene-wide" cosine heatmaps at deployment
look identical from the metrics in eval_alignment_full but require
opposite fixes:

  (a) feature collapse  — the projected 3D features have rank ~10 even
                          though they live in a 512-d space.  Cosine
                          becomes meaningless because every direction
                          points roughly the same way.  Retraining /
                          longer SSL is the only fix.

  (b) smooth manifold   — features are full-rank but neighbors on the
                          object manifold are TOO similar (small
                          margins between fridge / wall / floor).
                          Retraining with hard negatives helps, but
                          inference-time post-processing (bg-subtract,
                          top-percentile, clustering) often suffices.

CKA distinguishes "structurally aligned" but not the (a) vs (b) split.
Effective rank (singular-value entropy) does.

Metrics
=======
For each space (Qwen 2D patches, Utonia 3D points, both projected to
the 512-d common), we measure on the pooled features (concat across
sampled scenes):

  effective_rank :  exp( -Σ p_i log p_i )    where p_i = σ_i / Σ σ_j
                    (Roy & Vetterli 2007)
                    Range: [1, D].  Full-rank uniform ≈ D.

  rank_at_99pct  :  smallest k such that Σ_{i≤k} σ_i² / Σ σ_j² ≥ 0.99
                    "How many components capture 99% of variance"

  participation  :  effective_rank / D  ∈ [1/D, 1]
                    Quick "fraction-of-dim used" heuristic.

Calibration (rough — varies by model size / training duration)
==============================================================
  Healthy aligned features        : participation ≳ 0.20   (≥ ~100 of 512)
  Mild smoothness                 : participation ~ 0.10
  Borderline collapse             : participation < 0.05
  Frank collapse                  : effective_rank < 10

Asymmetry between f2 and f3 is also informative:
  f2 narrow, f3 wide   → student over-broadened to compensate
  f2 wide,  f3 narrow  → student collapsed; ALL points look ≈ same
  both narrow          → joint collapse (degenerate solution)

Usage
=====
    cd third_party/Pointcept
    export PYTHONPATH=./
    python tools/feature_rank_diagnostic.py \\
        --config-file configs/utonia/distill-utonia-v1m3-I-scannet-only-qwen3_5-4b.py \\
        --weight     exp/utonia_q35_i/model/model_last.pth \\
        --num-scenes 20 \\
        --out-dir    exp/utonia_q35_i/feature_rank

Outputs
=======
    <out-dir>/spectrum.png   singular-value spectrum, log scale
    <out-dir>/summary.txt    effective_rank, rank_at_99, participation
"""

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pointcept.engines.defaults import default_config_parser
from pointcept.datasets import build_dataset
from pointcept.models import build_model
from pointcept.models.utils import offset2batch, bincount2offset
from pointcept.models.utils.structure import Point
import torch.nn.functional as F  # noqa: F401  (used inside _extract_pairs)
import torch_scatter


# ---------------------------------------------------------------------------
# Inline helpers — copied from eval_alignment_full.py so this script is
# standalone (avoids import-fragility on stale Pointcept installs).  If
# the originals diverge meaningfully, update here too.
# ---------------------------------------------------------------------------
def _coerce_sample_for_model(raw):
    """Promote Python scalars to 1-element tensors (mimics DataLoader
    collate for a single sample)."""
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
    """Best-effort checkpoint load; tries 4 prefix strategies and picks
    the one that hits the most live-model keys."""
    print(f"[setup/{label}] loading: {weight_path}")
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
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

    stripped = _strip_module(raw)
    candidates = [
        ("identity", dict(raw)),
        ("strip_module", stripped),
        ("strip_module+student.backbone.",
         {f"student.backbone.{k}": v for k, v in stripped.items()}),
        ("student.backbone.",
         {f"student.backbone.{k}": v for k, v in raw.items()}),
    ]
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
    if hits == 0:
        raise RuntimeError(
            f"[setup/{label}] no checkpoint keys matched the model.  "
            f"First 3 ckpt keys: {list(raw.keys())[:3]}; "
            f"first 3 model keys: {list(model_keys)[:3]}"
        )


@torch.inference_mode()
def _extract_pairs(model, batch, device):
    """Extract paired (f3_proj, f2, f3_raw) features for one scene.
    Mirror of eval_alignment_full._extract_pairs — duplicated here so
    this script is standalone."""
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
            return None, None, None

    feature3d_pixel_raw = torch_scatter.scatter_mean(
        feature3d_pixel, feature_index, dim=0, dim_size=feature2d.shape[0]
    )
    feature3d_pixel_proj = model.patch_proj(feature3d_pixel_raw)
    feature_index_unique = torch.unique(feature_index)
    f2 = feature2d[feature_index_unique]
    f3_proj = feature3d_pixel_proj[feature_index_unique]
    f3_raw = feature3d_pixel_raw[feature_index_unique]

    if getattr(model, "common_dim", None) is not None and hasattr(
        model, "qwen_proj"
    ):
        f2 = model.qwen_proj(f2.float())
    if getattr(model, "enc2d_cos_shift", False):
        f2 = f2 - f2.mean(dim=-1, keepdim=True)
        f3_proj = f3_proj - f3_proj.mean(dim=-1, keepdim=True)

    return f3_proj.float(), f2.float(), f3_raw.float()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--weight", required=True)
    p.add_argument("--num-scenes", type=int, default=20)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _spectrum(X):
    """Center X (N, D) on its mean and return singular values of the
    centered matrix.  Returns (sigmas, effective_rank, rank_at_99, D).
    Mean-centering kills the anisotropic DC offset that would dominate
    the leading singular value and mask the real rank structure.
    """
    X = X.float()
    Xc = X - X.mean(dim=0, keepdim=True)
    # `torch.linalg.svdvals` returns sorted-descending singular values.
    # Note: returns min(N, D) sigmas, not D.  Callers must use min(N, D)
    # as the denominator for participation, NOT D — otherwise a small-N
    # run looks like reduced participation by construction.
    s = torch.linalg.svdvals(Xc)
    s_np = s.detach().cpu().numpy().astype(np.float64)
    N, D = X.shape

    # Normalize for Shannon-entropy-style effective rank.
    s_sum = s_np.sum()
    if s_sum <= 0:
        # All-zero (or numerically zero) spectrum: features are
        # constant — no rank to report.  Return NaN sentinels rather
        # than {1.0, 1} which look like a legitimate rank-1 manifold
        # in summary.txt and would silently masquerade as "real result".
        return s_np, float("nan"), 0, D, N
    p = s_np / s_sum
    p_safe = np.clip(p, 1e-12, 1.0)
    H = -(p_safe * np.log(p_safe)).sum()
    eff_rank = float(np.exp(H))

    # Rank at 99% of variance (σ², not σ).  argmax over a boolean
    # threshold is clearer than searchsorted, and conveys the intent
    # "first index where cumulative variance crosses 99%" directly.
    cumvar = np.cumsum(s_np ** 2)
    total = cumvar[-1]
    rank99 = int(np.argmax(cumvar >= 0.99 * total) + 1)

    return s_np, eff_rank, rank99, D, N


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = default_config_parser(args.config_file, None)
    if hasattr(cfg.data.train, "datasets"):
        ds_cfg = cfg.data.train.datasets[0]
    else:
        ds_cfg = cfg.data.train
    ds_cfg.test_mode = False
    ds = build_dataset(ds_cfg)

    model = build_model(cfg.model)
    _load_into_model(model, args.weight, "aligned")
    model = model.to(args.device).eval()

    rng = np.random.RandomState(args.seed)
    n = min(args.num_scenes, len(ds))
    indices = rng.choice(len(ds), n, replace=False)

    f3_proj_all, f2_all = [], []
    skipped = 0
    for i, idx in enumerate(indices):
        try:
            raw = ds[int(idx)]
            sample = _coerce_sample_for_model(raw)
        except Exception as e:
            print(f"[{i+1}/{n}] DATA ERROR: {e}")
            skipped += 1
            continue
        batch = {
            k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
            for k, v in sample.items()
        }
        try:
            f3_proj, f2, _f3_raw = _extract_pairs(model, batch, args.device)
        except Exception as e:
            print(f"[{i+1}/{n}] FORWARD ERROR: {e}")
            skipped += 1
            continue
        if f3_proj is None:
            skipped += 1
            continue
        f3_proj_all.append(f3_proj.cpu())
        f2_all.append(f2.cpu())
        print(f"[{i+1}/{n}] K={f3_proj.shape[0]}")

    if not f3_proj_all:
        raise SystemExit("No usable scenes.")

    f3_cat = torch.cat(f3_proj_all, dim=0)
    f2_cat = torch.cat(f2_all, dim=0)
    print(f"[setup] pooled: f3_proj {tuple(f3_cat.shape)}  "
          f"f2 {tuple(f2_cat.shape)}")

    s3, er3, r99_3, D3, N3 = _spectrum(f3_cat)
    s2, er2, r99_2, D2, N2 = _spectrum(f2_cat)

    # Maximum representable rank is min(N, D), not D.  Normalize
    # participation by that so a small-N run doesn't silently look
    # like "reduced participation" when in fact features perfectly
    # span the available rank.  Warn at the bottom if N < D.
    f3_max_rank = max(min(N3, D3), 1)
    f2_max_rank = max(min(N2, D2), 1)

    summary = dict(
        config=args.config_file,
        weight=args.weight,
        scenes_used=int(n - skipped),
        n_f3=int(f3_cat.shape[0]), D_f3=int(D3),
        n_f2=int(f2_cat.shape[0]), D_f2=int(D2),
        f3_effective_rank=er3,
        f3_rank_at_99pct=r99_3,
        f3_max_rank=int(f3_max_rank),
        f3_participation=(er3 / f3_max_rank) if np.isfinite(er3) else float("nan"),
        f3_top1_singval=float(s3[0]) if s3.size else float("nan"),
        f3_median_singval=float(np.median(s3)) if s3.size else float("nan"),
        f2_effective_rank=er2,
        f2_rank_at_99pct=r99_2,
        f2_max_rank=int(f2_max_rank),
        f2_participation=(er2 / f2_max_rank) if np.isfinite(er2) else float("nan"),
        f2_top1_singval=float(s2[0]) if s2.size else float("nan"),
        f2_median_singval=float(np.median(s2)) if s2.size else float("nan"),
    )

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as fh:
        for k, v in summary.items():
            fh.write(f"{k}: {v}\n")

    # Spectrum plot (log y, both spaces overlaid).
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(s3) + 1), s3,
            color="#2c5fa3", lw=1.4,
            label=(f"f3_proj (Utonia post-patch_proj)  "
                   f"eff_rank={er3:.1f}  R@99={r99_3}"))
    ax.plot(np.arange(1, len(s2) + 1), s2,
            color="#c0392b", lw=1.4,
            label=(f"f2 (Qwen patches post-qwen_proj)  "
                   f"eff_rank={er2:.1f}  R@99={r99_2}"))
    ax.set_yscale("log")
    ax.set_xlabel("singular-value index (descending)")
    ax.set_ylabel("singular value (log)")
    ax.set_title(
        f"Centered SVD spectrum — pooled across "
        f"{summary['scenes_used']} scenes  (D={D3})"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    spec_path = os.path.join(args.out_dir, "spectrum.png")
    fig.savefig(spec_path, dpi=120)

    # Diagnosis text.  NaN effective_rank means the spectrum was all-zero
    # — surface as a hard failure rather than letting downstream thresholds
    # silently classify it.
    diag_lines = []
    if not np.isfinite(er3):
        diag_lines.append(
            "FAIL: f3_proj spectrum is all-zero — patch_proj output "
            "is the zero tensor (bad load? collapsed weights?)."
        )
    if not np.isfinite(er2):
        diag_lines.append(
            "FAIL: f2 spectrum is all-zero — qwen_proj output is zero."
        )
    if np.isfinite(er3) and er3 < 10:
        diag_lines.append(
            "WARN: f3_proj effective_rank < 10 — possible collapse."
        )
    if np.isfinite(summary["f3_participation"]) and summary["f3_participation"] < 0.05:
        diag_lines.append(
            f"WARN: f3 participation {summary['f3_participation']:.3f} < 0.05 "
            f"— 3D features barely use the available rank "
            f"(max={f3_max_rank})."
        )
    if np.isfinite(summary["f2_participation"]) and summary["f2_participation"] < 0.05:
        diag_lines.append(
            f"WARN: f2 participation {summary['f2_participation']:.3f} < 0.05 "
            "— Qwen patches projected into a very narrow cone (qwen_proj "
            "may itself be undertrained)."
        )
    if (np.isfinite(er3) and np.isfinite(er2)
            and abs(er3 - er2) > 0.5 * max(er3, er2)):
        diag_lines.append(
            f"NOTE: large asymmetry  eff_rank f3={er3:.1f}  f2={er2:.1f}  "
            "— one side has collapsed/over-broadened relative to the other."
        )
    if N3 < D3 or N2 < D2:
        diag_lines.append(
            f"NOTE: pooled sample N is smaller than feature dim "
            f"(f3: N={N3}/D={D3}, f2: N={N2}/D={D2}).  Participation is "
            f"normalized by min(N,D); increase --num-scenes for a tighter "
            f"upper bound on rank."
        )
    if not diag_lines:
        diag_lines.append("Spectrum looks healthy on both sides.")

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n=== Diagnosis ===")
    for line in diag_lines:
        print(f"  {line}")
    with open(summary_path, "a") as fh:
        fh.write("\nDiagnosis:\n")
        for line in diag_lines:
            fh.write(f"  {line}\n")
    print(f"\nWrote:\n  {summary_path}\n  {spec_path}")


if __name__ == "__main__":
    main()
