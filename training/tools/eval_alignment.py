"""
Patch-level Qwen↔Utonia alignment evaluation.

Loads a trained v1m3-A / v1m3-B checkpoint and runs the forward path on a
held-out subset of scenes. For every (point, image) pair with a valid
correspondence, computes:

  pos_cos : cosine similarity between the 3D point feature (projected to
            1024-d via patch_proj) and the Qwen ViT feature of the patch
            that point projects into.
  neg_cos : cosine similarity between the same 3D feature and a *randomly
            shuffled* 2D feature (within-batch shuffle).

A well-aligned model produces pos_cos clearly above neg_cos. Mode collapse
would show pos_cos near 1 with no diversity. No alignment learned → the
two distributions overlap.

Usage (from this script's parent dir, with PYTHONPATH set as for training):

    cd third_party/Pointcept
    export PYTHONPATH=./
    python tools/eval_alignment.py \
        --config-file configs/utonia/distill-utonia-v1m3-A-scannet-only-qwen3_5-4b.py \
        --weight     exp/utonia_q35_align_only/model/model_last.pth \
        --num-scenes 50 \
        --out-dir    exp/utonia_q35_align_only/alignment_eval

Outputs:
    <out-dir>/cosine_histogram.png   side-by-side pos / neg histogram
    <out-dir>/summary.txt            n_scenes, n_pairs, means, stds, gap
    <out-dir>/per_scene.csv          per-scene mean(pos), mean(neg)
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True,
                   help="Same config as training (v1m3-A or v1m3-B).")
    p.add_argument("--weight", required=True,
                   help="Trained checkpoint (model_last.pth / epoch_N.pth).")
    p.add_argument("--num-scenes", type=int, default=50,
                   help="How many held-out scenes to sample.")
    p.add_argument("--out-dir", required=True,
                   help="Where to write histogram.png, summary.txt, per_scene.csv.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for scene sampling and negative shuffle.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Alignment feature extraction
# ---------------------------------------------------------------------------
def _eval_one_sample(model, data_dict, device, generator):
    """
    Reproduce the alignment branch of `Utonia-v1m3{a,b}.forward` but, instead
    of returning a scalar loss, return per-pair cosine similarities for both
    positive (correct) and negative (shuffled) pairings.

    Returns (pos_cos, neg_cos) torch tensors on `device`, or (None, None) if
    the sample has no valid correspondences.
    """
    # Lift any tensor into device.
    batch = {}
    for k, v in data_dict.items():
        batch[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    # Build the global Point as model.forward does.
    global_point = Point(
        feat=batch.get("global_feat_full", batch["global_feat"]),
        coord=batch["global_coord"],
        origin_coord=batch["global_origin_coord"],
        offset=batch["global_offset"],
        grid_size=batch["grid_size"][0],
    )

    # No masking for evaluation — we want the "best case" features, not
    # the masked-input setting from training.
    with torch.inference_mode():
        point_ = model.student.backbone(global_point)
        point_ = model.up_cast(point_)
        point_enc2d = model.up_cast(
            point_,
            upcast_level=model.enc2d_upcast_level - model.up_cast_level,
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
            torch.arange(0, c, device=device)
            + offset0[i * model.num_global_view]
            for i, c in enumerate(enc2d_count)
        ], dim=0)

        offset_points_3d = enc2d_offset[1:]
        batch_points_3d = offset2batch(offset_points_3d)

        imgs = batch["images"]
        if imgs.shape[0] == 0:
            return None, None

        feature3d = to_feature["feat"][enc2d_mask]
        correspondence = to_feature["correspondence"][enc2d_mask]
        valid_mask = torch.any(
            correspondence != torch.tensor([-1, -1], device=device), dim=2
        )
        valid_index = torch.where(valid_mask)
        if valid_index[0].numel() == 0:
            return None, None

        bincount_img_num = batch["img_num"]
        offset_img_num = bincount2offset(bincount_img_num)

        # Qwen ViT forward (already wrapped in no_grad inside ENC2D_forward).
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
        feature3d_pixel = torch_scatter.scatter_mean(
            feature3d_pixel, feature_index, dim=0, dim_size=feature2d.shape[0]
        )
        feature3d_pixel = model.patch_proj(feature3d_pixel)

        # Pick only patches that actually received a point.
        feature_index_unique = torch.unique(feature_index)
        f2 = feature2d[feature_index_unique]
        f3 = feature3d_pixel[feature_index_unique]

        if getattr(model, "enc2d_cos_shift", False):
            f2 = f2 - f2.mean(dim=-1, keepdim=True)
            f3 = f3 - f3.mean(dim=-1, keepdim=True)

        pos_cos = F.cosine_similarity(f3, f2, dim=-1)

        # Negative pairs: random shuffle within this batch of patches.
        perm = torch.randperm(f2.shape[0], generator=generator, device=device)
        neg_cos = F.cosine_similarity(f3, f2[perm], dim=-1)

        return pos_cos, neg_cos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[setup] parsing config: {args.config_file}")
    cfg = default_config_parser(args.config_file, None)

    # Need batch_size_per_gpu set for default_setup, but we don't call it —
    # build the dataset directly. cfg.data.train.datasets[0] is the
    # ConcatDataset's only inner dataset for the scannet-only recipes.
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

    print(f"[setup] loading weights: {args.weight}")
    ckpt = torch.load(args.weight, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # Strip "module." prefix (DDP wrapper) if present.
    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    info = model.load_state_dict(state_dict, strict=False)
    print(f"[setup] missing keys: {len(info.missing_keys)}, "
          f"unexpected: {len(info.unexpected_keys)}")
    if info.missing_keys:
        print(f"        first missing: {info.missing_keys[:3]}")
    if info.unexpected_keys:
        print(f"        first unexpected: {info.unexpected_keys[:3]}")

    model = model.to(args.device).eval()

    # Sample scene indices.
    rng = np.random.RandomState(args.seed)
    n = min(args.num_scenes, len(ds))
    indices = rng.choice(len(ds), n, replace=False)

    # CUDA generator for the negative shuffle (deterministic per scene).
    cuda_gen = torch.Generator(device=args.device)
    cuda_gen.manual_seed(args.seed)

    pos_all = []
    neg_all = []
    per_scene_rows = []
    skipped = 0

    for i, idx in enumerate(indices):
        try:
            sample = ds[int(idx)]
        except Exception as e:
            print(f"[{i+1}/{n}] idx={idx} DATA ERROR: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            skipped += 1
            continue
        try:
            pos, neg = _eval_one_sample(model, sample, args.device, cuda_gen)
        except Exception as e:
            print(f"[{i+1}/{n}] idx={idx} MODEL ERROR: "
                  f"{type(e).__name__}: {str(e)[:120]}")
            skipped += 1
            continue
        if pos is None:
            print(f"[{i+1}/{n}] idx={idx} (no valid correspondences)")
            skipped += 1
            continue

        pos_np = pos.cpu().numpy()
        neg_np = neg.cpu().numpy()
        pos_all.append(pos_np)
        neg_all.append(neg_np)

        per_scene_rows.append({
            "idx": int(idx),
            "name": sample.get("name", "?") if isinstance(sample, dict) else "?",
            "n_pairs": int(pos_np.size),
            "pos_mean": float(pos_np.mean()),
            "pos_std": float(pos_np.std()),
            "neg_mean": float(neg_np.mean()),
            "neg_std": float(neg_np.std()),
        })
        print(f"[{i+1}/{n}] idx={idx} pairs={pos_np.size} "
              f"pos={pos_np.mean():.3f}±{pos_np.std():.3f} "
              f"neg={neg_np.mean():.3f}±{neg_np.std():.3f}")

    if not pos_all:
        raise SystemExit("No usable scenes. Check the dataset path / "
                         "correspondence files.")

    pos = np.concatenate(pos_all)
    neg = np.concatenate(neg_all)

    summary = {
        "config": args.config_file,
        "weight": args.weight,
        "scenes_sampled": int(n),
        "scenes_used": int(len(pos_all)),
        "scenes_skipped": int(skipped),
        "total_pairs": int(pos.size),
        "pos_mean": float(pos.mean()),
        "pos_std": float(pos.std()),
        "pos_median": float(np.median(pos)),
        "pos_p05": float(np.percentile(pos, 5)),
        "pos_p95": float(np.percentile(pos, 95)),
        "neg_mean": float(neg.mean()),
        "neg_std": float(neg.std()),
        "neg_median": float(np.median(neg)),
        "neg_p05": float(np.percentile(neg, 5)),
        "neg_p95": float(np.percentile(neg, 95)),
        "discrim_gap": float(pos.mean() - neg.mean()),
    }

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    csv_path = os.path.join(args.out_dir, "per_scene.csv")
    if per_scene_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_scene_rows[0].keys()))
            w.writeheader()
            w.writerows(per_scene_rows)

    fig_path = os.path.join(args.out_dir, "cosine_histogram.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-1.0, 1.0, 100)
    ax.hist(neg, bins=bins, alpha=0.55,
            label=f"neg (shuffle)  μ={summary['neg_mean']:.3f} "
                  f"σ={summary['neg_std']:.3f}",
            color="#c0392b", density=True)
    ax.hist(pos, bins=bins, alpha=0.55,
            label=f"pos (matched)  μ={summary['pos_mean']:.3f} "
                  f"σ={summary['pos_std']:.3f}",
            color="#2c5fa3", density=True)
    ax.axvline(summary["pos_mean"], color="#2c5fa3", ls="--", lw=1)
    ax.axvline(summary["neg_mean"], color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("cosine similarity (3D point feature ↔ 2D patch feature)")
    ax.set_ylabel("density")
    ax.set_title(
        f"Patch-level alignment — {summary['scenes_used']} scenes, "
        f"{summary['total_pairs']} pairs | "
        f"gap = {summary['discrim_gap']:.3f}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote:\n  {summary_path}\n  {csv_path}\n  {fig_path}")


if __name__ == "__main__":
    main()
