"""
Zero-shot 3D semantic segmentation via H/I alignment + SigLIP per-class ROIs.

Question this answers
=====================
Does the qwen_proj / patch_proj alignment that H/I training produced
actually carry useful zero-shot semantic signal at deployment?  The
text-query localization demo uses the alignment only to refine ranking
WITHIN a single bbox-frustum — leaving most of the teacher's effect
unverified.  This tool runs N classes in parallel, picks the argmax-
class per 3D point, and measures mIoU against ScanNet's 20-class
semantic GT.

Pipeline (per scene + image)
============================
1. Per class c:
     SigLIP("{c}", image) → MaskCLIP-modified per-patch cosine
                          → threshold @ p85 → ROI mask on Qwen 16×16 grid
2. Class query (the teacher's effect lives here):
     q_c = qwen_proj( mean( Qwen ViT patches inside ROI_c ) )      # (512,)
3. Per-point label:
     f_p = patch_proj( backbone(scene)_stage1 )[i]                 # (512,)
     label[i] = argmax_c  cosine(f_p, q_c)
4. mIoU vs segment20.npy (if present in scene-dir).

Baseline comparison
===================
--randomize-proj re-initialises patch_proj and qwen_proj with random
weights (backbone stays trained).  Run the tool twice — once with and
once without --randomize-proj — and compare summary.txt mIoUs:

    mIoU(trained) >> mIoU(baseline)  →  teacher's effect is real.
    mIoU(trained) ≈ mIoU(baseline)   →  trained projectors not adding
                                         zero-shot signal; retraining
                                         is justified.

Usage
=====
    # Trained (H/I projectors)
    python demo/zero_shot_3d_semseg.py \\
        --scene-dir   /.../val/scene0011_00 \\
        --image-path  images/val/scene0011_00/color/1200.png \\
        --h-ckpt      exp/utonia_q35_i/model/model_last.pth \\
        --qwen-path   /group-volume/.../Qwen3.5-4B \\
        --out-dir     exp/zeroshot/scene0011_00_trained

    # Baseline (random projectors)
    python demo/zero_shot_3d_semseg.py \\
        ...same args... \\
        --randomize-proj \\
        --out-dir     exp/zeroshot/scene0011_00_baseline

Outputs
=======
    <out-dir>/pred.ply        per-point predicted class (ScanNet palette)
    <out-dir>/gt.ply          per-point GT class (same palette, if available)
    <out-dir>/scene_rgb.ply   original colors (orientation)
    <out-dir>/summary.txt     per-class IoU + mIoU + ROI sizes
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


# Reuse model + projection helpers from the localization demo.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from text_query_to_3d_localization import (  # noqa: E402
    build_h_modules,
    build_qwen,
    build_siglip,
    siglip_grounding_map,
    qwen_vision_full_merger,
    image_to_qwen_tensor,
    utonia_point_features,
    write_ply,
)


# ---------------------------------------------------------------------------
# ScanNet 20-class definitions — names and palette match Pointcept's
# preprocessing/meta_data and the visualization in compare_seg_predictions.py.
# ---------------------------------------------------------------------------
SCANNET20_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator",
    "shower curtain", "toilet", "sink", "bathtub", "other furniture",
]

SCANNET20_PALETTE = np.array(
    [
        [174, 199, 232], [152, 223, 138], [ 31, 119, 180], [255, 187, 120],
        [188, 189,  34], [140,  86,  75], [255, 152, 150], [214,  39,  40],
        [197, 176, 213], [148, 103, 189], [196, 156, 148], [ 23, 190, 207],
        [247, 182, 210], [219, 219, 141], [255, 127,  14], [158, 218, 229],
        [ 44, 160,  44], [112, 128, 144], [227, 119, 194], [ 82,  84, 163],
    ],
    dtype=np.uint8,
)
IGNORE_COLOR = np.array([80, 80, 80], dtype=np.uint8)
IGNORE_INDEX = -1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir",  required=True)
    p.add_argument("--image-path", required=True,
                   help="Image used to build per-class queries via SigLIP+Qwen.")
    p.add_argument("--h-ckpt",     required=True)
    p.add_argument("--qwen-path",  required=True)
    p.add_argument(
        "--siglip-path",
        default=os.environ.get("SIGLIP_PATH", "google/siglip2-base-patch16-256"),
    )
    p.add_argument("--out-dir",    required=True)
    p.add_argument(
        "--classes", default="scannet20",
        help="Either 'scannet20' (default, 20 ScanNet classes) or a comma-"
             "separated list of class names. With a custom list, mIoU "
             "computation skips (GT label space won't match).",
    )
    p.add_argument(
        "--siglip-thr-percentile", type=float, default=85.0,
        help="SigLIP heatmap percentile threshold for per-class ROI build.",
    )
    p.add_argument(
        "--randomize-proj", action="store_true",
        help="Re-initialize patch_proj + qwen_proj with random weights. "
             "Baseline that isolates the teacher's contribution: any mIoU "
             "above this comes from H/I-trained alignment.",
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Semantic GT loading — try several common filenames.  Pointcept's
# concerto-scannet preprocessing dumps coord.npy/color.npy/normal.npy
# alongside segment20.npy (or similar) for the 20-class label space.
# ---------------------------------------------------------------------------
def load_semantic_gt(scene_dir):
    candidates = [
        "segment20.npy", "segment_20.npy",
        "semantic_gt20.npy", "label20.npy",
        "semantic.npy", "segment.npy",
    ]
    for name in candidates:
        p = os.path.join(scene_dir, name)
        if os.path.isfile(p):
            arr = np.load(p)
            print(f"[gt] loaded: {p}  shape={arr.shape}  "
                  f"unique={int(np.unique(arr).size)} values")
            return arr.astype(np.int64), p
    print(f"[gt] no semantic GT found in {scene_dir} (tried: "
          + ", ".join(candidates) + ")")
    return None, None


# ---------------------------------------------------------------------------
# Random-init baseline — replace the H/I-trained patch_proj and qwen_proj
# with freshly-initialized layers of the same shape.  Tests whether the
# zero-shot mIoU we measure actually depends on the trained alignment
# (rather than just the backbone's geometric / semantic signal flowing
# through any random linear map by luck).
# ---------------------------------------------------------------------------
def randomize_projectors(h_modules):
    n_reset = 0
    for mod_name in ["patch_proj", "qwen_proj"]:
        seq = h_modules[mod_name]
        for layer in seq:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
                n_reset += 1
    print(f"[baseline] reset {n_reset} layers in patch_proj + qwen_proj; "
          "the projectors no longer carry H/I training.")


def labels_to_colors(labels):
    """labels: int array, values in [0, 19] or <0 (ignore). → (N, 3) uint8."""
    labels = np.asarray(labels, dtype=np.int64)
    out = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    ignore = labels < 0
    valid = ~ignore
    idx = np.clip(labels[valid], 0, len(SCANNET20_PALETTE) - 1)
    out[valid] = SCANNET20_PALETTE[idx]
    out[ignore] = IGNORE_COLOR
    return out


def per_class_iou(pred, gt, num_classes, ignore_index=IGNORE_INDEX):
    """Standard per-class IoU + mIoU.

    Ignores GT entries equal to `ignore_index`.  Classes absent from GT
    return NaN in the per-class array and are excluded from the mIoU
    mean (so a query that only sees 8 of 20 classes isn't unfairly
    penalised by 12 NaN-as-zero entries).
    """
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    keep = gt != ignore_index
    pred = pred[keep]
    gt = gt[keep]
    ious = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        denom = tp + fp + fn
        if denom > 0:
            ious[c] = tp / denom
    miou = float(np.nanmean(ious))
    return ious, miou


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # --- class list ---------------------------------------------------------
    if args.classes == "scannet20":
        classes = SCANNET20_NAMES
        is_scannet20 = True
    else:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        is_scannet20 = False
    print(f"[setup] {len(classes)} classes; scannet20={is_scannet20}")

    # --- scene + image ------------------------------------------------------
    coord = np.load(os.path.join(args.scene_dir, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(args.scene_dir, "color.npy")).astype(np.uint8)
    normal = np.load(os.path.join(args.scene_dir, "normal.npy")).astype(np.float32)
    gt, gt_path = load_semantic_gt(args.scene_dir)
    print(f"[scene] {args.scene_dir}  N_points={len(coord)}")

    # Resolve image (relative paths against scene_dir or its grandparent —
    # same convention as text_query_to_3d_localization).
    if not os.path.isabs(args.image_path):
        for prefix in [
            args.scene_dir,
            os.path.dirname(os.path.dirname(args.scene_dir.rstrip("/"))),
        ]:
            cand = os.path.join(prefix, args.image_path)
            if os.path.isfile(cand):
                args.image_path = cand
                break
    if not os.path.isfile(args.image_path):
        sys.exit(f"[error] image not found: {args.image_path}")
    print(f"[image] {args.image_path}")

    # --- models -------------------------------------------------------------
    qwen = build_qwen(args.qwen_path, device, keep_llm=False)
    h = build_h_modules(args.h_ckpt, device)
    if args.randomize_proj:
        randomize_projectors(h)
    siglip = build_siglip(args.siglip_path, device)

    # --- image → Qwen 16×16 merged patches (2560-d each) -------------------
    image_tensor, image_pil = image_to_qwen_tensor(args.image_path, 512)
    image_tensor = image_tensor.to(device)
    patch_2d = qwen_vision_full_merger(qwen["visual"], image_tensor, device)
    h_grid, w_grid, D2 = patch_2d.shape
    patch_2d_flat = patch_2d.reshape(-1, D2).float()  # (256, 2560)
    print(f"[2D] Qwen merged: {h_grid}x{w_grid} x {D2}")

    # --- Utonia 3D features → 512-d common space ---------------------------
    feat_s1, inv_s1_to_s0, inv_grid, coord_s1 = utonia_point_features(
        h["backbone"], coord, color, normal, device,
    )
    print(f"[3D] stage-1 feat: {tuple(feat_s1.shape)}   "
          f"(stage-0 N={inv_s1_to_s0.shape[0]}, orig N={inv_grid.shape[0]})")
    if not torch.isfinite(feat_s1).all():
        feat_s1 = torch.nan_to_num(feat_s1, nan=0.0)
    with torch.inference_mode():
        point_common_s1 = h["patch_proj"](feat_s1.float())  # (N_s1, 512)
    if not torch.isfinite(point_common_s1).all():
        point_common_s1 = torch.nan_to_num(point_common_s1, nan=0.0)

    # --- per-class query construction via SigLIP ROI + Qwen patches ---------
    # MaskCLIP's value-only trick is applied inside siglip_grounding_map
    # (first call patches the model in place; subsequent calls are no-ops).
    print(f"[query] building {len(classes)} class queries "
          f"({'TRAINED' if not args.randomize_proj else 'BASELINE'} projectors)")
    queries = []
    valid_class = []
    roi_cells = []
    for c_idx, c_name in enumerate(classes):
        sig_heat, h_sig, w_sig = siglip_grounding_map(
            siglip, image_pil, c_name, device,
        )
        qwen_heat = F.interpolate(
            sig_heat[None, None], size=(h_grid, w_grid),
            mode="bilinear", align_corners=False,
        )[0, 0]  # (h_grid, w_grid)
        thr = torch.quantile(
            qwen_heat.flatten(), args.siglip_thr_percentile / 100.0,
        )
        roi = (qwen_heat >= thr).float().flatten()
        n_cells = int((roi > 0).sum().item())
        roi_cells.append(n_cells)
        if n_cells == 0:
            queries.append(torch.zeros(512, device=device))
            valid_class.append(False)
            print(f"  [{c_idx:2d}] {c_name:18s}  ROI EMPTY  → skipped")
            continue
        weights = roi / roi.sum()
        q_2560 = (weights.unsqueeze(-1) * patch_2d_flat).sum(dim=0)  # (2560,)
        with torch.inference_mode():
            q_512 = h["qwen_proj"](q_2560.unsqueeze(0).float())[0]
        if not torch.isfinite(q_512).all():
            q_512 = torch.nan_to_num(q_512, nan=0.0)
        queries.append(q_512)
        valid_class.append(True)
        print(f"  [{c_idx:2d}] {c_name:18s}  ROI={n_cells:3d}/256")

    queries = torch.stack(queries, dim=0)  # (C, 512)
    valid_class_np = np.array(valid_class, dtype=bool)
    if not valid_class_np.any():
        sys.exit("[fatal] all classes had empty SigLIP ROIs — "
                 "try a different image or lower --siglip-thr-percentile.")

    # --- per-point argmax-cosine across class queries ----------------------
    pcn = F.normalize(point_common_s1.float(), dim=-1, eps=1e-6)
    qcn = F.normalize(queries.float(), dim=-1, eps=1e-6)
    sim = pcn @ qcn.T  # (N_s1, C)
    # Mask invalid classes so argmax never picks them.
    valid_mask_tensor = torch.tensor(
        valid_class_np, dtype=torch.bool, device=device,
    )
    sim = sim.masked_fill(~valid_mask_tensor.unsqueeze(0), float("-inf"))
    pred_s1 = sim.argmax(dim=-1)  # (N_s1,)
    pred_s0 = pred_s1[inv_s1_to_s0]                       # (N_s0,)
    pred_orig = pred_s0[inv_grid].cpu().numpy()           # (N_input,)
    print(f"[pred] per-class point counts: "
          + ", ".join(
              f"{classes[c]}={int((pred_orig == c).sum())}"
              for c in range(len(classes)) if valid_class_np[c]
          ))

    # --- visualizations ----------------------------------------------------
    write_ply(os.path.join(args.out_dir, "scene_rgb.ply"), coord, color)
    pred_colors = labels_to_colors(pred_orig)
    write_ply(os.path.join(args.out_dir, "pred.ply"), coord, pred_colors)
    print(f"[save] pred.ply  scene_rgb.ply")

    # --- mIoU vs GT (if available) ----------------------------------------
    variant = "BASELINE_random_proj" if args.randomize_proj else "TRAINED_H_I"
    summary_lines = [
        f"variant: {variant}",
        f"scene: {args.scene_dir}",
        f"image: {args.image_path}",
        f"classes: {len(classes)} ({'scannet20' if is_scannet20 else 'custom'})",
        f"valid (non-empty ROI): {int(valid_class_np.sum())}/{len(classes)}",
        f"siglip threshold percentile: {args.siglip_thr_percentile}",
        "",
    ]

    if gt is not None and gt.shape[0] == coord.shape[0] and is_scannet20:
        write_ply(
            os.path.join(args.out_dir, "gt.ply"),
            coord, labels_to_colors(gt),
        )
        print("[save] gt.ply")
        ious, miou = per_class_iou(pred_orig, gt, num_classes=len(classes))
        # Headline metric: average over classes that ARE present in this
        # scene's GT.  Listing absent (NaN) classes separately so the
        # denominator is honest.
        n_present = int(np.sum(np.isfinite(ious)))
        summary_lines.append(
            f"mIoU (over {n_present} GT-present classes): {miou:.4f}"
        )
        summary_lines.append("")
        summary_lines.append("per-class IoU:")
        for c_idx, c_name in enumerate(classes):
            v = ious[c_idx]
            roi_note = f"  (ROI={roi_cells[c_idx]:3d})"
            empty_note = "" if valid_class_np[c_idx] else "  [ROI empty]"
            if np.isnan(v):
                summary_lines.append(
                    f"  {c_idx:2d} {c_name:18s}      -   (absent in GT)"
                    f"{roi_note}{empty_note}"
                )
            else:
                summary_lines.append(
                    f"  {c_idx:2d} {c_name:18s}  {v:.4f}{roi_note}{empty_note}"
                )
        print(f"[miou] {variant}: {miou:.4f}  (over {n_present} present classes)")
    else:
        if gt is None:
            summary_lines.append("GT: not available — only PLYs produced.")
        elif gt.shape[0] != coord.shape[0]:
            summary_lines.append(
                f"GT shape mismatch (gt={gt.shape}, coord={coord.shape}) — "
                "skipping mIoU."
            )
        elif not is_scannet20:
            summary_lines.append(
                "Custom class list; cannot match ScanNet20 GT — skipping mIoU."
            )

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"[done] {summary_path}")


if __name__ == "__main__":
    main()
